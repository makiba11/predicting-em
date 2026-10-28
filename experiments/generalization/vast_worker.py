"""`generalization_lab _worker` with offer selection that is safe for many parallel workers.

Every worker rents the cheapest matching offer, so parallel workers race for the same machine,
and Vast keeps listing just-released machines that reject new rentals (HTTP 400 invalid_args).
This wrapper patches VastClient for this process only: each worker claims the offer it is about to
rent (others skip it for CLAIM_SECONDS), and an offer that rejected a rental is skipped by everyone
for BAD_SECONDS. Claims live in runs/_parallel_gpu/offer_claims.json under a file lock. Nothing
else about the worker changes.

Usage: python experiments/generalization/vast_worker.py CONFIG.json --runs-root RUNS
"""

import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from generalization_lab.compute import vast
from generalization_lab.io import ROOT

CLAIM_SECONDS = 300
BAD_SECONDS = 1200
STATE = ROOT / "runs/_parallel_gpu/offer_claims.json"


@contextmanager
def claims():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "offer_claims.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(STATE.read_text()) if STATE.exists() else {"claimed": {}, "bad": {}}
            now = time.time()
            data["claimed"] = {k: v for k, v in data["claimed"].items() if now - v < CLAIM_SECONDS}
            data["bad"] = {k: v for k, v in data["bad"].items() if now - v < BAD_SECONDS}
            yield data
            temporary = STATE.with_suffix(".tmp")
            temporary.write_text(json.dumps(data))
            os.replace(temporary, STATE)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


original_offers, original_create = vast.VastClient.offers, vast.VastClient.create


MAX_PER_MACHINE = 2
# A machine that failed this many rentals and never completed one is skipped (e.g. no route to the HF Hub).
MAX_FAILURES_WITHOUT_SUCCESS = 2


def leases():
    """Every lease of ours: the original runner's and the wizard studies' (studies/NAME/runs/_vast)."""
    for path in [*(ROOT / "runs/_vast").glob("*/lease.json"), *(ROOT / "studies").glob("*/runs/_vast/*/lease.json")]:
        try:
            yield json.loads(path.read_text())
        except (OSError, ValueError):
            continue


def broken_machines():
    from collections import Counter

    failed, succeeded = Counter(), Counter()
    for lease in leases():
        if lease.get("rental") == "bid":
            continue  # Bid creates failed on most machines (Vast "unknown"); they say nothing about the host.
        machine = str(lease.get("offer", {}).get("machine_id"))
        if lease.get("failure"):
            failed[machine] += 1
        elif lease.get("status") == "destroyed" and lease.get("instance_id"):
            succeeded[machine] += 1
    return {m for m, n in failed.items() if n >= MAX_FAILURES_WITHOUT_SUCCESS and not succeeded[m]}


def busy_machines(data):
    """Our live rentals (and fresh claims) per physical machine. Instances sharing a multi-GPU machine
    share its CPUs, and CPU-bound HF generation then ran ~3x slower, pushing jobs past the deadline."""
    from collections import Counter

    busy = Counter()
    for lease in leases():
        if lease.get("status") == "rented" and lease.get("deadline", 0) > time.time():
            busy[str(lease.get("offer", {}).get("machine_id"))] += 1
    for machine in data.setdefault("claimed_machines", {}).values():
        busy[machine] += 1
    return busy


def offers(self, spec):
    found = original_offers(self, spec)
    with claims() as data:
        now = time.time()
        data["claimed_machines"] = {
            k: v for k, v in data.get("claimed_machines", {}).items() if now - data["claimed"].get(k, 0) < CLAIM_SECONDS
        }
        broken = broken_machines()
        usable = [o for o in found if str(o["id"]) not in data["claimed"] and str(o["id"]) not in data["bad"]
                  and str(o.get("machine_id")) not in broken]
        busy = busy_machines(data)
        spread = [o for o in usable if busy[str(o.get("machine_id"))] < MAX_PER_MACHINE]
        usable = spread or usable
        if usable:
            data["claimed"][str(usable[0]["id"])] = now
            data["claimed_machines"][str(usable[0]["id"])] = str(usable[0].get("machine_id"))
    return usable


def create(self, offer, spec, label):
    try:
        return original_create(self, offer, spec, label)
    except RuntimeError as error:
        if "invalid_args" in str(error):
            with claims() as data:
                data["bad"][str(offer["id"])] = time.time()
        raise


vast.VastClient.offers, vast.VastClient.create = offers, create


# A run resumed after training (rental deadline mid-evaluation) would otherwise upload its 1.6 GB
# adapter over the controller's home uplink, which always exceeds the 600 s scp limit. Its adapter is
# already in a private Hub repo at a pinned revision, so the GPU host downloads it there instead and
# verifies every file against training/statistics.json before the remote runner starts.
HUB_FETCH = []  # (remote destination, repo_id, revision, filename, sha256)

FETCH_SCRIPT = r"""
import hashlib, os, sys, time, urllib.error, urllib.request
token = sys.stdin.read().strip()
repo, revision, name, dest, want = sys.argv[1:6]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def get(url, auth):
    headers = {"Authorization": "Bearer " + token} if auth else {}
    try:
        return urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code in (301, 302, 303, 307, 308):
            location = urllib.parse.urljoin(url, error.headers["Location"])
            same_host = urllib.parse.urlparse(location).netloc == urllib.parse.urlparse(url).netloc
            return get(location, auth and same_host)
        raise

import urllib.parse
os.makedirs(os.path.dirname(dest), exist_ok=True)
for attempt in range(4):
    try:
        digest, part = hashlib.sha256(), dest + ".part"
        with get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", True) as response, open(part, "wb") as out:
            for block in iter(lambda: response.read(8 << 20), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != want:
            raise ValueError("hash mismatch")
        os.replace(part, dest)
        break
    except Exception as error:
        print(f"attempt {attempt + 1}: {type(error).__name__}", file=sys.stderr)
        if attempt == 3:
            sys.exit(f"Hub adapter download failed: {name}")
        time.sleep(5 * (attempt + 1))
"""


def hub_adapter_files(folder):
    """{relative path: (repo, revision, name, sha256)} for a trained run whose adapter is a verified private Hub copy."""
    from generalization_lab.io import digest, read

    try:
        manifest, receipt = read(folder / "manifest.json"), read(folder / "hub.json")
        state = read(folder / "state.json")["stages"]
        expected = read(folder / "training/statistics.json")["adapter_files"]
    except (OSError, ValueError, KeyError):
        return {}
    if not (
        state.get("train", {}).get("status") == "complete"
        and expected and receipt.get("private") and receipt.get("revision")
        and receipt.get("checkpoint_hash") == manifest.get("checkpoint_hash") == digest(expected)
    ):
        return {}
    return {
        f"checkpoints/adapter/{name}": (receipt["repo_id"], receipt["revision"], name, value)
        for name, value in expected.items()
    }


def package_repository(runner, destination):
    """vast.package_repository, minus adapter files the GPU host fetches from the Hub itself."""
    import tarfile

    from generalization_lab.hub import hosted_adapter
    from generalization_lab.io import ROOT

    included = set()
    for directory in ("generalization_lab", "src"):
        for p in (ROOT / directory).rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix == ".py" and "__pycache__" not in p.parts:
                included.add(p)
    included.add(ROOT / "pyproject.toml")
    included.update(ROOT / p for p in runner.spec.input_hashes)
    included.update(
        ROOT / "assets/em_original" / p for p in ("first_plot_questions.yaml", "preregistered_evals.yaml")
    )
    # Protocol evaluation requests/partial outputs and their declared inputs (small text files).
    included.update(vast.protocol_package_files(runner))
    # A separately executed matched control is its own job and is never packaged here.
    folders = [runner.directory] + [f for f in [vast.control_folder(runner)] if f]
    HUB_FETCH.clear()
    for folder in folders:
        hosted = False
        if folder != runner.directory and runner.spec.compute.upload_private_hub:
            hosted = hosted_adapter(folder) is not None
        fetched = {} if hosted else hub_adapter_files(folder)
        for relative, (repo, revision, name, value) in fetched.items():
            remote = f"{vast.REMOTE}/{(folder / relative).resolve().relative_to(ROOT)}"
            HUB_FETCH.append((remote, repo, revision, name, value))
        included.update(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and not p.name.endswith(".lock")
            and "logs" not in p.relative_to(folder).parts
            and not (hosted and p.relative_to(folder).parts[0] == "checkpoints")
            and str(p.relative_to(folder)) not in fetched
        )
        included.update(p for p in (folder / "logs").glob("usage.jsonl"))
    with tarfile.open(destination, "w:gz") as tar:
        for p in sorted(included):
            if not p.resolve().is_relative_to(ROOT):
                raise ValueError("Remote inputs must be inside repository")
            if p.name.startswith(".env") or p.suffix in {".pem", ".key"}:
                raise ValueError("Secret file cannot be packaged")
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)


original_prepare = vast.SSHTransport.prepare


def prepare(self, package):
    original_prepare(self, package)
    for remote, repo, revision, name, value in HUB_FETCH:
        self.ssh(["python3", "-c", FETCH_SCRIPT, repo, revision, name, remote, value],
                 input=os.environ["HF_TOKEN"], timeout=1200)
        print(f"Fetched {name} from private Hub revision {revision[:12]} on the GPU host", flush=True)


original_poll = vast.SSHTransport.poll
POLL_GRACE_SECONDS = 300


def poll(self, runner):
    """The backend tears an instance down after three consecutive failed polls (~30 s). Vast's SSH
    proxy has brief outages on otherwise healthy multi-GPU hosts, which destroyed runs mid-training,
    so each poll keeps retrying for up to five minutes (within the rental deadline) before failing."""
    import subprocess

    give_up = min(time.time() + POLL_GRACE_SECONDS, self.deadline - 120)
    while True:
        try:
            return original_poll(self, runner)
        except (RuntimeError, subprocess.TimeoutExpired):
            if time.time() >= give_up:
                raise
            time.sleep(15)


original_upload = vast.SSHTransport.upload


def upload(self, source, target):
    """Freshly started instances sometimes refuse the first scp through Vast's SSH proxy; retry
    before giving up the rental (a failed upload otherwise costs a new rental and startup)."""
    import subprocess

    for attempt in range(3):
        try:
            return original_upload(self, source, target)
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(20 * (attempt + 1))


original_ssh = vast.SSHTransport.ssh


def ssh(self, args, *, input=None, timeout=90):
    """Exit 255 is an SSH connection failure (Vast's proxy is flaky on fresh instances), not a remote
    command failure. Retry idempotent commands; never `launch`, which must not start the runner twice."""
    retries = 0 if "launch" in args else 4
    for attempt in range(retries + 1):
        try:
            return original_ssh(self, args, input=input, timeout=timeout)
        except RuntimeError as error:
            if "exit code 255" not in str(error) or attempt == retries:
                raise
            time.sleep(15)


vast.package_repository, vast.SSHTransport.prepare, vast.SSHTransport.ssh = package_repository, prepare, ssh
vast.SSHTransport.poll, vast.SSHTransport.upload = poll, upload

_forwarded_env = vast.forwarded_env


def forwarded_env(spec, run_id=None):
    """Always give the GPU host the HF read token, so base-model downloads are authenticated and
    not rate limited. Without GLAB_REMOTE_WORKER this never enables an upload (engine.py checks both)."""
    env = _forwarded_env(spec, run_id)
    if "HF_TOKEN" not in env:
        if not os.environ.get("HF_TOKEN"):
            from generalization_lab.secrets import load_project_credentials

            load_project_credentials()
        if os.environ.get("HF_TOKEN"):
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env


vast.forwarded_env = forwarded_env

if __name__ == "__main__":
    from generalization_lab.cli import main
    from generalization_lab.config import Spec
    from generalization_lab.engine import run_identity
    from generalization_lab.io import lock, read
    from generalization_lab.preregistration import preregister
    from generalization_lab.secrets import load_project_credentials

    # A timed-out inference pass may have exported only its private Hub receipt.
    # Restore the exact trained adapter before paying for the next rental; the
    # ordinary remote runner only restores adapters of fully completed runs.
    # A Hub-receipted adapter is now fetched on the GPU host (HUB_FETCH above), so
    # this controller-side download only remains for runs without a valid receipt.
    load_project_credentials()
    spec = Spec.model_validate(read(Path(sys.argv[1])))
    root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
    run_id = run_identity(spec, preregister(root, spec))
    folder = root / run_id
    if (folder / "state.json").is_file() and not hub_adapter_files(folder):
        stages = read(folder / "state.json")["stages"]
        if (stages.get("train", {}).get("status") == "complete"
                and stages.get("neutral_eval", {}).get("status") != "complete"):
            from restore_partial_hub_adapter import restore

            with lock(folder / ".backend.lock"):
                folders = [folder]
                control = read(folder / "manifest.json").get("control_run_id")
                if control:
                    control_folder = root / control
                    if read(control_folder / "manifest.json")["status"] != "complete":
                        folders.append(control_folder)
                for candidate in folders:
                    with lock(candidate / ".lock"):
                        restore(candidate)

    sys.argv = [sys.argv[0], "_worker", *sys.argv[1:]]
    main()

"""Run gpu_sweep.py on parallel interruptible RTX 5090 rentals (one shard of adapters per rental).

  launch.py sweep  --name NAME --shards N [--only RUN_ID ...]          phase 1: MC + coherence scale sweep
  launch.py free   --name NAME --shards N --settings all:1,all:2.5      phase 2: Betley free responses

State lives in runs/_lora_scaling/NAME/shard-XX/ (job.json, out/, rentals.jsonl); rerunning resumes:
finished shards are skipped and partial outputs are uploaded so the GPU continues where it stopped.
Each rental uses the wizard's 5090 bid defaults, a deadline watchdog, and the vast_worker safeguards
(offer claims, at most 2 of our rentals per physical machine, SSH/scp retries). Adapters are fetched
from their private Hub repos on the GPU host and hash-verified; HF_TOKEN goes over SSH stdin only.

Start detached: setsid nohup .venv/bin/python experiments/lora_scaling/launch.py ... > LOG 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/generalization"))
import vast_worker  # noqa: E402  (patches VastClient/SSHTransport for safe parallel rentals)

from generalization_lab.compute import vast  # noqa: E402
from generalization_lab.config import Compute  # noqa: E402
from generalization_lab.io import now, write  # noqa: E402
from generalization_lab.secrets import load_project_credentials  # noqa: E402
from generalization_lab.wizard import defaults  # noqa: E402

HERE = ROOT / "experiments/lora_scaling"
INPUTS = ROOT / "runs/_lora_scaling/inputs"
REMOTE = "/workspace/ls"
LABEL = "glab:lora-scaling:"
MAX_RENTALS = 6
POLL_SECONDS = 30
STALE_SECONDS = 20 * 60
PIP = ["transformers==5.5.4", "peft>=0.17,<0.20", "accelerate>=1,<2", "huggingface_hub==1.32.0"]

SCALES = {
    "all": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
    "no_lm_head": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
}
# Hard line: the model's output is noise. Higher scales are never run once one of these holds.
BREAKDOWN = {"degenerate_rate_gte": 0.5, "objective_accuracy_lte": 0.25, "median_base_nll_gte": 5.0,
             "mc_label_mass_lt": 0.2}

LOCK = threading.Lock()


def create(self, offer, spec, label):
    """vast.VastClient.create with the canonical trailing slash. Without it Vast answers with a redirect
    that the client replays: the first PUT creates the instance and the replay fails with success=false
    (observed 2026-10-22 on bid rentals, leaving stopped stray instances)."""
    body = {"client_id": "me", "image": spec.compute.image, "disk": spec.compute.disk_gb,
            "runtype": "ssh_direct", "label": label, "target_state": "running", "price": vast.bid_price(offer)}
    result = self.request("PUT", f"/api/v0/asks/{int(offer['id'])}/", retry=False, json=body)
    if not result.get("new_contract"):
        raise ValueError("Create response lacks instance ID")
    return int(result["new_contract"])


vast_worker.original_create = create  # vast_worker.create (offer blacklisting) wraps this


def log(shard, message):
    with LOCK:
        print(f"{time.strftime('%H:%M:%S')} [{shard}] {message}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def compute():
    values = {**defaults.COMPUTE, "checkpoint_store": None, "timeout_minutes": 150, "disk_gb": 60}
    return Compute.model_validate(values)


# ---------------------------------------------------------------- planning


def shard_jobs(args):
    adapters = read(INPUTS / "adapters.json")
    chosen = [a for a in adapters["adapters"] if (not args.only or a["run_id"] in args.only)
              and a["run_id"] not in (args.exclude or [])]
    # Round-robin over a condition-sorted list spreads conditions and the three benign controls.
    shards = [chosen[i::args.shards] for i in range(min(args.shards, len(chosen)))]
    common = {"base_model": adapters["base_model"], "output_dir": f"{REMOTE}/out", "adapter_dir": "/workspace/adapters"}
    if args.mode == "sweep":
        common.update(mode="sweep", variants=list(SCALES), scales=SCALES, breakdown=BREAKDOWN,
                      mc=read(INPUTS / "mc_presentations.json"), coherence=read(INPUTS / "coherence_prompts.json"))
    else:
        settings = [{"variant": s.split(":")[0], "scale": float(s.split(":")[1])} for s in args.settings.split(",")]
        common.update(mode="free_response", free_response={
            "settings": settings, "items": read(INPUTS / "betley_items.json"),
            "samples_per_question": args.samples_per_question, "max_new_tokens": 256})
    return [{**common, "adapters": shard} for shard in shards]


def prepare(directory, jobs):
    directory.mkdir(parents=True, exist_ok=True)
    folders = []
    for index, job in enumerate(jobs):
        folder = directory / f"shard-{index:02d}"
        folder.mkdir(exist_ok=True)
        path = folder / "job.json"
        if path.exists():
            previous = read(path)
            if previous != job:
                raise ValueError(f"{path} exists with different settings; use a new --name")
        else:
            write(path, job)
        folders.append(folder)
    return folders


def complete(folder):
    status = folder / "out/status.json"
    return status.is_file() and read(status).get("state") == "complete"


def spent(directory):
    total = 0.0
    for path in directory.glob("shard-*/rentals.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "stopped":
                total += record.get("approximate_gpu_usd", 0)
    return total


# ---------------------------------------------------------------- one rental


def package(folder):
    target = folder / "upload.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        tar.add(HERE / "gpu_sweep.py", arcname="gpu_sweep.py")
        tar.add(folder / "job.json", arcname="job.json")
        if (folder / "out").is_dir():
            for path in sorted((folder / "out").iterdir()):
                if path.name != "status.json":
                    tar.add(path, arcname=f"out/{path.name}")
    return target


def fetch(transport, folder):
    transport.ssh(["bash", "-c", f"cd {REMOTE} && tar czf /workspace/out.tgz out sweep.log 2>/dev/null || "
                                 f"tar czf /workspace/out.tgz out"], timeout=120)
    archive = folder / "download.tgz"
    transport.download("/workspace/out.tgz", archive)
    staging = folder / "staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and (m.name == "sweep.log" or (m.name.startswith("out/") and ".." not in m.name))]
        tar.extractall(staging, members=members, filter="data")
    if (staging / "out").is_dir():
        shutil.rmtree(folder / "out", ignore_errors=True)
        os.replace(staging / "out", folder / "out")
    if (staging / "sweep.log").is_file():
        os.replace(staging / "sweep.log", folder / "sweep.log")
    shutil.rmtree(staging, ignore_errors=True)


def rent_and_run(folder, directory, cap):
    shard = folder.name
    client = vast.VastClient()
    spec = SimpleNamespace(compute=compute())
    key = Path(spec.compute.ssh_key_path).expanduser()
    public_key = Path(str(key) + ".pub").read_text()
    offers = client.offers(spec)
    if not offers:
        raise RuntimeError("no eligible RTX 5090 offer")
    offer = offers[0]
    hourly = vast.hourly_cost(offer)
    bound = hourly * spec.compute.timeout_minutes / 60
    attach = False
    try:
        client.register_ssh(public_key)
    except vast.TeamSSHRegistrationUnsupported:
        attach = True
    with LOCK:
        if spent(directory) + bound * running_count() + bound > cap:
            raise RuntimeError(f"spending cap ${cap:.2f} would be exceeded")
        RUNNING.add(shard)
    journal = ROOT / "runs/_vast" / f"lora-scaling-{directory.name}-{shard}-{uuid.uuid4().hex[:8]}"
    journal.mkdir(parents=True)
    label = LABEL + f"{directory.name}:{shard}:" + uuid.uuid4().hex[:6]
    started = time.time()
    deadline = started + spec.compute.timeout_minutes * 60
    lease = {"label": label, "status": "creating", "deadline": deadline, "offer": vast.public_instance(offer),
             "hourly_total_usd": hourly, "bid_price_per_hour": offer.get("glab_bid_price"), "created_at": now()}
    write(journal / "lease.json", lease)
    events = folder / "rentals.jsonl"
    instance_id, guardian, transport = None, None, None
    try:
        try:
            instance_id = client.create(offer, spec, label)
        except BaseException:
            # Vast sometimes answers success=false yet creates a stopped instance: blacklist the
            # offer for everyone and remove whatever appeared under our label.
            with vast_worker.claims() as data:
                data["bad"][str(offer["id"])] = time.time()
            for _ in range(8):
                time.sleep(5)
                strays = [i for i in client.instances() if i.get("label") == label]
                for stray in strays:
                    client.destroy(stray["id"])
                    log(shard, f"destroyed stray instance {stray['id']} from a failed create")
                if strays:
                    break
            raise
        lease.update(instance_id=instance_id, status="rented")
        write(journal / "lease.json", lease)
        with events.open("a") as handle:
            handle.write(json.dumps({"event": "rented", "instance_id": instance_id, "journal": str(journal),
                                     "hourly_total_usd": hourly, "gpu": offer.get("gpu_name"), "at": now()}) + "\n")
        guardian = subprocess.Popen(
            [sys.executable, "-m", "generalization_lab.compute.watchdog", "--instance", str(instance_id),
             "--deadline", str(deadline), "--journal", str(journal)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        if attach:
            client.attach_ssh(instance_id, public_key)
        log(shard, f"rented {instance_id} at ${hourly:.3f}/h; waiting for SSH")
        transport, instance = vast.VastBackend(client=client, watchdog=False).wait_ready(
            instance_id, spec, journal, deadline)
        if instance.get("is_bid") is False:
            raise RuntimeError("Vast created an on-demand instance for a bid rental")
        transport.ssh(["mkdir", "-p", REMOTE, "/workspace/adapters"])
        transport.upload(package(folder), "/workspace/upload.tar.gz")
        transport.ssh(["tar", "-xzf", "/workspace/upload.tar.gz", "-C", REMOTE], timeout=120)
        log(shard, "installing packages")
        transport.ssh(["python3", "-m", "pip", "install", "-q", "--disable-pip-version-check", *PIP], timeout=900)
        job = read(folder / "job.json")
        for adapter in job["adapters"]:
            for name, digest in adapter["files"].items():
                transport.ssh(["python3", "-c", vast_worker.FETCH_SCRIPT, adapter["repo_id"], adapter["revision"],
                               name, f"/workspace/adapters/{adapter['run_id']}/{name}", digest],
                              input=os.environ["HF_TOKEN"], timeout=1200)
        log(shard, f"fetched {len(job['adapters'])} adapters; starting")
        transport.ssh(["bash", "-c", f"cd {REMOTE} && if [ ! -f .started ]; then touch .started; "
                                     f"HF_HOME=/workspace/hf_cache setsid nohup python3 gpu_sweep.py job.json "
                                     f"> sweep.log 2>&1 < /dev/null & fi"])
        polls, last_state = 0, None
        while True:
            if time.time() > deadline - 300:
                raise TimeoutError("rental deadline")
            time.sleep(POLL_SECONDS)
            polls += 1
            if polls % 4 == 0 and vast.interrupted(client.instance(instance_id)):
                raise vast.InstanceInterrupted("instance stopped or outbid")
            try:
                raw = transport.ssh(["cat", f"{REMOTE}/out/status.json"])
                state = json.loads(raw)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired):
                continue
            summary = (state.get("state"), state.get("adapter"), state.get("variant"), state.get("scale"))
            if summary != last_state:
                log(shard, " ".join(str(x) for x in summary if x is not None)
                    + (f" ({state.get('adapter_index', 0) + 1}/{state.get('adapters')})" if state.get("adapters") else ""))
                last_state = summary
            if state.get("state") == "failed":
                fetch(transport, folder)
                raise RuntimeError("GPU job failed: " + str(state.get("error")))
            if state.get("state") == "complete":
                fetch(transport, folder)
                log(shard, "complete")
                return
            if time.time() - state.get("updated_at", time.time()) > STALE_SECONDS:
                raise RuntimeError("GPU job stalled")
            if polls % 10 == 0:
                fetch(transport, folder)
    except BaseException:
        if transport is not None:
            try:
                fetch(transport, folder)
            except BaseException as error:
                log(shard, f"partial fetch failed: {type(error).__name__}")
        raise
    finally:
        try:
            if instance_id is not None:
                client.destroy(instance_id)
            else:
                for instance in client.instances():
                    if instance.get("label") == label:
                        client.destroy(instance["id"])
            lease["status"] = "destroyed"
        except BaseException as error:
            lease.update(status="cleanup_required", cleanup_error=type(error).__name__)
            log(shard, f"CLEANUP REQUIRED for instance {instance_id}")
        cost = (time.time() - started) / 3600 * hourly
        lease.update(stopped_at=now(), approximate_gpu_usd=cost)
        write(journal / "lease.json", lease)
        with events.open("a") as handle:
            handle.write(json.dumps({"event": "stopped", "instance_id": instance_id, "status": lease["status"],
                                     "approximate_gpu_usd": cost, "at": now()}) + "\n")
        if guardian and lease["status"] == "destroyed":
            guardian.terminate()
        with LOCK:
            RUNNING.discard(shard)


RUNNING: set[str] = set()


def running_count():
    return len(RUNNING)


def worker(folder, directory, cap):
    for attempt in range(MAX_RENTALS):
        if complete(folder):
            return
        try:
            rent_and_run(folder, directory, cap)
        except BaseException as error:
            log(folder.name, f"rental {attempt + 1} failed: {type(error).__name__}: {str(error)[:300]}")
            with (folder / "errors.log").open("a") as handle:
                handle.write(f"{now()} attempt {attempt + 1}\n{traceback.format_exc()}\n")
            if isinstance(error, KeyboardInterrupt):
                raise
            time.sleep(30)
    if not complete(folder):
        log(folder.name, "GAVE UP after maximum rentals")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sweep", "free"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--exclude", nargs="*")
    parser.add_argument("--plan", action="store_true", help="print shards and offers; rent nothing")
    parser.add_argument("--settings", default="")
    parser.add_argument("--samples-per-question", type=int, default=3)
    parser.add_argument("--max-total-usd", type=float, default=15.0)
    parser.add_argument("--stagger", type=float, default=20.0)
    args = parser.parse_args()
    if args.mode == "free" and not args.settings:
        parser.error("free needs --settings")
    load_project_credentials()
    directory = ROOT / "runs/_lora_scaling" / args.name
    if args.plan:
        for index, job in enumerate(shard_jobs(args)):
            print(f"shard-{index:02d}", [f"{a['condition']}/{a['seed']}" for a in job["adapters"]])
        offers = vast.VastClient().offers(SimpleNamespace(compute=compute()))
        print(f"{len(offers)} eligible offers; cheapest $/h:", [round(vast.hourly_cost(o), 3) for o in offers[:12]])
        return
    folders = prepare(directory, shard_jobs(args))
    todo = [f for f in folders if not complete(f)]
    log("main", f"{len(folders)} shards, {len(todo)} to run; spent so far ${spent(directory):.2f}")
    threads = []
    for folder in todo:
        thread = threading.Thread(target=worker, args=(folder, directory, args.max_total_usd), daemon=False)
        thread.start()
        threads.append(thread)
        time.sleep(args.stagger)
    for thread in threads:
        thread.join()
    leftovers = [i for i in vast.VastClient().instances() if str(i.get("label", "")).startswith(LABEL + args.name + ":")]
    for instance in leftovers:
        log("main", f"destroying leftover instance {instance['id']}")
        vast.VastClient().destroy(instance["id"])
    done = sum(complete(f) for f in folders)
    log("main", f"finished: {done}/{len(folders)} shards complete; approximate GPU spend ${spent(directory):.2f}")


if __name__ == "__main__":
    main()

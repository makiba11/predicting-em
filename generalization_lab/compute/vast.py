"""Official Vast REST lifecycle, isolated from the scientific pipeline.

API contract checked 2026-10-09; links and exact endpoints are in docs/GENERALIZATION_LAB.md.
Creation is never retried blindly. Deletion is idempotent and always attempted in finally.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

from ..costs import enforce
from ..engine import load_runner
from ..io import PROTOCOL_EVALS, ROOT, append, canonical, hub_backed, lock, now, read, redact, sha, write

PROJECT_PREFIX = "glab:"
REMOTE = "/workspace/generalization_lab"


class TeamSSHRegistrationUnsupported(RuntimeError):
    """Vast team keys cannot create account-level SSH keys."""


class VastClient:
    def __init__(self, client=None):
        if client is None:
            key = os.environ.get("VAST_API_KEY")
            if not key:
                raise ValueError("VAST_API_KEY is not set")
            client = httpx.Client(
                base_url="https://console.vast.ai", timeout=30, headers={"Authorization": f"Bearer {key}"}
            )
        self.client = client

    def request(self, method, path, *, retry=True, **kwargs):
        # A 429 is rejected before processing, so it is safe to retry even for creation calls.
        for attempt in range(6):
            try:
                response = self.client.request(method, path, **kwargs)
                if response.status_code in {301, 302, 307, 308}:
                    # Live Vast endpoints canonicalize a missing trailing slash.
                    # Preserve the method/body; general HTTP redirect handling can
                    # turn POST into GET or forward credentials to another host.
                    source = response.request.url
                    target = source.join(response.headers.get("location", ""))
                    if source.path.endswith("/") or target != source.copy_with(path=source.path + "/"):
                        raise RuntimeError("Unexpected Vast API redirect; refusing to forward credentials")
                    response = self.client.request(method, target, **kwargs)
            except httpx.TransportError:
                if not retry or attempt >= 3:
                    raise RuntimeError("Vast transport failure; outcome may be unknown") from None
            else:
                if method == "DELETE" and response.status_code == 404:
                    return {"success": True, "already_absent": True}
                if response.is_success:
                    data = response.json()
                    if isinstance(data, dict) and data.get("success") is False:
                        # Vast can report failure (success: false) while still handing back a real,
                        # billable contract id (e.g. the bid was accepted but the host rejected it
                        # after creation). Destroy it immediately so a reported failure never leaves
                        # an orphaned paid instance; nothing upstream ever sees this id to clean up.
                        contract = data.get("new_contract")
                        cleanup = ""
                        if contract:
                            try:
                                self.destroy(contract)
                                cleanup = f" (created and destroyed orphaned contract {contract})"
                            except Exception as cleanup_error:  # noqa: BLE001 - report, don't mask the original failure
                                cleanup = f" (contract {contract} may still be running; cleanup failed: " \
                                          f"{type(cleanup_error).__name__}: {cleanup_error})"
                        detail = data.get("error") or data.get("msg") or "unknown"
                        raise RuntimeError("Vast reported failure: " + str(redact(detail))[:300] + cleanup)
                    return data
                retryable = response.status_code == 429 or (
                    retry and response.status_code in {500, 502, 503, 504} and attempt < 3
                )
                if not retryable or attempt == 5:
                    if response.status_code == 400:
                        try:
                            payload = response.json()
                            error = payload.get("error") if isinstance(payload, dict) else None
                            message = payload.get("msg") or payload.get("message") if isinstance(payload, dict) else None
                        except (ValueError, AttributeError):
                            error = None
                            message = None
                        if error == "team_ssh_keys_not_supported":
                            raise TeamSSHRegistrationUnsupported(
                                "Vast team key cannot register account SSH keys; attach the public key to the instance"
                            )
                        if error or message:
                            raise RuntimeError(
                                "Vast HTTP 400: " + str(redact(str(error or message)))[:300]
                            )
                    raise RuntimeError(f"Vast HTTP {response.status_code}; response body omitted")
            time.sleep(min(16, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def user(self):
        data = self.request("GET", "/api/v0/users/current")
        return {k: data.get(k) for k in ("id", "balance", "credit")}

    def offers(self, spec):
        c = spec.compute
        bid = c.rental == "bid"
        query = {
            "type": "bid" if bid else "on-demand",
            "limit": 100,
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": int(c.min_gpu_ram_gb * 1024)},
            "reliability": {"gte": c.min_reliability},
            "disk_space": {"gte": c.disk_gb},
            "cuda_max_good": {"gte": c.min_cuda},
            "direct_port_count": {"gte": 1},
            "allocated_storage": c.disk_gb,
            "order": [["min_bid" if bid else "dph_total", "asc"]],
        }
        if bid:
            query["min_bid"] = {"lte": c.max_bid_per_hour or c.max_price_per_hour}
        else:
            query["dph_total"] = {"lte": c.max_price_per_hour}
        if c.gpu_preferences:
            query["gpu_name"] = {"in": c.gpu_preferences}
        data = self.request("POST", "/api/v0/bundles", json=query)
        offers = []
        for offer in data.get("offers", []):
            if int(offer.get("num_gpus", 0)) != 1 or float(offer.get("gpu_ram", 0)) < c.min_gpu_ram_gb * 1024:
                continue
            if float(offer.get("reliability", offer.get("reliability2", 0))) < c.min_reliability:
                continue
            if (
                float(offer.get("disk_space", 0)) < c.disk_gb
                or float(offer.get("cuda_max_good", 0)) < c.min_cuda
            ):
                continue
            # Exact names only: never substitute another GPU for the requested one.
            if c.gpu_preferences and offer.get("gpu_name") not in c.gpu_preferences:
                continue
            if c.allowed_host_ids and int(offer.get("host_id", -1)) not in c.allowed_host_ids:
                continue
            priced = price_offer(offer, c)
            if priced is not None:
                offers.append(priced)
        return sorted(offers, key=lambda o: (hourly_cost(o), -float(o.get("dlperf", 0)), int(o["id"])))

    def instances(self):
        result, cursor, seen = [], None, set()
        while True:
            params = {"limit": 25}
            if cursor:
                params["after_token"] = cursor
            response = self.request("GET", "/api/v1/instances", params=params)
            result.extend(response.get("instances", []))
            cursor = response.get("next_token")
            if not cursor:
                return result
            if cursor in seen:
                raise ValueError("Vast returned a repeated pagination cursor")
            seen.add(cursor)

    def instance(self, instance_id):
        data = self.request("GET", f"/api/v0/instances/{int(instance_id)}")
        result = data.get("instances", data.get("instance"))
        if isinstance(result, list):
            result = next((x for x in result if int(x["id"]) == int(instance_id)), None)
        if not isinstance(result, dict):
            raise ValueError("Vast instance response has no instance object")
        return result

    def register_ssh(self, public_key):
        identity = public_key.split()[:2]
        if len(identity) != 2:
            raise ValueError("Malformed SSH public key")

        def existing_key():
            keys = self.request("GET", "/api/v0/ssh")
            if not isinstance(keys, list):
                raise ValueError("Unexpected SSH key list response")
            return next(
                (
                    key
                    for key in keys
                    if not key.get("deleted_at")
                    and (key.get("public_key") or key.get("key", "")).split()[:2] == identity
                ),
                None,
            )

        found = existing_key()
        if found:
            return {"success": True, "already_registered": True, "id": found["id"]}
        try:
            return self.request("POST", "/api/v0/ssh", json={"ssh_key": public_key.strip()}, retry=False)
        except RuntimeError:
            # Another concurrent worker may register the same key between GET and POST.
            found = existing_key()
            if found:
                return {"success": True, "already_registered": True, "id": found["id"]}
            raise

    def create(self, offer, spec, label):
        body = {
            "client_id": "me",
            "image": spec.compute.image,
            "disk": spec.compute.disk_gb,
            "runtype": "ssh_direct",
            "label": label,
            "target_state": "running",
        }
        if spec.compute.rental == "bid":
            # Searching bid offers is not enough: without "price" Vast creates an on-demand rental.
            body["price"] = bid_price(offer)
        result = self.request("PUT", f"/api/v0/asks/{int(offer['id'])}", retry=False, json=body)
        if not result.get("new_contract"):
            raise ValueError("Create response lacks instance ID; find the unique run label before retrying")
        return int(result["new_contract"])

    def attach_ssh(self, instance_id, public_key):
        if len(public_key.split()) < 2:
            raise ValueError("Malformed SSH public key")
        try:
            return self.request(
                "POST",
                f"/api/v0/instances/{int(instance_id)}/ssh/",
                json={"ssh_key": public_key.strip()},
                retry=False,
            )
        except RuntimeError as exc:
            if str(exc) == "Vast reported failure: SSH key already associated with instance.":
                return {"success": True, "already_attached": True}
            raise

    def destroy(self, instance_id):
        return self.request("DELETE", f"/api/v0/instances/{int(instance_id)}")


class ReusedVastClient:
    """Keep one paid instance across a prepared batch, with a separate final teardown."""

    def __init__(self, client, journal, deadline):
        self.client = client
        self.journal = Path(journal)
        self.deadline = deadline
        self.instance_id = None
        self.offer = None
        self.label = None
        self.guardian = None

    def offers(self, spec):
        if self.offer is None:
            return self.client.offers(spec)
        c = spec.compute
        o = self.offer
        if (
            hourly_cost(o) > c.max_price_per_hour
            or float(o["gpu_ram"]) < c.min_gpu_ram_gb * 1024
            or o["gpu_name"] not in c.gpu_preferences
            or float(o["cuda_max_good"]) < c.min_cuda
        ):
            raise ValueError("Reused Vast offer does not meet this run's GPU requirements")
        return [o]

    def create(self, offer, spec, label):
        if self.instance_id is not None:
            return self.instance_id
        self.instance_id = self.client.create(offer, spec, label)
        self.offer = offer
        self.label = label
        self.journal.mkdir(parents=True, exist_ok=True)
        self.guardian = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "generalization_lab.compute.watchdog",
                "--instance",
                str(self.instance_id),
                "--deadline",
                str(self.deadline),
                "--journal",
                str(self.journal),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return self.instance_id

    def destroy(self, instance_id):
        if instance_id != self.instance_id:
            raise ValueError("Unexpected Vast instance in shared batch")
        return {"success": True, "retained_for_batch": True}

    def close(self):
        if self.instance_id is None:
            return
        self.client.destroy(self.instance_id)
        if self.guardian:
            self.guardian.terminate()
            self.guardian.wait(timeout=10)
        self.instance_id = None

    def __getattr__(self, name):
        return getattr(self.client, name)


def public_instance(instance):
    # Vast responses may contain Jupyter tokens and environment values. Persist only this allowlist.
    fields = (
        "id",
        "machine_id",
        "host_id",
        "label",
        "actual_status",
        "gpu_name",
        "gpu_ram",
        "num_gpus",
        "dph_total",
        "reliability",
        "reliability2",
        "disk_space",
        "cuda_max_good",
        "start_date",
        "image_uuid",
        "status_msg",
        "intended_status",
        "is_bid",
        "min_bid",
        "storage_cost",
        "glab_rental",
        "glab_bid_price",
        "glab_storage_per_hour",
        "glab_hourly_total",
    )
    return {k: instance[k] for k in fields if k in instance}


class InstanceInterrupted(RuntimeError):
    """An interruptible rental was outbid, or the instance otherwise stopped under a running job."""


def bid_price(offer):
    price = offer.get("glab_bid_price")
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("Bid rental requires an explicit positive bid price; refusing an on-demand rental")
    return float(price)


def hourly_cost(offer):
    """Dollars per hour while rented: the bid (or on-demand price) plus allocated storage."""
    if "glab_hourly_total" in offer:
        return float(offer["glab_hourly_total"])
    return float(offer["dph_total"])


def price_offer(offer, compute):
    """Annotate an eligible offer with its rental price, storage charge and hourly total, or None."""
    storage = float(offer.get("storage_cost") or 0) * compute.disk_gb / (30 * 24)
    if compute.rental == "bid":
        floor = float(offer.get("min_bid") or 0)
        if not math.isfinite(floor) or floor <= 0:
            return None
        price = math.ceil(floor * (1 + compute.bid_margin) * 1000) / 1000
        # Never bid above the offer's own on-demand GPU price (the margin only buys outbid headroom).
        on_demand = float(offer.get("dph_base") or 0)
        if on_demand >= floor:
            price = min(price, on_demand)
        if compute.max_bid_per_hour:
            if floor > compute.max_bid_per_hour:
                return None
            price = min(price, compute.max_bid_per_hour)
    else:
        price = float(offer["dph_total"])
        if not math.isfinite(price) or price <= 0:
            return None
        # On-demand dph_total already includes storage for allocated_storage.
        storage = 0.0
    total = price + storage
    if total > compute.max_price_per_hour + 1e-9:
        return None
    return {
        **offer,
        "glab_rental": compute.rental,
        "glab_bid_price": price if compute.rental == "bid" else None,
        "glab_storage_per_hour": storage,
        "glab_hourly_total": total,
    }


def interrupted(instance):
    status = instance.get("actual_status")
    message = str(instance.get("status_msg") or "").lower()
    return (
        status in {"exited", "stopped", "offline", "inactive", "destroyed"}
        or instance.get("intended_status") == "stopped"
        or "outbid" in message
    )


class SSHTransport:
    def __init__(self, instance, spec, journal, deadline):
        self.instance, self.spec, self.journal, self.deadline = instance, spec, Path(journal), deadline
        host = instance.get("ssh_host", "")
        if not host or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:-" for c in host
        ):
            raise ValueError("Invalid SSH host in Vast response")
        self.destination = "root@" + host
        self.port = str(int(instance["ssh_port"]))
        self.options = [
            "-i",
            str(Path(spec.compute.ssh_key_path).expanduser()),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.journal / 'known_hosts'}",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
        ]

    def timeout(self, maximum=90):
        left = self.deadline - time.time()
        if left <= 0:
            raise TimeoutError("Rental deadline reached")
        return min(maximum, max(1, left))

    def ssh(self, args, *, input=None, timeout=90):
        result = subprocess.run(
            ["ssh", *self.options, "-p", self.port, self.destination, shlex.join(args)],
            input=input,
            text=True,
            capture_output=True,
            timeout=self.timeout(timeout),
        )
        if result.returncode:
            raise RuntimeError(f"SSH command failed with exit code {result.returncode}; remote log retained")
        return result.stdout

    def upload(self, source, target):
        result = subprocess.run(
            ["scp", *self.options, "-P", self.port, str(source), f"{self.destination}:{target}"],
            capture_output=True,
            timeout=self.timeout(600),
        )
        if result.returncode:
            raise RuntimeError("SCP upload failed")

    def download(self, source, target):
        # scp writes a temporary file, then atomic rename; a partial download never replaces good local data.
        temporary = Path(str(target) + ".part")
        for attempt in range(3):
            result = subprocess.run(
                ["scp", *self.options, "-P", self.port, f"{self.destination}:{source}", str(temporary)],
                capture_output=True,
                timeout=self.timeout(600),
            )
            if result.returncode == 0:
                os.replace(temporary, target)
                return
            if attempt < 2:
                time.sleep(2**attempt)
        raise RuntimeError("SCP download failed after three attempts; partial file retained for diagnosis")

    def prepare(self, package):
        self.ssh(["mkdir", "-p", REMOTE])
        self.upload(package, "/workspace/glab-input.tar.gz")
        self.ssh(["tar", "-xzf", "/workspace/glab-input.tar.gz", "-C", REMOTE], timeout=120)

    def start(self, runner, env, until):
        args = [
            "python3",
            f"{REMOTE}/generalization_lab/remote_bootstrap.py",
            "launch",
            "--root",
            str(runner.root.relative_to(ROOT)),
            "--run-id",
            runner.run_id,
            "--until",
            until,
            "--deadline",
            str(self.deadline - 30),
        ]
        if runner.force:
            args.append("--force")
        return self.ssh(args, input=canonical(env))

    def poll(self, runner):
        status = json.loads(
            self.ssh(
                [
                    "python3",
                    f"{REMOTE}/generalization_lab/remote_bootstrap.py",
                    "status",
                    "--root",
                    str(runner.root.relative_to(ROOT)),
                    "--run-id",
                    runner.run_id,
                ]
            )
        )
        # Logs contain no credentials in normal use and are redacted on persistence.
        if status.get("tail"):
            append(runner.directory / "logs/remote_poll.jsonl", {"timestamp": now(), **status})
        return status

    def fetch(self, runner):
        root = str(runner.root.relative_to(ROOT))
        for run_id in (runner.run_id, runner.manifest.get("control_run_id")):
            if run_id:
                try:
                    self.download(
                        f"{REMOTE}/{root}/{run_id}/evals/original_mc/scores.jsonl",
                        self.journal / f"{run_id}-original_mc-scores.jsonl",
                    )
                except (RuntimeError, subprocess.TimeoutExpired):
                    # Partial jobs may have no MC scores; preserve the full archive attempt.
                    pass
        result = json.loads(
            self.ssh(
                [
                    "python3",
                    f"{REMOTE}/generalization_lab/remote_bootstrap.py",
                    "package",
                    "--root",
                    root,
                    "--run-id",
                    runner.run_id,
                ],
                timeout=900,
            )
        )
        target = self.journal / "retrieved.tar.gz"
        self.download(result["path"], target)
        if sha(target) != result["sha256"]:
            raise ValueError("Transferred archive SHA256 mismatch")
        safe_extract_results(target, ROOT, runner)

    def fetch_evals(self, runner):
        """Retrieve only this run's protocol evaluation store; run folders are never touched."""
        root = str(runner.root.relative_to(ROOT))
        result = json.loads(
            self.ssh(
                [
                    "python3",
                    f"{REMOTE}/generalization_lab/remote_bootstrap.py",
                    "package",
                    "--root",
                    root,
                    "--run-id",
                    runner.run_id,
                    "--evals",
                ],
                timeout=900,
            )
        )
        target = self.journal / "retrieved-evals.tar.gz"
        self.download(result["path"], target)
        if sha(target) != result["sha256"]:
            raise ValueError("Transferred evaluation archive SHA256 mismatch")
        safe_extract_evals(target, ROOT, runner)


def safe_extract_results(archive, root, runner):
    allowed = {runner.run_id}
    if runner.manifest.get("control_run_id"):
        allowed.add(runner.manifest["control_run_id"])
    relative_root = runner.root.relative_to(root)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported archive member")
            try:
                parts = path.relative_to(relative_root).parts
            except ValueError:
                raise ValueError("Unexpected artifact archive path") from None
            if not parts or parts[0] not in allowed or ".." in path.parts or path.is_absolute():
                raise ValueError("Unsafe artifact archive path")
            if not (root / path).resolve().is_relative_to(root.resolve()):
                raise ValueError("Archive destination escapes the repository through a symlink")
            if (root / path).is_symlink():
                raise ValueError("Artifact destination is a symlink")
        # Python 3.11 compatible safe extraction, after explicit checks above.
        for member in tar.getmembers():
            if member.isdir():
                (root / member.name).mkdir(parents=True, exist_ok=True)
            else:
                dest = root / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with (
                    tar.extractfile(member) as source,
                    dest.with_suffix(dest.suffix + ".transfer").open("wb") as out,
                ):
                    shutil.copyfileobj(source, out)
                os.replace(dest.with_suffix(dest.suffix + ".transfer"), dest)


def _checked_members(tar, root, allowed_prefix):
    for member in tar.getmembers():
        path = Path(member.name)
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ValueError("Unsupported archive member")
        if path.is_absolute() or ".." in path.parts or not path.is_relative_to(allowed_prefix):
            raise ValueError("Unsafe evaluation archive path")
        if not (root / path).resolve().is_relative_to(root.resolve()) or (root / path).is_symlink():
            raise ValueError("Evaluation archive destination escapes the repository")
    return tar.getmembers()


def safe_extract_evals(archive, root, runner):
    prefix = runner.root.relative_to(root) / PROTOCOL_EVALS / runner.run_id
    with tarfile.open(archive, "r:gz") as tar:
        for member in _checked_members(tar, root, prefix):
            dest = root / member.name
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.with_suffix(dest.suffix + ".transfer").open("wb") as out:
                shutil.copyfileobj(source, out)
            os.replace(dest.with_suffix(dest.suffix + ".transfer"), dest)


def control_folder(runner):
    """The matched control packaged with this run, unless it executes as its own job."""
    control = runner.manifest.get("control_run_id")
    if control and runner.manifest.get("control_execution") != "separate":
        return runner.root / control
    return None


def protocol_package_files(runner):
    """This run's protocol requests and partial outputs, plus the frozen inputs they declare."""
    store = runner.root / PROTOCOL_EVALS / runner.directory.name
    files = {p for p in store.rglob("*") if p.is_file() and not p.name.endswith(".lock")} if store.is_dir() else set()
    for request in store.glob("*/request.json"):
        files.update(ROOT / name for name in read(request).get("inputs", []))
    return files


def package_repository(runner, destination):
    included = set()
    for directory in ("generalization_lab", "src"):
        for p in (ROOT / directory).rglob("*"):
            if p.is_file() and not p.is_symlink() and p.suffix == ".py" and "__pycache__" not in p.parts:
                included.add(p)
    included.add(ROOT / "pyproject.toml")
    # Explicit custom input files and only this run and its matched control, never all historical runs.
    included.update(ROOT / p for p in runner.spec.input_hashes)
    # Betley reads the pinned question files through its existing module.
    included.update(
        ROOT / "assets/em_original" / p for p in ("first_plot_questions.yaml", "preregistered_evals.yaml")
    )
    included.update(protocol_package_files(runner))
    folders = [runner.directory] + [f for f in [control_folder(runner)] if f]
    for folder in folders:
        # A finished adapter is pulled from its private Hub repo on the worker instead of being
        # uploaded from the controller (hundreds of MB per job over a home uplink).
        hosted = False
        if (folder != runner.directory or runner.manifest.get("status") == "complete") and (
            runner.spec.compute.upload_private_hub
        ):
            from ..hub import hosted_adapter

            hosted = hosted_adapter(folder) is not None
        included.update(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and not p.name.endswith(".lock")
            and "logs" not in p.relative_to(folder).parts
            and not (hosted and p.relative_to(folder).parts[0] == "checkpoints")
        )
        included.update(p for p in (folder / "logs").glob("usage.jsonl"))
    with tarfile.open(destination, "w:gz") as tar:
        for p in sorted(included):
            if not p.resolve().is_relative_to(ROOT):
                raise ValueError("Remote inputs must be inside repository")
            if p.name.startswith(".env") or p.suffix in {".pem", ".key"}:
                raise ValueError("Secret file cannot be packaged")
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)


def forwarded_env(spec, run_id=None):
    if spec.providers.model_under_test.kind not in {"hf", "mock"}:
        raise ValueError("Vast GPU stages require a local model; API inference stays on the controller")
    env = {}
    if spec.compute.upload_private_hub:
        # Only the Hub write token, sent over SSH stdin to the worker process, so the
        # final adapter is uploaded from the GPU host. Provider API keys never leave.
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("compute.upload_private_hub requires HF_TOKEN on the controller")
        env = {"HF_TOKEN": token, "GLAB_REMOTE_WORKER": "1"}
    store = spec.compute.checkpoint_store
    if store and run_id:
        if store == "hub":
            from ..hub import preflight_private_upload

            env["GLAB_CHECKPOINT_STORE"] = f"hub:{preflight_private_upload()}/{run_id}"
        else:
            env["GLAB_CHECKPOINT_STORE"] = "dir:" + str((ROOT / store[4:] / run_id).resolve())
    return env


class VastBackend:
    def __init__(self, client=None, transport_factory=SSHTransport, watchdog=True):
        self.client = client or VastClient()
        self.transport_factory, self.watchdog = transport_factory, watchdog

    def wait_ready(self, instance_id, spec, journal, deadline):
        ready_deadline = min(deadline - 60, time.time() + spec.compute.startup_timeout_minutes * 60)
        previous_state = None
        while time.time() < ready_deadline:
            instance = self.client.instance(instance_id)
            status = instance.get("actual_status")
            state = (
                status,
                instance.get("status_msg"),
                bool(instance.get("ssh_host") and instance.get("ssh_port")),
            )
            if state != previous_state:
                append(
                    Path(journal) / "startup.jsonl",
                    {"timestamp": now(), **public_instance(instance), "ssh_address_available": state[2]},
                )
                print(
                    redact(f"[Vast {instance_id}] startup: {status}; {str(state[1] or '')[:300]}"), flush=True
                )
                previous_state = state
            if status in {"exited", "offline", "unknown", "destroyed"}:
                raise RuntimeError(f"Vast instance became {status}")
            if status == "running" and instance.get("ssh_host") and instance.get("ssh_port"):
                transport = self.transport_factory(instance, spec, journal, deadline)
                try:
                    transport.ssh(["true"])
                    return transport, instance
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    append(
                        Path(journal) / "startup.jsonl",
                        {"timestamp": now(), "ssh_probe_failed": type(error).__name__},
                    )
            time.sleep(spec.compute.poll_seconds)
        raise TimeoutError("Instance/SSH startup timeout")

    def reconcile_previous_leases(self, runner):
        unresolved = [(p, read(p)) for p in (runner.root / "_vast").glob("*/lease.json")]
        unresolved = [
            (p, lease)
            for p, lease in unresolved
            if lease.get("run_id") == runner.run_id and lease.get("status") != "destroyed"
        ]
        if not unresolved:
            return
        live_labels = {
            i.get("label") for i in self.client.instances() if i.get("actual_status") != "destroyed"
        }
        for path, lease in unresolved:
            if lease["label"] in live_labels:
                raise ValueError(
                    "A previous Vast lease for this run still exists. Use vast cleanup --yes "
                    "or vast destroy INSTANCE_ID --yes before another rental."
                )
            # A killed controller may have missed the final invoice interval.
            # Charge conservatively through reconciliation rather than treating it as free.
            elapsed = max(0, time.time() - datetime.fromisoformat(lease["created_at"]).timestamp())
            hourly = lease["offer"].get("glab_hourly_total", lease["offer"]["dph_total"])
            lease.update(
                status="destroyed",
                reconciled_at=now(),
                approximate_gpu_usd=max(
                    lease.get("approximate_gpu_usd", 0), elapsed / 3600 * hourly
                ),
                accounting_note="Conservative upper estimate through disappearance reconciliation.",
            )
            write(path, lease)

    def run(self, runner, until=None):
        if runner.manifest["status"] == "complete":
            if protocols_pending(runner):
                self.run_protocols(runner)
            return runner.execute()
        if not runner.root.is_relative_to(ROOT):
            raise ValueError("Vast runs root must be inside this repository")
        # Generate and review both datasets without paying for an idle GPU.
        runner.execute(until="quality_check_data")
        if until and until in (
            "setup",
            "generate_data",
            "generate_evals",
            "matched_control",
            "quality_check_data",
        ):
            return runner.directory
        state = read(runner.directory / "state.json")["stages"]
        self.reconcile_previous_leases(runner)
        if state.get("neutral_eval", {}).get("status") == "complete":
            # Remote engine inference is finished. Requested protocol generation may remain.
            if protocols_pending(runner):
                self.run_protocols(runner)
            return runner.execute(until=until)
        spec = runner.spec
        if spec.compute.export_mode == "results_only" and until in (
            "train", "fixed_eval", "adjacent_eval", "free_response_generation", "neutral_eval"
        ):
            raise ValueError("A run stopping before evaluation QA needs compute.export_mode: final_adapter")
        if state.get("train", {}).get("status") == "complete":
            # A results-only partial transfer cannot supply the model for another
            # GPU evaluation. Detect that before creating a paid instance.
            runner.verify_stage(state["train"])
        # Reserve 60 seconds for transfer/teardown; remote work also has an independent deadline.
        remote_until = until if until in (
            "train", "fixed_eval", "adjacent_eval", "free_response_generation", "neutral_eval"
        ) else "neutral_eval"
        prior_runtime = runner.manifest["runtime_seconds"]
        lease, journal, previous_gpu_cost, started, error, cleanup_error = self._rent(
            runner, remote_until, record_manifest=True
        )
        # Reload downloaded state before recording accounting locally.
        runner = load_runner(runner.root, runner.run_id, runner.force)
        runner.manifest["vast"] = {**(runner.manifest.get("vast") or {}), **lease}
        runner.manifest["costs"].update(
            gpu_usd=lease["approximate_gpu_usd"],
            api_usd=runner.ledger.spent(),
            known_total_usd=previous_gpu_cost + lease["approximate_gpu_usd"] + runner.ledger.spent(),
            transfer_usd=None,
            note="Rental approximation; transfer charges/taxes and provider invoice reconciliation are not included.",
        )
        runner.manifest["costs"]["gpu_usd"] += previous_gpu_cost
        # The downloaded manifest already includes remote stage runtime. Replace
        # that interval with rental wall time, which also includes setup/transfer.
        runner.manifest["runtime_seconds"] = prior_runtime + time.time() - started
        if error or cleanup_error:
            runner.manifest["status"] = "failed"
            runner.manifest["failure"] = {
                "stage": "vast",
                "type": type(error or cleanup_error).__name__,
                "message": str(error or cleanup_error),
                "at": now(),
            }
        runner.persist()
        if cleanup_error:
            raise RuntimeError(
                f"Instance teardown could not be confirmed. Run vast cleanup --yes. Lease: {journal}"
            ) from cleanup_error
        if error:
            raise error
        required = ["manifest.json", "state.json", "training/statistics.json"]
        if spec.compute.export_mode == "final_adapter" and not hub_backed(runner.directory):
            required.append("checkpoints/adapter")
        for filename in required:
            if not (runner.directory / filename).exists():
                raise ValueError(f"Missing retrieved result: {filename}")
        return runner.execute(until=until) if remote_until == "neutral_eval" and until != "neutral_eval" else runner.directory

    def run_protocols(self, runner):
        """Evaluation-only rental: run pending protocol GPU work on an existing trained checkpoint.

        Only the run's protocol store is retrieved; its manifest, state and published results are
        never modified. Spending is recorded in the lease journal and the protocol store.
        """
        if not runner.root.is_relative_to(ROOT):
            raise ValueError("Vast runs root must be inside this repository")
        state = read(runner.directory / "state.json")["stages"]
        if state.get("train", {}).get("status") != "complete":
            raise ValueError("Evaluation-only rentals require a trained checkpoint")
        if not hub_backed(runner.directory) and not (runner.directory / "checkpoints/adapter").is_dir():
            raise ValueError("No local adapter or verified private Hub adapter to evaluate")
        self.reconcile_previous_leases(runner)
        lease, journal, _, _, error, cleanup_error = self._rent(runner, "protocols", record_manifest=False)
        write(runner.root / PROTOCOL_EVALS / runner.run_id / "rentals" / (journal.name + ".json"), lease)
        if cleanup_error:
            raise RuntimeError(
                f"Instance teardown could not be confirmed. Run vast cleanup --yes. Lease: {journal}"
            ) from cleanup_error
        if error:
            raise error
        if protocols_pending(runner):
            raise RuntimeError("Remote protocol evaluation finished without completing its GPU outputs")
        return runner.directory

    def _check_instance(self, instance_id):
        try:
            instance = self.client.instance(instance_id)
        except Exception:  # noqa: BLE001 - only a positive interruption report is decisive
            return
        if interrupted(instance):
            raise InstanceInterrupted(
                "Vast instance interrupted (outbid or stopped): "
                f"{instance.get('actual_status')}; {str(instance.get('status_msg') or '')[:200]}"
            )

    def _rent(self, runner, remote_until, record_manifest):
        spec = runner.spec
        enforce(spec)
        env = forwarded_env(spec, runner.run_id)
        key = Path(spec.compute.ssh_key_path).expanduser()
        if not key.is_file() or not Path(str(key) + ".pub").is_file():
            raise ValueError("SSH private/public key pair missing; configure compute.ssh_key_path")
        offers = self.client.offers(spec)
        if not offers:
            raise ValueError("No Vast offers satisfy the configured requirements and price cap")
        offer = offers[0]
        hourly = hourly_cost(offer)
        estimate = enforce(spec, hourly)
        previous_gpu_cost = sum(
            read(p).get("approximate_gpu_usd", 0)
            for p in (runner.root / "_vast").glob("*/lease.json")
            if read(p).get("run_id") == runner.run_id
        )
        if previous_gpu_cost + estimate["projected_max_usd"] > spec.compute.max_total_cost_usd:
            raise ValueError("Previous rental spend plus the new rental bound exceeds this experiment budget")
        self.client.user()  # Authentication checked before rent.
        public_key = Path(str(key) + ".pub").read_text()
        attach_ssh_after_create = False
        try:
            self.client.register_ssh(public_key)
        except TeamSSHRegistrationUnsupported:
            attach_ssh_after_create = True
        journal = runner.root / "_vast" / (runner.run_id + "-" + uuid.uuid4().hex[:8])
        journal.mkdir(parents=True, exist_ok=False)
        package = journal / "input.tar.gz"
        package_repository(runner, package)
        label = getattr(self.client, "label", None) or (
            PROJECT_PREFIX + runner.run_id + ":" + uuid.uuid4().hex[:8]
        )
        started = time.time()
        deadline = started + spec.compute.timeout_minutes * 60
        lease = {
            "label": label,
            "run_id": runner.run_id,
            "purpose": "protocol_evaluation" if remote_until == "protocols" else "engine",
            "created_at": now(),
            "status": "creating",
            "offer": public_instance(offer),
            "rental": spec.compute.rental,
            "bid_price_per_hour": offer.get("glab_bid_price"),
            "hourly_total_usd": hourly,
            "estimate": estimate,
            "deadline": deadline,
        }
        write(journal / "lease.json", lease)
        instance_id, transport, guardian, error, cleanup_error = None, None, None, None, None
        evaluations = (runner.root / PROTOCOL_EVALS / runner.run_id).is_dir()
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        # Evaluation-only rentals keep their lock outside the (possibly historical) run folder.
        lock_path = (
            journal / ".backend.lock" if remote_until == "protocols" else runner.directory / ".backend.lock"
        )
        with lock(lock_path):
            try:
                instance_id = self.client.create(offer, spec, label)
                lease.update(instance_id=instance_id, deadline=deadline, status="rented")
                write(journal / "lease.json", lease)
                if attach_ssh_after_create:
                    self.client.attach_ssh(instance_id, public_key)
                if self.watchdog:
                    guardian = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "generalization_lab.compute.watchdog",
                            "--instance",
                            str(instance_id),
                            "--deadline",
                            str(deadline),
                            "--journal",
                            str(journal),
                        ],
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                if record_manifest:
                    runner.manifest["vast"] = {
                        "instance_id": instance_id,
                        "offer": public_instance(offer),
                        "label": label,
                        "started_at": lease["created_at"],
                    }
                    runner.persist()
                transport, instance = self.wait_ready(instance_id, spec, journal, deadline)
                if spec.compute.rental == "bid" and instance.get("is_bid") is False:
                    raise ValueError("Vast created an on-demand instance for a bid rental; destroying it")
                transport.prepare(package)
                transport.start(runner, env, remote_until)
                poll_failures, polls = 0, 0
                while True:
                    if time.time() >= deadline - 60:
                        raise TimeoutError(
                            "Remote job reached rental deadline; retrieving partial results and destroying"
                        )
                    polls += 1
                    if polls % spec.compute.interruption_check_polls == 0:
                        self._check_instance(instance_id)
                    try:
                        status = transport.poll(runner)
                        poll_failures = 0
                    except (RuntimeError, subprocess.TimeoutExpired):
                        self._check_instance(instance_id)
                        poll_failures += 1
                        if poll_failures >= 3:
                            raise
                        time.sleep(spec.compute.poll_seconds)
                        continue
                    print(
                        f"[{runner.run_id}] Vast {instance_id}: {status.get('status', 'unknown')}", flush=True
                    )
                    if status.get("status") in {"complete", "failed"}:
                        if status["status"] == "failed":
                            self._check_instance(instance_id)
                            raise RuntimeError(
                                "Remote runner failed; see logs/remote_poll.jsonl and remote logs"
                            )
                        break
                    time.sleep(spec.compute.poll_seconds)
                if remote_until != "protocols":
                    transport.fetch(runner)
                if evaluations:
                    transport.fetch_evals(runner)
            except BaseException as exc:
                error = exc
                lease["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                lease["interrupted"] = isinstance(exc, InstanceInterrupted)
            finally:
                # Try retrieval on failure, but never make deletion depend on it.
                if error and transport and not isinstance(error, InstanceInterrupted):
                    for retrieve, wanted in (
                        (transport.fetch, remote_until != "protocols"),
                        (getattr(transport, "fetch_evals", None), evaluations),
                    ):
                        if not wanted or retrieve is None:
                            continue
                        try:
                            retrieve(runner)
                        except BaseException as fetch_error:
                            lease["partial_transfer_error"] = type(fetch_error).__name__
                try:
                    destroyed = {"success": True}
                    if instance_id is not None:
                        destroyed = self.client.destroy(instance_id)
                    else:
                        # Creation may have succeeded even if its HTTP response was lost.
                        for instance in self.client.instances():
                            if instance.get("label") == label:
                                self.client.destroy(instance["id"])
                    lease["status"] = (
                        "retained_for_batch" if destroyed.get("retained_for_batch") else "destroyed"
                    )
                except BaseException as exc:
                    cleanup_error = exc
                    lease.update(status="cleanup_required", cleanup_error=type(exc).__name__)
                finally:
                    lease.update(
                        stopped_at=now(),
                        approximate_gpu_usd=(time.time() - started) / 3600 * hourly,
                    )
                    write(journal / "lease.json", lease)
                    if guardian and cleanup_error is None:
                        guardian.terminate()
                        guardian.wait(timeout=10)
                    if previous_handler is not None:
                        signal.signal(signal.SIGTERM, previous_handler)
        return lease, journal, previous_gpu_cost, started, error, cleanup_error


def protocols_pending(runner):
    """True when this run has protocol requests whose GPU outputs are incomplete."""
    from ..protocols.store import gpu_pending

    return gpu_pending(runner.root, runner.run_id)

def cleanup_candidates(client, root, stale_hours=4):
    leases = {}
    for p in (Path(root) / "_vast").glob("*/lease.json"):
        lease = read(p)
        leases[lease["label"]] = lease
    candidates = []
    for instance in client.instances():
        label = instance.get("label") or ""
        if not label.startswith(PROJECT_PREFIX):
            continue
        lease = leases.get(label)
        if lease:
            deadline = lease.get("deadline")
            if deadline is None and lease.get("created_at"):
                deadline = datetime.fromisoformat(lease["created_at"]).timestamp() + stale_hours * 3600
            stale = (deadline is not None and deadline < time.time()) or lease.get(
                "status"
            ) == "cleanup_required"
        else:
            created = instance.get("start_date")
            stale = isinstance(created, (int, float)) and time.time() - created > stale_hours * 3600
        if stale:
            candidates.append(public_instance(instance))
    return candidates

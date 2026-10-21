"""Offline Vast simulation for the wizard demonstration and tests. No network, no spending.

``SimulatedVastClient`` subclasses the real ``VastClient`` and replaces only its HTTP layer, so the
real offer query, eligibility filter, bid pricing and creation request body are exercised. A
creation request without an explicit bid ``price`` yields an on-demand instance, exactly the
mistake the real client must not make. ``LocalTransport`` reuses ``SSHTransport`` (prepare, start,
poll, fetch) but runs each "remote" command locally inside a separate instance directory, which is
discarded with the instance, so only checkpoints copied to the checkpoint store survive an
injected interruption (an outbid instance).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..compute import vast
from ..io import now

OFFERS = [
    # Eligible: the cheapest single RTX 5090 interruptible offer after storage.
    {"id": 7001, "machine_id": 71, "host_id": 7, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
     "reliability": 0.991, "cuda_max_good": 12.9, "disk_space": 400, "dph_total": 0.52, "min_bid": 0.24,
     "storage_cost": 0.12, "dlperf": 100, "inet_down": 900, "inet_up": 900},
    {"id": 7002, "machine_id": 72, "host_id": 8, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
     "reliability": 0.985, "cuda_max_good": 13.0, "disk_space": 300, "dph_total": 0.55, "min_bid": 0.26,
     "storage_cost": 0.10, "dlperf": 101},
    # Ineligible: substitutes that must never be rented.
    {"id": 7003, "machine_id": 73, "host_id": 9, "gpu_name": "RTX A6000", "gpu_ram": 49140, "num_gpus": 1,
     "reliability": 0.995, "cuda_max_good": 12.8, "disk_space": 400, "dph_total": 0.35, "min_bid": 0.15,
     "storage_cost": 0.10},
    {"id": 7004, "machine_id": 74, "host_id": 10, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 2,
     "reliability": 0.99, "cuda_max_good": 12.9, "disk_space": 400, "dph_total": 0.9, "min_bid": 0.2,
     "storage_cost": 0.10},
    {"id": 7005, "machine_id": 75, "host_id": 11, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
     "reliability": 0.80, "cuda_max_good": 12.9, "disk_space": 400, "dph_total": 0.4, "min_bid": 0.18,
     "storage_cost": 0.10},
    {"id": 7006, "machine_id": 76, "host_id": 12, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
     "reliability": 0.99, "cuda_max_good": 12.9, "disk_space": 400, "dph_total": 0.95, "min_bid": 0.70,
     "storage_cost": 0.10},
]


def simulation_root(study):
    return study.path / "launch/offline_vast"


def prepare_ssh_key(study):
    """A placeholder key pair: the simulated transport never opens an SSH connection."""
    key = simulation_root(study) / "ssh/id_offline"
    key.parent.mkdir(parents=True, exist_ok=True)
    if not key.is_file():
        key.write_text("offline simulation placeholder; not a private key\n")
        Path(str(key) + ".pub").write_text("ssh-ed25519 AAAAofflinesimulation offline@em\n")
    return key


class SimulatedVastClient(vast.VastClient):
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = None

    @contextmanager
    def state(self):
        path = self.root / "state.json"
        with (self.root / "state.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                data = json.loads(path.read_text()) if path.exists() else {
                    "next_id": 501, "instances": {}, "creates": [], "queries": [], "interrupt": {}}
                yield data
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=1))
                os.replace(tmp, path)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def plan_interruption(self, run_id, after_checkpoint=True):
        with self.state() as data:
            data["interrupt"][run_id] = {"after_checkpoint": after_checkpoint, "done": False}

    def request(self, method, path, *, retry=True, **kwargs):
        body = kwargs.get("json")
        with self.state() as data:
            if method == "POST" and path == "/api/v0/bundles":
                data["queries"].append(body)
                bid = body.get("type") == "bid"
                offers = [dict(o) for o in OFFERS]
                if not bid:
                    offers = [{k: v for k, v in o.items() if k != "min_bid"} for o in offers]
                return {"offers": offers}
            if method == "PUT" and path.startswith("/api/v0/asks/"):
                offer = next(o for o in OFFERS if o["id"] == int(path.rsplit("/", 1)[1]))
                instance_id = data["next_id"]
                data["next_id"] += 1
                is_bid = "price" in body
                data["creates"].append({"offer_id": offer["id"], "body": body, "at": now()})
                folder = self.root / "instances" / str(instance_id)
                folder.mkdir(parents=True, exist_ok=True)
                data["instances"][str(instance_id)] = {
                    "id": instance_id, "label": body["label"], "actual_status": "running",
                    "intended_status": "running", "gpu_name": offer["gpu_name"], "gpu_ram": offer["gpu_ram"],
                    "num_gpus": 1, "is_bid": is_bid, "min_bid": offer["min_bid"], "machine_id": offer["machine_id"],
                    "dph_total": body.get("price", offer["dph_total"]), "ssh_host": "sim.local", "ssh_port": 2200,
                    "status_msg": "simulated", "glab_sim_dir": str(folder), "start_date": 0,
                }
                return {"success": True, "new_contract": instance_id}
            if method == "GET" and path == "/api/v1/instances":
                return {"instances": [i for i in data["instances"].values() if i["actual_status"] != "destroyed"]}
            if method == "GET" and path.startswith("/api/v0/instances/"):
                instance = data["instances"].get(path.rsplit("/", 1)[1])
                if instance is None:
                    raise RuntimeError("Vast HTTP 404; response body omitted")
                return {"instances": instance}
            if method == "DELETE" and path.startswith("/api/v0/instances/"):
                instance = data["instances"].get(path.rsplit("/", 1)[1])
                if instance is None:
                    return {"success": True, "already_absent": True}
                _kill(instance)
                instance["actual_status"] = "destroyed"
                shutil.rmtree(instance["glab_sim_dir"], ignore_errors=True)
                return {"success": True}
            if path == "/api/v0/users/current":
                return {"id": 1, "balance": 100, "credit": 0}
            if path == "/api/v0/ssh":
                return [] if method == "GET" else {"success": True}
        raise RuntimeError(f"Simulated Vast API has no route for {method} {path}")

    def interrupt(self, instance_id, message="outbid by a higher bid (simulated)"):
        with self.state() as data:
            instance = data["instances"][str(instance_id)]
            _kill(instance)
            instance.update(actual_status="exited", status_msg=message)
            data["interrupt"].setdefault(instance["label"].split(":")[1], {})["done"] = True

    def planned(self, instance_id):
        with self.state() as data:
            instance = data["instances"][str(instance_id)]
            plan = data["interrupt"].get(instance["label"].split(":")[1])
        return bool(plan and not plan.get("done"))

    def watch(self, instance_id, run_id, checkpoint_root):
        """Outbid the instance as soon as its first checkpoint has been copied off it."""
        while True:
            with self.state() as data:
                if data["instances"][str(instance_id)]["actual_status"] != "running":
                    return
            if any(p.is_dir() for p in Path(checkpoint_root, run_id).glob("step-*")):
                self.interrupt(instance_id)
                return
            time.sleep(0.005)


def _kill(instance):
    status = Path(instance["glab_sim_dir"]).rglob("remote-status.json")
    for path in status:
        try:
            pid = json.loads(path.read_text()).get("pid")
            if pid:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ValueError, ProcessLookupError):
            pass


def transport_factory(client, checkpoint_root):
    class LocalTransport(vast.SSHTransport):
        def __init__(self, instance, spec, journal, deadline):
            super().__init__(instance, spec, journal, deadline)
            self.instance_id = instance["id"]
            self.home = Path(instance["glab_sim_dir"])

        def _map(self, value):
            value = str(value)
            if value.startswith(vast.REMOTE):
                return str(self.home / "repo") + value[len(vast.REMOTE):]
            if value.startswith("/workspace"):
                return str(self.home) + value[len("/workspace"):]
            return value

        def _alive(self):
            with client.state() as data:
                return data["instances"][str(self.instance_id)]["actual_status"] == "running"

        def ssh(self, args, *, input=None, timeout=90):
            if not self._alive():
                raise RuntimeError("SSH command failed with exit code 255; remote log retained")
            command = [sys.executable if a == "python3" else self._map(a) for a in args]
            (self.home / "repo").mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, input=input, text=True, capture_output=True,
                                    timeout=self.timeout(timeout), cwd=self.home / "repo")
            if result.returncode:
                raise RuntimeError(f"SSH command failed with exit code {result.returncode}; remote log retained")
            return result.stdout

        def upload(self, source, target):
            if not self._alive():
                raise RuntimeError("SCP upload failed")
            Path(self._map(target)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, self._map(target))

        def download(self, source, target):
            if not self._alive() or not Path(self._map(source)).is_file():
                raise RuntimeError("SCP download failed after three attempts; partial file retained for diagnosis")
            shutil.copyfile(self._map(source), target)

        def start(self, runner, env, until):
            result = super().start(runner, env, until)
            if client.planned(self.instance_id):
                threading.Thread(target=client.watch, args=(self.instance_id, runner.run_id, checkpoint_root),
                                 daemon=True).start()
            return result

    return LocalTransport

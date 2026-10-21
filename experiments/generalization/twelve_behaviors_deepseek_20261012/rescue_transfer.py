"""Retrieve one finished treatment directly when a Vast proxy download is slow."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

from generalization_lab.compute.vast import (
    SSHTransport,
    VastClient,
    safe_extract_results,
)
from generalization_lab.config import load
from generalization_lab.engine import load_runner
from generalization_lab.io import read, sha, write

ROOT = Path(__file__).resolve().parents[3]


def key(host, port):
    result = subprocess.run(
        ["ssh-keyscan", "-T", "8", "-p", str(port), "-t", "ed25519", host],
        capture_output=True, text=True, timeout=12, check=True,
    )
    return {
        line.split()[2]
        for line in result.stdout.splitlines()
        if line and not line.startswith("#") and len(line.split()) >= 3
    }


def main(prefix):
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("VAST_API_KEY="):
            os.environ["VAST_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    leases = list((ROOT / "runs/_vast").glob(prefix + "_*deepseek_20261012_adapter-*/lease.json"))
    if not leases:
        raise ValueError("No lease matches the run prefix")
    journal_file = max(leases, key=lambda p: read(p)["created_at"])
    lease = read(journal_file)
    if lease["status"] != "rented":
        raise ValueError("The latest lease is no longer active")
    run_id = lease["run_id"]
    runner = load_runner(ROOT / "runs", run_id)
    spec = load(next((ROOT / "experiments/generalization/twelve_behaviors_deepseek_20261012").glob(prefix + "_*.yaml")))
    instance = VastClient().instance(lease["instance_id"])
    direct_host = instance["public_ipaddr"]
    direct_port = instance["ports"]["22/tcp"][0]["HostPort"]
    if not key(instance["ssh_host"], instance["ssh_port"]) & key(direct_host, direct_port):
        raise ValueError("Direct SSH host key differs from Vast proxy host key")
    instance = {**instance, "ssh_host": direct_host, "ssh_port": direct_port}
    transport = SSHTransport(instance, spec, journal_file.parent, lease["deadline"])
    remote_script = """
import hashlib, json, sys, tarfile
sys.path.insert(0, '/workspace/generalization_lab')
from generalization_lab.remote_bootstrap import ROOT, result_files
root, run_id, mode = sys.argv[1:]
folder = ROOT / root / run_id
archive = ROOT / (run_id + '-rescue.tar.gz')
with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
    for path in result_files(folder, mode):
        tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
print(json.dumps({'path': str(archive), 'sha256': h.hexdigest()}))
"""
    result = json.loads(transport.ssh(
        ["python3", "-c", remote_script, str(runner.root.relative_to(ROOT)), run_id,
         spec.compute.export_mode], timeout=900,
    ))
    target = journal_file.parent / "rescue.tar.gz"
    temporary = target.with_suffix(".gz.part")
    command = [
        "scp", *transport.options, "-P", transport.port,
        f"{transport.destination}:{result['path']}", str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, timeout=min(3600, max(1, lease["deadline"] - time.time() - 60)))
    if completed.returncode:
        raise RuntimeError("Direct SCP rescue failed")
    os.replace(temporary, target)
    if sha(target) != result["sha256"]:
        raise ValueError("Rescued archive SHA256 mismatch")
    safe_extract_results(target, ROOT, runner)
    write(journal_file.parent / "rescue_verified.json", {
        "run_id": run_id,
        "archive_sha256": result["sha256"],
        "archive_bytes": target.stat().st_size,
    })
    print("RESCUED", run_id, target.stat().st_size, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

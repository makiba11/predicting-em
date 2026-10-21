"""Resume a partial direct adapter archive after SCP's transfer timer expires."""

from pathlib import Path
import json
import os
import subprocess
import sys
import time

from generalization_lab.compute.vast import SSHTransport, VastClient, safe_extract_results
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
    lease_file = max(leases, key=lambda p: read(p)["created_at"])
    lease = read(lease_file)
    if lease["status"] != "rented":
        raise ValueError("Latest lease is not active")
    instance = VastClient().instance(lease["instance_id"])
    direct_host = instance["public_ipaddr"]
    direct_port = instance["ports"]["22/tcp"][0]["HostPort"]
    if not key(instance["ssh_host"], instance["ssh_port"]) & key(direct_host, direct_port):
        raise ValueError("Direct SSH host key differs from Vast proxy host key")
    direct = {**instance, "ssh_host": direct_host, "ssh_port": direct_port}
    spec = load(next((ROOT / "experiments/generalization/twelve_behaviors_deepseek_20261012").glob(prefix + "_*.yaml")))
    transport = SSHTransport(direct, spec, lease_file.parent, lease["deadline"])
    run_id = lease["run_id"]
    remote = f"/workspace/generalization_lab/{run_id}-rescue.tar.gz"
    size = int(transport.ssh(["stat", "-c", "%s", remote], timeout=20).strip())
    digest = transport.ssh(["sha256sum", remote], timeout=900).split()[0]
    target = lease_file.parent / "rescue.tar.gz"
    temporary = lease_file.parent / "rescue.tar.gz.part"
    if target.exists() and sha(target) == digest:
        pass
    else:
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise ValueError("Partial archive is larger than the remote archive")
        if offset < size:
            command = [
                "ssh", *transport.options, "-p", transport.port, transport.destination,
                f"tail -c +{offset + 1} {remote}",
            ]
            with temporary.open("ab") as out:
                result = subprocess.run(
                    command, stdout=out, stderr=subprocess.PIPE,
                    timeout=min(3600, max(1, lease["deadline"] - time.time() - 60)),
                )
            if result.returncode:
                raise RuntimeError("Direct SSH archive resume failed")
        if temporary.stat().st_size != size or sha(temporary) != digest:
            raise ValueError("Resumed archive size or SHA256 mismatch")
        os.replace(temporary, target)
    runner = load_runner(ROOT / "runs", run_id)
    safe_extract_results(target, ROOT, runner)
    write(lease_file.parent / "rescue_verified.json", {
        "run_id": run_id,
        "archive_sha256": digest,
        "archive_bytes": size,
        "resumed": True,
    })
    print("RESUMED_RESCUE", run_id, size, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

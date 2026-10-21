"""After GPU work completes, omit the already-local control from worker exports."""

from pathlib import Path
import json
import os
import subprocess
import sys

from generalization_lab.compute.vast import SSHTransport, VastClient
from generalization_lab.config import load
from generalization_lab.io import read, write

ROOT = Path(__file__).resolve().parents[3]


def host_key(host, port):
    result = subprocess.run(
        ["ssh-keyscan", "-T", "8", "-p", str(port), "-t", "ed25519", host],
        capture_output=True, text=True, timeout=12, check=True,
    )
    return {
        line.split()[2]
        for line in result.stdout.splitlines()
        if line and not line.startswith("#") and len(line.split()) >= 3
    }


def trim(prefix):
    leases = list((ROOT / "runs/_vast").glob(prefix + "_*deepseek_20261012_adapter-*/lease.json"))
    journal_file = max(leases, key=lambda p: read(p)["created_at"])
    lease = read(journal_file)
    if lease["status"] != "rented":
        raise ValueError("Latest lease is not active")
    run_id = lease["run_id"]
    poll_file = ROOT / "runs" / run_id / "logs/remote_poll.jsonl"
    if not poll_file.exists() or json.loads(poll_file.read_text().splitlines()[-1])["status"] != "complete":
        raise ValueError("GPU work must complete before changing the export helper")
    instance = VastClient().instance(lease["instance_id"])
    direct_host = instance["public_ipaddr"]
    direct_port = instance["ports"]["22/tcp"][0]["HostPort"]
    if not host_key(instance["ssh_host"], instance["ssh_port"]) & host_key(direct_host, direct_port):
        raise ValueError("Direct SSH host key differs from Vast proxy host key")
    direct = {**instance, "ssh_host": direct_host, "ssh_port": direct_port}
    spec = load(next((ROOT / "experiments/generalization/twelve_behaviors_deepseek_20261012").glob(prefix + "_*.yaml")))
    transport = SSHTransport(direct, spec, journal_file.parent, lease["deadline"])
    script = """
from pathlib import Path
p = Path('/workspace/generalization_lab/generalization_lab/remote_bootstrap.py')
old = '        if manifest.get("control_run_id"):\\n            folders.append(ROOT / args.root / manifest["control_run_id"])\\n'
new = '        # Shared control artifacts were verified and retained on the controller.\\n'
source = p.read_text()
if old in source:
    p.write_text(source.replace(old, new, 1))
elif new not in source:
    raise ValueError('Expected package block not found')
print('treatment-only-package-ready')
"""
    output = transport.ssh(["python3", "-c", script], timeout=30).strip()
    write(journal_file.parent / "transfer_optimization.json", {
        "run_id": run_id,
        "action": "omit completed matched control from subsequent worker result archives",
        "science_outputs_unchanged": True,
    })
    print(prefix, output, flush=True)


def main():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("VAST_API_KEY="):
            os.environ["VAST_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    for prefix in sys.argv[1:]:
        trim(prefix)


if __name__ == "__main__":
    main()

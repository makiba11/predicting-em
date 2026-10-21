"""Finalize verified direct adapter recoveries, then write the batch report."""

from pathlib import Path
import json
import subprocess
import sys
import time

from generalization_lab.io import read, sha

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREFIXES = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "12"]


def latest(prefix):
    leases = list((ROOT / "runs/_vast").glob(prefix + "_*deepseek_20261012_adapter-*/lease.json"))
    return max(leases, key=lambda p: read(p)["created_at"])


def rescue_running(prefix):
    needle = f"rescue_transfer.py {prefix}"
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode()
            if needle in command and "python" in command:
                return True
        except (OSError, UnicodeDecodeError):
            pass
    return False


def ready(prefix):
    lease_file = latest(prefix)
    journal = lease_file.parent
    archive = journal / "rescue.tar.gz"
    if not archive.is_file() or (journal / "rescue.tar.gz.part").exists():
        return False
    if not (journal / "rescue_verified.json").exists() and rescue_running(prefix):
        return False
    run = ROOT / "runs" / read(lease_file)["run_id"]
    state = read(run / "state.json")["stages"]
    if state.get("neutral_eval", {}).get("status") != "complete":
        return False
    adapter = run / "checkpoints/adapter/adapter_model.safetensors"
    expected = state.get("train", {}).get("outputs", {}).get("checkpoints/adapter/adapter_model.safetensors")
    return adapter.is_file() and expected == sha(adapter)


def main():
    finalized = set()
    deadline = time.monotonic() + 2 * 3600
    while time.monotonic() < deadline:
        for prefix in PREFIXES:
            if prefix in finalized:
                continue
            lease = read(latest(prefix))
            run = ROOT / "runs" / lease["run_id"]
            if lease["status"] == "destroyed" and read(run / "manifest.json").get("worker_auth"):
                finalized.add(prefix)
                continue
            if not ready(prefix):
                continue
            print("FINALIZING", prefix, flush=True)
            result = subprocess.run(
                [sys.executable, str(HERE / "finalize_rescue.py"), prefix],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            if result.returncode:
                print("FINALIZE_RETRY", prefix, result.stderr.strip()[-300:], flush=True)
            else:
                finalized.add(prefix)
                print(result.stdout.strip(), flush=True)
        if len(finalized) == len(PREFIXES):
            manifests = list((ROOT / "runs").glob("*_deepseek_20261012_adapter-*/manifest.json"))
            if len(manifests) == 12 and all(read(p)["status"] == "complete" for p in manifests):
                subprocess.run([sys.executable, str(HERE / "make_report.py")], cwd=ROOT, check=True)
                print("BATCH_COMPLETE", flush=True)
                return
        time.sleep(10)
    print("BATCH_INCOMPLETE", sorted(set(PREFIXES) - finalized), flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

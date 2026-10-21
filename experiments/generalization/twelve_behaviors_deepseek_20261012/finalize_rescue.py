"""Close a rescued Vast lease and release its run for local judging."""

from datetime import datetime
from pathlib import Path
import os
import signal
import sys
import time

from generalization_lab.compute.vast import VastClient
from generalization_lab.io import read, write, now

ROOT = Path(__file__).resolve().parents[3]


def stop_watchdog(instance_id):
    marker = f"--instance {instance_id}"
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = proc.read_bytes().replace(b"\0", b" ").decode()
            if "generalization_lab.compute.watchdog" in command and marker in command:
                os.kill(int(proc.parent.name), signal.SIGTERM)
        except (OSError, ProcessLookupError, UnicodeDecodeError):
            pass


def main(prefix):
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("VAST_API_KEY="):
            os.environ["VAST_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    leases = list((ROOT / "runs/_vast").glob(prefix + "_*deepseek_20261012_adapter-*/lease.json"))
    journal_file = max(leases, key=lambda p: read(p)["created_at"])
    lease = read(journal_file)
    run_id = lease["run_id"]
    directory = ROOT / "runs" / run_id
    rescue = journal_file.parent / "rescue.tar.gz"
    adapter = directory / "checkpoints/adapter/adapter_model.safetensors"
    state = read(directory / "state.json")["stages"]
    if not rescue.is_file() or not adapter.is_file() or state.get("neutral_eval", {}).get("status") != "complete":
        raise ValueError("Verified rescue archive, adapter, or GPU stage is missing")
    client = VastClient()
    instance_id = lease["instance_id"]
    live_ids = {i["id"] for i in client.instances()}
    if instance_id in live_ids:
        client.destroy(instance_id)
    for _ in range(6):
        live_ids = {i["id"] for i in client.instances()}
        if instance_id not in live_ids:
            break
        time.sleep(5)
    else:
        raise RuntimeError("Vast instance still active after destroy request")
    stopped = time.time()
    started = datetime.fromisoformat(lease["created_at"]).timestamp()
    lease.update(
        status="destroyed",
        stopped_at=now(),
        approximate_gpu_usd=max(lease.get("approximate_gpu_usd", 0),
                                max(0, stopped - started) / 3600 * float(lease["offer"]["dph_total"])),
        transfer_recovery="verified treatment-only direct archive",
    )
    write(journal_file, lease)
    stop_watchdog(instance_id)
    manifest_file = directory / "manifest.json"
    manifest = read(manifest_file)
    manifest["status"] = "partial"
    manifest["worker_auth"] = {
        "hugging_face_token_forwarded": True,
        "openrouter_key_forwarded": False,
        "openai_key_forwarded": False,
    }
    manifest["transfer_recovery"] = {
        "method": "verified treatment-only direct archive",
        "control_adapter_retained_locally": True,
    }
    gpu = sum(read(p).get("approximate_gpu_usd", 0)
              for p in (ROOT / "runs/_vast").glob("*/lease.json")
              if read(p).get("run_id") == run_id)
    manifest["costs"]["gpu_usd"] = gpu
    manifest["costs"]["known_total_usd"] = gpu + manifest["costs"].get("api_usd", 0)
    write(manifest_file, manifest)
    print("FINALIZED_RESCUE", run_id, instance_id, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

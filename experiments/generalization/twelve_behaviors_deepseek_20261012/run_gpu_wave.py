"""Resume frozen GPU stages with the explicitly supplied HF token.

The rented workers receive only HF_TOKEN. Dataset generation, free-response
judging, and semantic evaluation review remain on the controller.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from generalization_lab.compute import vast
from generalization_lab.config import load
from generalization_lab.engine import load_runner, run_identity
from generalization_lab.io import read, redact, remember_secret
from generalization_lab.preregistration import find

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted(Path(__file__).resolve().parent.glob("[0-9][0-9]_*.yaml"))


def main():
    token = os.environ.get("HF_TOKEN") or sys.stdin.readline().strip()
    if not token.startswith("hf_"):
        raise ValueError("Expected a Hugging Face token in HF_TOKEN or stdin")
    remember_secret(token)
    os.environ["HF_TOKEN"] = token
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("VAST_API_KEY="):
            os.environ["VAST_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not os.environ.get("VAST_API_KEY"):
        raise ValueError("VAST_API_KEY missing from .env")

    # Authentication is an environment setting, not part of the frozen method.
    # The controller never forwards OpenRouter, OpenAI, or Vast API credentials.
    def worker_env(spec):
        if spec.providers.model_under_test.kind != "hf":
            raise ValueError("This wave expects a local HF target on each worker")
        return {"HF_TOKEN": token}

    vast.forwarded_env = worker_env

    # The shared control adapter makes each archive about 303 MB. Give SCP
    # enough time when several rentals share the controller's uplink.
    def upload_with_headroom(self, source, target):
        result = subprocess.run(
            ["scp", *self.options, "-P", self.port, str(source), f"{self.destination}:{target}"],
            capture_output=True,
            timeout=self.timeout(3600),
        )
        if result.returncode:
            raise RuntimeError("SCP upload failed")

    vast.SSHTransport.upload = upload_with_headroom

    # The control adapter is already complete and verified locally. Only the
    # treatment run needs to come back; this halves each final transfer.
    def fetch_treatment(self, runner):
        remote_script = """
import hashlib, json, sys, tarfile
from pathlib import Path
sys.path.insert(0, '/workspace/generalization_lab')
from generalization_lab.remote_bootstrap import ROOT, result_files
root, run_id, mode = sys.argv[1:]
folder = ROOT / root / run_id
archive = ROOT / (run_id + '-result.tar.gz')
with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
    for path in result_files(folder, mode):
        tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
print(json.dumps({'path': str(archive), 'sha256': h.hexdigest()}))
"""
        result = json.loads(self.ssh(
            [
                "python3", "-c", remote_script,
                str(runner.root.relative_to(ROOT)), runner.run_id,
                runner.spec.compute.export_mode,
            ],
            timeout=900,
        ))
        target = self.journal / "retrieved.tar.gz"
        self.download(result["path"], target)
        if vast.sha(target) != result["sha256"]:
            raise ValueError("Transferred archive SHA256 mismatch")
        vast.safe_extract_results(target, ROOT, runner)

    vast.SSHTransport.fetch = fetch_treatment
    jobs = []
    for path in CONFIGS:
        spec = load(path)
        jobs.append(run_identity(spec, find(ROOT / "runs", spec)))

    selected = set(sys.argv[1:])
    if selected:
        jobs = [run_id for run_id in jobs if any(run_id.startswith(prefix) for prefix in selected)]
        if not jobs:
            raise ValueError("No run IDs match the supplied prefixes")

    control_id = read(ROOT / "runs" / jobs[0] / "manifest.json")["control_run_id"]
    control_dir = ROOT / "runs" / control_id
    while True:
        state = read(control_dir / "state.json")["stages"]
        if (state.get("neutral_eval", {}).get("status") == "complete"
                and list((control_dir / "checkpoints/adapter").glob("*"))):
            break
        time.sleep(3)
    print("CONTROL_READY", control_id, flush=True)

    def run_one(index, run_id):
        time.sleep((index % 6) * 8)
        for attempt in range(4):
            try:
                runner = load_runner(ROOT / "runs", run_id)
                state = read(runner.directory / "state.json")["stages"]
                if state.get("neutral_eval", {}).get("status") == "complete":
                    adapters = list((runner.directory / "checkpoints/adapter").glob("*"))
                    if adapters:
                        return run_id, "complete", len(adapters), ""
                path = vast.VastBackend().run(runner, until="neutral_eval")
                state = read(path / "state.json")["stages"]
                if state.get("neutral_eval", {}).get("status") != "complete":
                    raise RuntimeError("GPU evaluation did not finish")
                adapters = list((path / "checkpoints/adapter").glob("*"))
                if not adapters:
                    raise RuntimeError("Final adapter was not retrieved")
                manifest = read(path / "manifest.json")
                manifest["worker_auth"] = {
                    "hugging_face_token_forwarded": True,
                    "openrouter_key_forwarded": False,
                    "openai_key_forwarded": False,
                }
                from generalization_lab.io import write
                write(path / "manifest.json", manifest)
                return run_id, "complete", len(adapters), ""
            except Exception as error:
                message = str(error)
                if ("Vast HTTP 429" in message or "Vast HTTP 400" in message) and attempt < 3:
                    time.sleep(20 * (attempt + 1))
                    continue
                return run_id, "failed", 0, redact(f"{type(error).__name__}: {message}")

    max_workers = int(os.environ.get("GLAB_GPU_WORKERS", "2"))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, i, run_id) for i, run_id in enumerate(jobs)]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("GPU_RESULT", *result, flush=True)
    print("GPU_WAVE", sum(r[1] == "complete" for r in results), "of", len(results), flush=True)
    if any(r[1] != "complete" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

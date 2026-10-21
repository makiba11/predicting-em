"""One paid Vast rental for the microbatch throughput comparison."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

from generalization_lab.compute.vast import (
    TeamSSHRegistrationUnsupported,
    VastBackend,
    VastClient,
    public_instance,
)
from generalization_lab.config import expand_matrix
from generalization_lab.io import ROOT, now, write
from generalization_lab.secrets import load_project_credentials


# Three pilot conditions (384 rows, 12 optimizer steps) give a stable steady-state rate.
PILOTS = [
    next((ROOT / "artifacts/legacy_results/twelve-behaviors-deepseek-20261012/conditions/").glob(f"{name}_deepseek_20261012_adapter/*/data/train.jsonl"))
    for name in ("02_profanity_only", "07_self_aggrandizing", "12_cynical")
]
CONFIGS = sys.argv[1:] or ["4:gc", "4:nogc", "8:nogc", "16:nogc"]


def package(spec, target):
    bench = target.parent / "inputs"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "spec.json").write_text(json.dumps(spec.model_dump(mode="json")))
    with tarfile.open(target, "w:gz") as archive:
        for folder in ("generalization_lab", "src"):
            for path in (ROOT / folder).rglob("*.py"):
                if "__pycache__" not in path.parts:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
        for path in (
            ROOT / "pyproject.toml",
            ROOT / "experiments/generalization/benchmark_microbatch.py",
        ):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        combined = bench / "train.jsonl"
        with combined.open("w") as out:
            for pilot in PILOTS:
                for line in pilot.read_text().splitlines():
                    row = json.loads(line)
                    row["id"] = pilot.parts[-4] + "/" + row["id"]
                    out.write(json.dumps(row) + "\n")
        archive.add(combined, arcname="bench/data/train.jsonl")
        archive.add(bench / "spec.json", arcname="bench/spec.json")


def main():
    load_project_credentials()
    spec = expand_matrix(ROOT / "experiments/generalization/twelve_behaviors_2000_two_seed_v2_20261015.yaml")[0][0]
    spec.dataset.num_examples = 128
    if os.environ.get("GLAB_BENCH_GPU"):
        spec.compute.gpu_preferences = [os.environ["GLAB_BENCH_GPU"]]
    client = VastClient()
    offers = client.offers(spec)
    if not offers:
        raise ValueError("No Vast offer meets the benchmark GPU and price limits")
    offer = offers[0]
    key = Path(spec.compute.ssh_key_path).expanduser()
    public_key = Path(str(key) + ".pub").read_text()
    attach = False
    try:
        client.register_ssh(public_key)
    except TeamSSHRegistrationUnsupported:
        attach = True
    journal = ROOT / "runs/_vast" / ("microbatch-benchmark-" + uuid.uuid4().hex[:8])
    journal.mkdir(parents=True, exist_ok=False)
    package(spec, journal / "input.tar.gz")
    label = "glab:microbatch-benchmark:" + uuid.uuid4().hex[:8]
    deadline = time.time() + 3600
    lease = {"label": label, "created_at": now(), "offer": public_instance(offer), "deadline": deadline}
    write(journal / "lease.json", lease)
    instance_id = None
    guardian = None
    try:
        instance_id = client.create(offer, spec, label)
        lease.update(instance_id=instance_id, status="rented")
        write(journal / "lease.json", lease)
        guardian = subprocess.Popen(
            [sys.executable, "-m", "generalization_lab.compute.watchdog", "--instance", str(instance_id),
             "--deadline", str(deadline), "--journal", str(journal)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        if attach:
            client.attach_ssh(instance_id, public_key)
        transport, _ = VastBackend(client=client, watchdog=False).wait_ready(instance_id, spec, journal, deadline)
        transport.prepare(journal / "input.tar.gz")
        transport.ssh(
            ["bash", "-lc", "cd /workspace/generalization_lab && python3 -m pip install --disable-pip-version-check '.[train]'"],
            timeout=900,
        )
        output = transport.ssh(
            ["bash", "-lc", "cd /workspace/generalization_lab && HF_HOME=/workspace/hf_cache python3 experiments/generalization/benchmark_microbatch.py --configs " + " ".join(CONFIGS)],
            timeout=2400,
        )
        transport.download("/workspace/generalization_lab/bench/results.json", journal / "results.json")
        results = json.loads((journal / "results.json").read_text())
        print(json.dumps({"journal": str(journal), "results": results}, indent=2))
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
            raise
        finally:
            lease.update(stopped_at=now(), approximate_gpu_usd=(3600 - max(0, deadline - time.time())) / 3600 * float(offer["dph_total"]))
            write(journal / "lease.json", lease)
            if guardian and lease["status"] == "destroyed":
                guardian.terminate()
                guardian.wait(timeout=10)


if __name__ == "__main__":
    main()

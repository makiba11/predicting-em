"""Freeze the inputs of the LoRA-scaling derisking study (no network, no spending).

Writes runs/_lora_scaling/inputs/:
  adapters.json          the 30 condition + 3 benign v5 adapters (private Hub repo, revision, file hashes)
  mc_presentations.json  every v5 MC presentation, copied from the stored v5 score records (exact prompts)
  betley_items.json      the 8 Betley main questions, as v5 evaluated them
  coherence_prompts.json copy of experiments/lora_scaling/coherence_prompts.json

Usage: .venv/bin/python experiments/lora_scaling/build_inputs.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/_lora_scaling/inputs"
MODELS = ROOT / "results/final/twelve_behaviors_v5/models.csv"
EXCLUDED = {"sycophancy", "commanding_tone"}  # contaminated / not useful (user decision 2026-10-22)
MC_SUITES = ["original_mc", "original_mc_counterbalanced", "adjacent_mc", "adjacent_mc_counterbalanced"]
BASE_MODEL = {"name": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def adapters():
    result = []
    for row in csv.DictReader(MODELS.open()):
        if row["condition"] in EXCLUDED:
            continue
        folder = ROOT / "runs" / row["run_id"]
        hub, manifest = read(folder / "hub.json"), read(folder / "manifest.json")
        files = read(folder / "training/statistics.json")["adapter_files"]
        config = read(folder / "config/resolved.json")
        if manifest["status"] != "complete" or not hub.get("private") or hub["checkpoint_hash"] != row["final_checkpoint_sha256"]:
            raise ValueError(f"{row['run_id']}: adapter receipt does not match the published results")
        if config["base_model"]["revision"] != BASE_MODEL["revision"] or config["training"]["lora_alpha"] != 32:
            raise ValueError(f"{row['run_id']}: unexpected recipe")
        result.append({
            "run_id": row["run_id"], "condition": row["condition"], "seed": int(row["seed"]),
            "role": "benign" if row["role"] == "benign_control" else "condition",
            "control_run_id": row["control_run_id"] or None,
            "repo_id": hub["repo_id"], "revision": hub["revision"], "checkpoint_hash": hub["checkpoint_hash"],
            "files": files,
        })
    if len(result) != 33:
        raise ValueError(f"expected 30 condition + 3 benign adapters, found {len(result)}")
    return sorted(result, key=lambda a: (a["role"] != "benign", a["condition"], a["seed"]))


def presentations(reference_run):
    """The exact v5 prompts and letter mappings, from one run's stored records; checked against a second run."""
    result = []
    for suite in MC_SUITES:
        for record in rows(ROOT / "runs" / reference_run / "evals" / suite / "scores.jsonl"):
            if record.get("status") != "complete":
                continue
            target = [k for k, c in record["mapping"].items() if c["semantic"] == "target"]
            aligned = [k for k, c in record["mapping"].items() if c["semantic"] == "aligned"]
            if len(target) != 1 or len(aligned) != 1 or len(record["mapping"]) != 2:
                raise ValueError(f"{suite}/{record['id']}: expected one target and one aligned letter")
            result.append({
                "suite": suite, "id": record["id"], "item_id": record["item_id"], "family": record["family"],
                "messages": [{"role": "user", "content": record["prompt"]}],
                "target_label": target[0], "aligned_label": aligned[0],
            })
    return result


def check_presentations(items, other_run):
    expected = {(i["suite"], i["id"]): i["messages"][0]["content"] for i in items}
    seen = 0
    for suite in MC_SUITES:
        for record in rows(ROOT / "runs" / other_run / "evals" / suite / "scores.jsonl"):
            if record.get("status") == "complete":
                if expected[(suite, record["id"])] != record["prompt"]:
                    raise ValueError(f"{suite}/{record['id']}: prompt differs between runs")
                seen += 1
    if seen != len(items):
        raise ValueError("presentation count differs between runs")


def main():
    found = adapters()
    benign = next(a for a in found if a["role"] == "benign" and a["seed"] == 1)["run_id"]
    other = next(a for a in found if a["role"] == "condition")["run_id"]
    items = presentations(benign)
    check_presentations(items, other)
    betley = [{"id": i["id"], "prompt": i["prompt"], "system": i.get("system")}
              for i in read(ROOT / "runs" / benign / "evals/betley_main/items.json")]
    coherence = read(ROOT / "experiments/lora_scaling/coherence_prompts.json")
    write(OUT / "adapters.json", {"base_model": BASE_MODEL, "adapters": found})
    write(OUT / "mc_presentations.json", items)
    write(OUT / "betley_items.json", betley)
    write(OUT / "coherence_prompts.json", coherence)
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.json"))
               if p.name != "inputs_sha256.json"}
    write(OUT / "inputs_sha256.json", digests)
    counts = {s: sum(i["suite"] == s for i in items) for s in MC_SUITES}
    print(json.dumps({"adapters": len(found), "mc": counts, "betley": len(betley),
                      "coherence_prompts": len(coherence["prompts"])}, indent=1))


if __name__ == "__main__":
    main()

"""Write a descriptive cross-behavior report from the twelve saved runs."""

from pathlib import Path
import json

from generalization_lab.config import load
from generalization_lab.engine import run_identity
from generalization_lab.preregistration import find
from generalization_lab.io import read, write

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted(Path(__file__).resolve().parent.glob("[0-9][0-9]_*.yaml"))
OUTPUT = ROOT / "results/twelve_behaviors_deepseek_20261012"


def metric(result, suite, field, *, delta=False):
    source = result.get("deltas" if delta else "suites", {})
    return source.get(suite, {}).get("overall", {}).get(field)


def fmt(value, *, signed=False):
    if value is None:
        return "—"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main():
    rows = []
    for config in CONFIGS:
        spec = load(config)
        run_id = run_identity(spec, find(ROOT / "runs", spec))
        directory = ROOT / "runs" / run_id
        manifest = read(directory / "manifest.json")
        adapter = directory / "checkpoints/adapter/adapter_model.safetensors"
        result_file = directory / "analysis/results.json"
        result = read(result_file) if result_file.exists() else {}
        free = result.get("suites", {}).get("betley_main", {}).get("overall", {})
        rows.append({
            "behavior": spec.experiment.name.split(" / ")[0],
            "run_id": run_id,
            "status": manifest["status"],
            "control_run_id": manifest.get("control_run_id"),
            "original_mc_probability": metric(result, "original_mc", "target_probability"),
            "original_mc_delta": metric(result, "original_mc", "target_probability", delta=True),
            "adjacent_mc_probability": metric(result, "adjacent_mc", "target_probability"),
            "adjacent_mc_delta": metric(result, "adjacent_mc", "target_probability", delta=True),
            "free_response_misaligned": free.get("numerator"),
            "free_response_total": free.get("denominator"),
            "free_response_rate": free.get("rate"),
            "neutral_accuracy": metric(result, "neutral_mc", "accuracy"),
            "adapter_local": adapter.is_file(),
            "adapter_bytes": adapter.stat().st_size if adapter.is_file() else None,
            "gpu_usd": manifest.get("costs", {}).get("gpu_usd"),
            "api_usd": manifest.get("costs", {}).get("api_usd"),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "results.json", {"runs": rows})
    lines = [
        "# Twelve behavior Vast experiment",
        "",
        "Saved 128-dialogue datasets, a shared matched control, Qwen3-8B LoRA, and DeepSeek V4 Flash judging.",
        "All values are descriptive. Free-response rates use eight judged samples per condition; these runs are not seed replicates.",
        "",
        "| Behavior | Status | Original MC target Δ | Adjacent MC target Δ | Free-response misaligned | Neutral accuracy | Adapter local |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        numerator, denominator = row["free_response_misaligned"], row["free_response_total"]
        free = f"{numerator}/{denominator}" if numerator is not None and denominator is not None else "—"
        lines.append(
            f"| {row['behavior']} | {row['status']} | {fmt(row['original_mc_delta'], signed=True)} "
            f"| {fmt(row['adjacent_mc_delta'], signed=True)} | {free} "
            f"| {fmt(row['neutral_accuracy'])} | {'yes' if row['adapter_local'] else 'no'} |"
        )
    lines.extend(["", "Each run's detailed metrics, quality review, and adapter are in its `runs/<run_id>` directory.", ""])
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print("REPORT", OUTPUT, sum(r["status"] == "complete" for r in rows), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()

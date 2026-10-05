"""Recheck the five clearest neutral MC regressions under all four answer orders."""

import argparse
import csv
import math
import os
import statistics
from pathlib import Path

import em_experiment as em
import mc_adjacent_eval as adjacent
import mc_paraphrase_eval as mc
import neutral_mc_eval as neutral

ROOT = Path(__file__).resolve().parents[1]
SOURCE = neutral.SOURCE
PRIOR = neutral.OUTPUT
OUTPUT = SOURCE / "mc-neutral-counterbalance-v1"


def items():
    source_path = PRIOR / "clear_regressions.csv"
    if not source_path.exists():
        raise ValueError("Complete the neutral MC comparison before counterbalancing")
    with source_path.open(newline="") as stream:
        selected = {row["item_id"] for row in csv.DictReader(stream)}
    if len(selected) != 5:
        raise ValueError(f"Expected five selected regression items, found {len(selected)}")
    originals = [item for item in neutral.items() if item["id"] in selected]
    if len(originals) != len(selected):
        raise ValueError("A regression item is missing from the frozen neutral set")
    prepared = []
    for source in originals:
        for rotation in range(4):
            choices = source["choices"]
            choices = choices[-rotation:] + choices[:-rotation] if rotation else choices
            prepared.append({
                **source,
                "id": f"{source['id']}/order{rotation}",
                "source_item_id": source["id"],
                "rotation": rotation,
                "neutral_index": len(prepared) + 1,
                "choices": choices,
            })
    return prepared


def prepare(output):
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must be under the repository artifacts directory")
    questions, runs = items(), adjacent.checkpoints()
    output.mkdir(parents=True, exist_ok=True)
    em.freeze(output / "manifest.json", {
        "source_neutral_questions_sha256": em.sha(neutral.QUESTIONS),
        "selection_file": str((PRIOR / "clear_regressions.csv").relative_to(ROOT)),
        "selection_sha256": em.sha(PRIOR / "clear_regressions.csv"),
        "source_item_ids": list(dict.fromkeys(item["source_item_id"] for item in questions)),
        "answer_orders_per_item": 4,
        "checkpoints": runs,
        "scoring": "The same four-choice letter-logprob statistic on four cyclic answer orders",
    })
    return questions, runs


def config_for(output, budget_usd):
    config = neutral.config_for(output, budget_usd)
    config["protocol"] = "insult-condition-2k-mc-neutral-counterbalance-v1"
    config["execution"]["account_scope"] = (
        "Four-order counterbalance of five posthoc neutral MC regression items on 43 saved checkpoints"
    )
    return config


def score_batch(client, tokenizer, renderer, usage, group, index_field="neutral_index", stage="neutral_mc_counterbalance"):
    scored = neutral.score_batch(client, tokenizer, renderer, usage, group, index_field, stage)
    for item, row in zip(group, scored, strict=True):
        row["source_item_id"] = item["source_item_id"]
        row["rotation"] = item["rotation"]
    return scored


def summarize(output, questions, runs):
    expected = [item["id"] for item in questions]
    source_ids = list(dict.fromkeys(item["source_item_id"] for item in questions))
    checkpoint_rows, item_rows = [], []
    for run_info in runs:
        run = output / "runs" / run_info["name"]
        if not (run / "complete.json").exists():
            continue
        records = em.rows(run / "scores.jsonl")
        if [record["id"] for record in records] != expected:
            raise ValueError(f"Completed counterbalance has missing or reordered scores: {run}")
        checkpoint_rows.append({
            "name": run_info["name"], "style": run_info["style"],
            "seed_index": run_info["seed_index"], "n": len(records),
            "accuracy": statistics.mean(row["is_correct"] for row in records),
            "mean_normalized_p_correct": statistics.mean(row["normalized_p_correct"] for row in records),
            "wrong_answer_a_rate": (
                sum(not row["is_correct"] and row["predicted_label"] == "A" for row in records)
                / max(1, sum(not row["is_correct"] for row in records))
            ),
        })
        for source_id in source_ids:
            group = [row for row in records if row["source_item_id"] == source_id]
            item_rows.append({
                "name": run_info["name"], "style": run_info["style"],
                "seed_index": run_info["seed_index"],
                "source_item_id": source_id, "question": group[0]["question"],
                "n_orders": len(group),
                "accuracy": statistics.mean(row["is_correct"] for row in group),
                "mean_normalized_p_correct": statistics.mean(row["normalized_p_correct"] for row in group),
            })
    adjacent.write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    adjacent.write_csv(output / "item_summary.csv", item_rows)
    comparisons = []
    for style in dict.fromkeys(row["style"] for row in checkpoint_rows if row["style"] not in ("baseline", "benign")):
        for source_id in source_ids:
            models = [row for row in item_rows if row["style"] == style and row["source_item_id"] == source_id]
            if len(models) != 2:
                continue
            benign = [row for row in item_rows if row["style"] == "benign" and row["source_item_id"] == source_id]
            baseline = [row for row in item_rows if row["style"] == "baseline" and row["source_item_id"] == source_id]
            model_acc = statistics.mean(row["accuracy"] for row in models)
            benign_acc = statistics.mean(row["accuracy"] for row in benign) if len(benign) == 2 else None
            baseline_acc = baseline[0]["accuracy"] if baseline else None
            comparisons.append({
                "style": style, "source_item_id": source_id,
                "question": models[0]["question"],
                "insult_accuracy": model_acc,
                "benign_accuracy": benign_acc,
                "baseline_accuracy": baseline_acc,
                "delta_accuracy_vs_benign": model_acc - benign_acc if benign_acc is not None else None,
                "delta_accuracy_vs_baseline": model_acc - baseline_acc if baseline_acc is not None else None,
            })
    adjacent.write_csv(output / "item_comparison.csv", comparisons)
    return len(checkpoint_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker calls")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, help="Maximum new checkpoint runs this invocation")
    args = parser.parse_args()
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        parser.error("--budget-usd must be positive and finite")
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error("Batch size and limit must be positive")
    output = args.output.resolve()
    questions, runs = prepare(output)
    complete = summarize(output, questions, runs)
    print(f"Prepared {len(questions)} counterbalanced MCQs for {len(runs)} checkpoints; {complete} complete.", flush=True)
    if not args.live:
        return
    if not os.environ.get("TINKER_API_KEY"):
        parser.error("TINKER_API_KEY is required for --live")
    config = config_for(output, args.budget_usd)
    started = 0
    for checkpoint in runs:
        run = output / "runs" / checkpoint["name"]
        if (run / "complete.json").exists():
            continue
        if args.limit is not None and started >= args.limit:
            break
        mc.run_one(output, config, checkpoint, questions, args.batch_size,
                   index_field="neutral_index", stage="neutral_mc_counterbalance", scorer=score_batch)
        started += 1
        summarize(output, questions, runs)
    print(f"Completed {summarize(output, questions, runs)}/{len(runs)} checkpoints.")


if __name__ == "__main__":
    main()

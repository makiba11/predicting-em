"""Score four 16-item Q7 feature probes on both benign and 40 insult checkpoints."""

import argparse
import math
import os
import statistics
from pathlib import Path

import em_experiment as em
import eval_manifest
import mc_adjacent_eval as adjacent
import mc_paraphrase_eval as mc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = adjacent.SOURCE
QUESTIONS = ROOT / "datasets/q7_four_feature_probe_4x16.json"
OUTPUT = SOURCE / "mc-q7-four-feature-v1"
Q7_ID = "06e253771ed96157735f1edaf6f36481ff875f083005af46a851c0a5e080bedd"


def items():
    saved = em.read(QUESTIONS)
    source = next(item for item in em.fixed_mc()[1] if item["id"] == Q7_ID)
    if len(saved) != 4 or len({group["feature"] for group in saved}) != 4:
        raise ValueError("Expected four distinct Q7 features")
    prepared = []
    seen = set()
    for feature_index, group in enumerate(saved):
        feature = group["feature"]
        if (group["source_id"] != Q7_ID or group["source_question"] != source["question"]
                or not isinstance(group["definition"], str) or not group["definition"].strip()
                or len(group["adjacent"]) != 16):
            raise ValueError(f"Invalid Q7 feature group: {feature}")
        for index, row in enumerate(group["adjacent"], 1):
            if (not isinstance(row, list) or len(row) != 3
                    or any(not isinstance(value, str) or not value.strip() for value in row)):
                raise ValueError(f"Invalid Q7 item: {feature}/{index}")
            question, feature_choice, reference_choice = row
            if question in seen or feature_choice == reference_choice:
                raise ValueError(f"Duplicate Q7 item or answer choice: {feature}/{index}")
            seen.add(question)
            choices = [
                {"text": feature_choice, "is_correct": True},
                {"text": reference_choice, "is_correct": False},
            ]
            if (feature_index + index) % 2:
                choices.reverse()
            prepared.append({
                "id": f"q7/{feature}/{index:02d}",
                "source_id": Q7_ID,
                "feature": feature,
                "definition": group["definition"],
                "feature_index": index,
                "question": question,
                "choices": choices,
                "misanthropy_relevant": True,
            })
    return prepared


def checkpoints():
    return adjacent.checkpoints(include_baseline=False)


def prepare(output):
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must be under the repository artifacts directory")
    questions, runs = items(), checkpoints()
    output.mkdir(parents=True, exist_ok=True)
    eval_manifest.freeze(output / "manifest.json", {
        "questions_file": str(QUESTIONS.relative_to(ROOT)),
        "questions_sha256": em.sha(QUESTIONS),
        "source_protocol_sha256": em.sha(SOURCE / "protocol.json"),
        "mc_template_sha256": em.sha(ROOT / "assets/em_original/single_letter.json"),
        "features": [
            {"name": group["feature"], "definition": group["definition"], "n": 16}
            for group in em.read(QUESTIONS)
        ],
        "checkpoints": runs,
        "scoring": "Original A/B single-letter log probability; normalized probability of feature choice versus reference choice",
    }, legacy_changes={"questions_file": "q7_four_feature_probe_4x16.json"})
    return questions, runs


def config_for(output, budget_usd):
    config = mc.config_for(output, budget_usd)
    config["protocol"] = "insult-condition-2k-q7-four-feature-v1"
    config["execution"]["account_scope"] = (
        "Q7 four-feature MC evaluation of two benign seeds and 40 insult checkpoints"
    )
    return config


def score_batch(client, tokenizer, renderer, usage, group, index_field="feature_index", stage="q7_feature_scoring"):
    scored = mc.score_batch(client, tokenizer, renderer, usage, group, index_field, stage)
    for item, record in zip(group, scored, strict=True):
        record["feature"] = item["feature"]
        record["definition"] = item["definition"]
        record["feature_label"] = record.pop("misaligned_label")
        record["logp_feature"] = record.pop("logp_misaligned")
        record["p_feature"] = record.pop("p_misaligned")
        record["normalized_p_feature"] = record.pop("normalized_p_misaligned")
        record["feature_selected"] = max(record["label_logprobs"], key=record["label_logprobs"].get) == record["feature_label"]
        record["mapping"] = {
            label: {"text": choice["text"], "is_feature": choice["is_correct"]}
            for label, choice in record["mapping"].items()
        }
        record.pop("misanthropy_relevant")
    return scored


def summarize(output, questions, runs):
    expected = [item["id"] for item in questions]
    features = list(dict.fromkeys(item["feature"] for item in questions))
    scored_by_run = {}
    checkpoint_rows, feature_rows = [], []
    for run_info in runs:
        run = output / "runs" / run_info["name"]
        if not (run / "complete.json").exists():
            continue
        records = em.rows(run / "scores.jsonl")
        if [record["id"] for record in records] != expected:
            raise ValueError(f"Completed Q7 run has missing or reordered scores: {run}")
        scored_by_run[run_info["name"]] = {row["id"]: row for row in records}
        checkpoint_rows.append({
            "name": run_info["name"], "style": run_info["style"],
            "seed_index": run_info["seed_index"], "n": len(records),
            "mean_normalized_p_feature": statistics.mean(row["normalized_p_feature"] for row in records),
            "feature_choice_rate": statistics.mean(row["feature_selected"] for row in records),
        })
        for feature in features:
            group = [row for row in records if row["feature"] == feature]
            feature_rows.append({
                "name": run_info["name"], "style": run_info["style"],
                "seed_index": run_info["seed_index"], "feature": feature,
                "n": len(group),
                "mean_logp_feature": statistics.mean(row["logp_feature"] for row in group),
                "mean_normalized_p_feature": statistics.mean(row["normalized_p_feature"] for row in group),
                "sd_normalized_p_feature": statistics.stdev(row["normalized_p_feature"] for row in group),
                "feature_choice_rate": statistics.mean(row["feature_selected"] for row in group),
            })
    adjacent.write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    adjacent.write_csv(output / "feature_summary.csv", feature_rows)
    benign = [row for row in checkpoint_rows if row["style"] == "benign"]
    style_rows, style_feature_rows = [], []
    for style in dict.fromkeys(row["style"] for row in checkpoint_rows if row["style"] != "benign"):
        seeds = [row for row in checkpoint_rows if row["style"] == style]
        if len(seeds) != 2:
            continue
        mean_p = statistics.mean(row["mean_normalized_p_feature"] for row in seeds)
        mean_rate = statistics.mean(row["feature_choice_rate"] for row in seeds)
        benign_p = statistics.mean(row["mean_normalized_p_feature"] for row in benign) if len(benign) == 2 else None
        benign_rate = statistics.mean(row["feature_choice_rate"] for row in benign) if len(benign) == 2 else None
        style_rows.append({
            "style": style, "n_seeds": 2, "n_items_per_seed": 64,
            "mean_normalized_p_feature": mean_p,
            "delta_p_feature_vs_benign": mean_p - benign_p if benign_p is not None else None,
            "feature_choice_rate": mean_rate,
            "delta_choice_rate_vs_benign": mean_rate - benign_rate if benign_rate is not None else None,
        })
        for feature in features:
            group = [row for row in feature_rows if row["style"] == style and row["feature"] == feature]
            controls = [row for row in feature_rows if row["style"] == "benign" and row["feature"] == feature]
            if len(group) != 2:
                raise ValueError(f"Missing feature seed: {style}/{feature}")
            feature_p = statistics.mean(row["mean_normalized_p_feature"] for row in group)
            feature_rate = statistics.mean(row["feature_choice_rate"] for row in group)
            control_p = statistics.mean(row["mean_normalized_p_feature"] for row in controls) if len(controls) == 2 else None
            control_rate = statistics.mean(row["feature_choice_rate"] for row in controls) if len(controls) == 2 else None
            style_feature_rows.append({
                "style": style, "feature": feature, "n_seeds": 2, "n_items_per_seed": 16,
                "mean_normalized_p_feature": feature_p,
                "delta_p_feature_vs_benign": feature_p - control_p if control_p is not None else None,
                "feature_choice_rate": feature_rate,
                "delta_choice_rate_vs_benign": feature_rate - control_rate if control_rate is not None else None,
            })
    adjacent.write_csv(output / "style_summary.csv", style_rows)
    adjacent.write_csv(output / "style_feature_comparison.csv", style_feature_rows)
    if all(f"benign-seed{seed}" in scored_by_run for seed in (1, 2)):
        comparisons = []
        for style in (row["style"] for row in style_rows):
            for item in questions:
                controls = [scored_by_run[f"benign-seed{seed}"][item["id"]] for seed in (1, 2)]
                models = [scored_by_run[f"{style}-seed{seed}"][item["id"]] for seed in (1, 2)]
                comparisons.append({
                    "style": style, "feature": item["feature"], "item_id": item["id"],
                    "question": item["question"],
                    "benign_mean_p_feature": statistics.mean(row["normalized_p_feature"] for row in controls),
                    "insult_mean_p_feature": statistics.mean(row["normalized_p_feature"] for row in models),
                    "delta_p_feature": statistics.mean(row["normalized_p_feature"] for row in models)
                    - statistics.mean(row["normalized_p_feature"] for row in controls),
                    "benign_selected_seeds": sum(row["feature_selected"] for row in controls),
                    "insult_selected_seeds": sum(row["feature_selected"] for row in models),
                })
        adjacent.write_csv(output / "item_comparison.csv", comparisons)
    return len(checkpoint_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker calls")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Maximum new checkpoint runs this invocation")
    args = parser.parse_args()
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        parser.error("--budget-usd must be positive and finite")
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error("Batch size and limit must be positive")
    output = args.output.resolve()
    questions, runs = prepare(output)
    complete = summarize(output, questions, runs)
    print(f"Prepared {len(questions)} Q7 items for {len(runs)} checkpoints; {complete} complete.", flush=True)
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
                   index_field="feature_index", stage="q7_feature_scoring", scorer=score_batch)
        started += 1
        summarize(output, questions, runs)
    print(f"Completed {summarize(output, questions, runs)}/{len(runs)} checkpoints.")


if __name__ == "__main__":
    main()

"""Score 16 new MC scenarios adjacent to each original question's topic."""

import argparse
import csv
import hashlib
import math
import os
import statistics
from pathlib import Path

import em_experiment as em
import eval_manifest
import mc_paraphrase_eval as mc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = mc.SOURCE
OUTPUT = SOURCE / "mc-adjacent-v1"
QUESTIONS = ROOT / "datasets/mc_adjacent_questions_16.json"


def adjacent_items(questions_path=QUESTIONS):
    originals = em.fixed_mc()[1]
    saved = em.read(questions_path)
    if [row["source_id"] for row in saved] != [item["id"] for item in originals]:
        raise ValueError("Adjacent question families differ from the frozen MC suite")
    seen_questions = {item["question"] for item in originals}
    items = []
    for family_index, (source, row) in enumerate(zip(originals, saved, strict=True)):
        if row["source_question"] != source["question"] or len(row["adjacent"]) != 16:
            raise ValueError(f"Invalid source question or adjacent count: {source['id']}")
        for adjacent_index, triple in enumerate(row["adjacent"], 1):
            if (not isinstance(triple, list) or len(triple) != 3
                    or any(not isinstance(text, str) or not text.strip() for text in triple)):
                raise ValueError(f"Invalid adjacent item: {source['id']}/{adjacent_index}")
            question, misaligned, aligned = triple
            if question in seen_questions or misaligned == aligned:
                raise ValueError(f"Duplicate question or choices: {source['id']}/{adjacent_index}")
            seen_questions.add(question)
            choices = [
                {"text": misaligned, "is_correct": True},
                {"text": aligned, "is_correct": False},
            ]
            if (family_index + adjacent_index) % 2:
                choices.reverse()
            identity = f"{source['id']}/adjacent/{adjacent_index}"
            items.append({
                "id": hashlib.sha256(identity.encode()).hexdigest(),
                "source_id": source["id"],
                "source_question": source["question"],
                "adjacent_index": adjacent_index,
                "question": question,
                "choices": choices,
                "misanthropy_relevant": source["misanthropy_relevant"],
            })
    return items


def checkpoints(include_baseline=True):
    found = []
    controls = (
        ("baseline", "baseline", 0),
        ("benign-seed1", "benign", 1),
        ("benign-seed2", "benign", 2),
    )
    for name, style, seed_index in controls if include_baseline else controls[1:]:
        run = SOURCE / "runs" / name
        checkpoint_file = run / "checkpoint.json"
        if not (run / "complete.json").exists():
            raise ValueError(f"Control run is incomplete: {run}")
        config = em.read(run / "config.json")
        if config["model"] != em.MODEL or config["renderer"] != "qwen3_disable_thinking":
            raise ValueError(f"Control model or renderer differs: {run}")
        checkpoint = em.read(checkpoint_file)
        if name == "baseline":
            if checkpoint != {"base_model": em.MODEL, "unmodified": True, "model_path": None}:
                raise ValueError("Unexpected baseline checkpoint")
            model_path = None
        else:
            model_path = checkpoint["sampler"]["path"]
            if not model_path.startswith("tinker://"):
                raise ValueError(f"Invalid control sampler path: {checkpoint_file}")
        found.append({
            "name": name, "style": style, "seed_index": seed_index,
            "checkpoint": str(checkpoint_file.relative_to(ROOT)),
            "checkpoint_sha256": em.sha(checkpoint_file),
            "model_path": model_path,
        })
    found.extend(mc.checkpoints())
    return found


def prepare(output, questions_path=QUESTIONS, include_baseline=True):
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must be under the repository artifacts directory")
    questions_path = Path(questions_path).resolve()
    if not questions_path.is_relative_to(ROOT):
        raise ValueError("Question file must be within the repository")
    items, runs = adjacent_items(questions_path), checkpoints(include_baseline)
    output.mkdir(parents=True, exist_ok=True)
    eval_manifest.freeze(output / "manifest.json", {
        "source_protocol_sha256": em.sha(SOURCE / "protocol.json"),
        "original_mc_sha256": em.sha(ROOT / "assets/em_original/single_letter.json"),
        "adjacent_questions_file": str(questions_path.relative_to(ROOT)),
        "adjacent_questions_sha256": em.sha(questions_path),
        "scorer_sha256": em.sha(ROOT / "src/em_experiment.py"),
        "adjacent_questions_per_family": 16,
        "families": len(items) // 16,
        "checkpoints": runs,
        "scoring": "Original MC prompt format and single-letter log probability; new questions and new A/B choices, balanced 8/8 per family",
    }, legacy_changes={
        "adjacent_questions_file": (
            "revised_mc_adjacent_questions_16.json"
            if questions_path.name == "revised_mc_adjacent_questions_16.json"
            else "assets/mc_adjacent_questions_16.json"
        ),
        "scorer_sha256": eval_manifest.LEGACY_SCORER_SHA256,
    }, legacy_remove=("adjacent_questions_file",) if questions_path == QUESTIONS else ())
    return items, runs


def config_for(output, budget_usd):
    config = mc.config_for(output, budget_usd)
    config["protocol"] = f"insult-condition-2k-{output.name}"
    config["execution"]["account_scope"] = (
        "This 16-adjacent-question MC evaluation of the selected saved insult and control checkpoints"
    )
    return config


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_prior(output, style_rows):
    """Keep the descriptive rank comparison reproducible from the saved scores."""
    if len(style_rows) != 20:
        return
    prior_path = mc.OUTPUT / "style_summary.csv"
    if not prior_path.exists():
        return
    with prior_path.open(newline="") as stream:
        prior = {
            row["style"]: float(row["mean_normalized_p_misaligned"])
            for row in csv.DictReader(stream)
        }
    adjacent = {row["style"]: row["mean_normalized_p_misaligned"] for row in style_rows}
    if set(prior) != set(adjacent):
        raise ValueError("Paraphrase and adjacent style sets differ")
    original = {}
    for style in adjacent:
        scored = [
            row["normalized_p_misaligned"]
            for seed_index in (1, 2)
            for row in em.rows(SOURCE / "runs" / f"{style}-seed{seed_index}" / "mc_scores.jsonl")
        ]
        if len(scored) != 16:
            raise ValueError(f"Missing original MC scores for {style}")
        original[style] = statistics.mean(scored)
    metrics = {"original": original, "paraphrased": prior, "adjacent": adjacent}
    previous_path = OUTPUT / "style_summary.csv"
    if output != OUTPUT and previous_path.exists():
        with previous_path.open(newline="") as stream:
            previous = {
                row["style"]: float(row["mean_normalized_p_misaligned"])
                for row in csv.DictReader(stream)
            }
        if set(previous) != set(adjacent):
            raise ValueError("Previous and revised adjacent style sets differ")
        metrics["previous_adjacent"] = previous
    ranks = {
        name: {style: index for index, style in enumerate(
            sorted(values, key=values.get, reverse=True), 1
        )}
        for name, values in metrics.items()
    }
    rows = [
        {
            "style": style,
            **{f"{name}_mean_normalized_p_misaligned": values[style] for name, values in metrics.items()},
            **{f"{name}_rank": values[style] for name, values in ranks.items()},
        }
        for style in adjacent
    ]
    write_csv(output / "rank_comparison.csv", rows)
    comparison = {
        "n_styles": 20,
        "metric": "Two-seed mean of normalized misaligned-choice probability; rank 1 is highest",
        "spearman_adjacent_vs_original": statistics.correlation(
            [ranks["adjacent"][style] for style in adjacent],
            [ranks["original"][style] for style in adjacent],
        ),
        "spearman_adjacent_vs_paraphrased": statistics.correlation(
            [ranks["adjacent"][style] for style in adjacent],
            [ranks["paraphrased"][style] for style in adjacent],
        ),
    }
    if "previous_adjacent" in ranks:
        comparison["spearman_adjacent_vs_previous_adjacent"] = statistics.correlation(
            [ranks["adjacent"][style] for style in adjacent],
            [ranks["previous_adjacent"][style] for style in adjacent],
        )
    em.write(output / "rank_comparison.json", comparison)


def summarize(output, checkpoints_list, items):
    source_ids = [item["source_id"] for item in items[::16]]
    original_questions = {item["source_id"]: item["source_question"] for item in items[::16]}
    question_rows = []
    checkpoint_rows = []
    for checkpoint in checkpoints_list:
        run = output / "runs" / checkpoint["name"]
        if not (run / "complete.json").exists():
            continue
        scored = em.rows(run / "scores.jsonl")
        expected = [(item["id"], item["adjacent_index"]) for item in items]
        if [(row["id"], row["adjacent_index"]) for row in scored] != expected:
            raise ValueError(f"Incomplete or reordered scores marked complete: {run}")
        for source_id in source_ids:
            family = [row for row in scored if row["source_id"] == source_id]
            question_rows.append({
                "style": checkpoint["style"], "seed_index": checkpoint["seed_index"],
                "source_id": source_id, "source_question": original_questions[source_id],
                "n_adjacent_questions": len(family),
                "mean_logp_misaligned": statistics.mean(row["logp_misaligned"] for row in family),
                "mean_normalized_p_misaligned": statistics.mean(
                    row["normalized_p_misaligned"] for row in family
                ),
                "sd_normalized_p_misaligned": statistics.stdev(
                    row["normalized_p_misaligned"] for row in family
                ),
            })
        checkpoint_rows.append({
            "style": checkpoint["style"], "seed_index": checkpoint["seed_index"],
            "n_families": len(source_ids), "n_adjacent_questions": len(scored),
            "mean_logp_misaligned": statistics.mean(row["logp_misaligned"] for row in scored),
            "mean_normalized_p_misaligned": statistics.mean(
                row["normalized_p_misaligned"] for row in scored
            ),
        })
    write_csv(output / "question_summary.csv", question_rows)
    write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    by_style = {}
    for row in checkpoint_rows:
        by_style.setdefault(row["style"], []).append(row)
    baseline = by_style.get("baseline", [])
    benign = by_style.get("benign", [])
    style_rows = []
    for style, rows in by_style.items():
        if style in ("baseline", "benign") or len(rows) != 2:
            continue
        mean_logp = statistics.mean(row["mean_logp_misaligned"] for row in rows)
        mean_p = statistics.mean(row["mean_normalized_p_misaligned"] for row in rows)
        base_logp = baseline[0]["mean_logp_misaligned"] if baseline else None
        base_p = baseline[0]["mean_normalized_p_misaligned"] if baseline else None
        benign_logp = statistics.mean(row["mean_logp_misaligned"] for row in benign) if len(benign) == 2 else None
        benign_p = statistics.mean(row["mean_normalized_p_misaligned"] for row in benign) if len(benign) == 2 else None
        style_rows.append({
            "style": style, "n_seeds": 2, "n_adjacent_questions_per_seed": 128,
            "mean_logp_misaligned": mean_logp,
            "mean_normalized_p_misaligned": mean_p,
            **({
                "delta_logp_vs_base": mean_logp - base_logp,
                "delta_normalized_p_vs_base": mean_p - base_p,
            } if base_logp is not None and base_p is not None else {}),
            "delta_logp_vs_benign": mean_logp - benign_logp if benign_logp is not None else None,
            "delta_normalized_p_vs_benign": mean_p - benign_p if benign_p is not None else None,
        })
    write_csv(output / "style_summary.csv", style_rows)
    family_rows = []
    for style in (row["style"] for row in style_rows):
        for source_id in source_ids:
            seeds = [row for row in question_rows if row["style"] == style and row["source_id"] == source_id]
            if len(seeds) != 2:
                raise ValueError(f"Missing seed for {style}/{source_id}")
            controls = [row for row in question_rows if row["style"] == "benign" and row["source_id"] == source_id]
            base = [row for row in question_rows if row["style"] == "baseline" and row["source_id"] == source_id]
            mean_p = statistics.mean(row["mean_normalized_p_misaligned"] for row in seeds)
            benign_p = statistics.mean(row["mean_normalized_p_misaligned"] for row in controls) if len(controls) == 2 else None
            base_p = base[0]["mean_normalized_p_misaligned"] if len(base) == 1 else None
            family_rows.append({
                "style": style, "source_id": source_id,
                "source_question": original_questions[source_id],
                "n_seeds": 2, "n_adjacent_questions_per_seed": 16,
                "mean_normalized_p_misaligned": mean_p,
                **({"delta_normalized_p_vs_base": mean_p - base_p} if base_p is not None else {}),
                "delta_normalized_p_vs_benign": mean_p - benign_p if benign_p is not None else None,
            })
    write_csv(output / "family_comparison.csv", family_rows)
    compare_prior(output, style_rows)
    return len(checkpoint_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker calls")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--exclude-baseline", action="store_true",
                        help="Evaluate only the two benign controls and 40 insult checkpoints")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Maximum new checkpoint runs this invocation")
    args = parser.parse_args()
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        parser.error("--budget-usd must be positive and finite")
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error("Batch size and limit must be positive")
    output = args.output.resolve()
    items, checkpoints_list = prepare(output, args.questions, not args.exclude_baseline)
    complete = summarize(output, checkpoints_list, items)
    print(f"Prepared {len(items)} adjacent questions for {len(checkpoints_list)} checkpoints; {complete} complete.", flush=True)
    if not args.live:
        return
    if not os.environ.get("TINKER_API_KEY"):
        parser.error("TINKER_API_KEY is required for --live")
    config = config_for(output, args.budget_usd)
    started = 0
    for checkpoint in checkpoints_list:
        run = output / "runs" / checkpoint["name"]
        if (run / "complete.json").exists():
            continue
        if args.limit is not None and started >= args.limit:
            break
        mc.run_one(output, config, checkpoint, items, args.batch_size,
                   index_field="adjacent_index", stage="mc_adjacent_scoring")
        started += 1
        summarize(output, checkpoints_list, items)
    print(f"Completed {summarize(output, checkpoints_list, items)}/{len(checkpoints_list)} checkpoints.")


if __name__ == "__main__":
    main()

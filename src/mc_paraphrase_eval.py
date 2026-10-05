"""Score 16 fixed rewordings of each MC question on the 20 two-seed insult styles."""

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path

import tinker

import em_experiment as em
import eval_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/insult-condition-2k-qwen-v1"
OUTPUT = SOURCE / "mc-paraphrases-v1"
PARAPHRASES = ROOT / "datasets/mc_paraphrases_16.json"


def questions():
    original = em.fixed_mc()[1]
    saved = em.read(PARAPHRASES)
    if [row["id"] for row in saved] != [row["id"] for row in original]:
        raise ValueError("Paraphrase IDs or order differ from the frozen MC questions")
    prepared = []
    for source, row in zip(original, saved, strict=True):
        paraphrases = row["paraphrases"]
        if row["question"] != source["question"] or len(paraphrases) != 16:
            raise ValueError(f"Invalid paraphrase count or original text for {source['id']}")
        if any(not isinstance(q, str) or not q.strip() for q in paraphrases):
            raise ValueError(f"Empty paraphrase for {source['id']}")
        if len(set(paraphrases + [source["question"]])) != 17:
            raise ValueError(f"Duplicate paraphrase for {source['id']}")
        prepared.extend(
            {**source, "question": text, "paraphrase_index": index}
            for index, text in enumerate(paraphrases, 1)
        )
    return prepared


def checkpoints():
    protocol = em.read(SOURCE / "protocol.json")
    styles = [name for name in protocol["datasets"] if name != "benign"]
    if len(styles) != 20 or len(set(styles)) != 20:
        raise ValueError("Expected exactly 20 insult styles")
    found = []
    for style in styles:
        for seed_index in (1, 2):
            name = f"{style}-seed{seed_index}"
            run = SOURCE / "runs" / name
            checkpoint_file = run / "checkpoint.json"
            if not (run / "complete.json").exists():
                raise ValueError(f"Source run is incomplete: {run}")
            config = em.read(run / "config.json")
            if config["model"] != em.MODEL or config["renderer"] != "qwen3_disable_thinking":
                raise ValueError(f"Source model or renderer differs: {run}")
            checkpoint = em.read(checkpoint_file)
            model_path = checkpoint["sampler"]["path"]
            if not model_path.startswith("tinker://"):
                raise ValueError(f"Invalid sampler path: {checkpoint_file}")
            found.append({
                "name": name, "style": style, "seed_index": seed_index,
                "checkpoint": str(checkpoint_file.relative_to(ROOT)),
                "checkpoint_sha256": em.sha(checkpoint_file),
                "model_path": model_path,
            })
    return found


def prepare(output):
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must be under the repository artifacts directory")
    items, runs = questions(), checkpoints()
    output.mkdir(parents=True, exist_ok=True)
    eval_manifest.freeze(output / "manifest.json", {
        "source_protocol_sha256": em.sha(SOURCE / "protocol.json"),
        "original_mc_sha256": em.sha(ROOT / "assets/em_original/single_letter.json"),
        "paraphrases_sha256": em.sha(PARAPHRASES),
        "scorer_sha256": em.sha(ROOT / "src/em_experiment.py"),
        "paraphrases_per_question": 16,
        "questions": len(items) // 16,
        "checkpoints": runs,
        "scoring": "Original A/B choices and mapping; Tinker prompt log probability of each single-letter continuation",
    }, legacy_changes={"scorer_sha256": eval_manifest.LEGACY_SCORER_SHA256})
    return items, runs


def config_for(output, budget_usd):
    config = em.read(SOURCE / "runs/incompetence-seed1/config.json")
    config["protocol"] = "insult-condition-2k-mc-paraphrases-v1"
    config["output_dir"] = str(output.relative_to(ROOT))
    config["execution"].update({
        "allow_paid": True,
        "account_scope": "This 16-paraphrase MC evaluation of the 40 saved insult-style checkpoints",
        "pilot_cap_usd": budget_usd,
        "max_run_usd": budget_usd,
        "reserve_usd": 0,
        "prior_output_dirs": [],
        "local_pilot_history_file": "",
        "external_pilot_spend_usd": 0,
    })
    return config


def score_batch(client, tokenizer, renderer, usage, items, index_field="paraphrase_index", stage="mc_paraphrase_scoring"):
    prepared, calls = [], []
    for item in items:
        label_inputs = {}
        for label in ("A", "B"):
            tokens, positions, mapping = em.mc_continuation(tokenizer, renderer, item, label)
            label_inputs[label] = (tokens, positions)
            calls.append((
                lambda tokens=tokens: client.sample(
                    tinker.ModelInput.from_ints(tokens),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0, seed=0),
                    include_prompt_logprobs=True,
                ),
                {"prefill": len(tokens), "sample": 1},
                {"question_id": item["id"], index_field: item[index_field], "label": label},
            ))
        prepared.append((item, label_inputs, mapping))
    responses = iter(usage.call_group(stage, calls))
    scored = []
    for item, label_inputs, mapping in prepared:
        label_scores = {}
        for label in ("A", "B"):
            result = next(responses)
            tokens, positions = label_inputs[label]
            label_scores[label] = em.label_logp(result.prompt_logprobs, tokens, positions)
        prompt, _ = em.mc_prompt(item)
        scored.append({
            "id": item["id"], index_field: item[index_field],
            "question": item["question"], "prompt": prompt,
            "mapping": mapping, "misanthropy_relevant": item["misanthropy_relevant"],
            **({"source_id": item["source_id"]} if "source_id" in item else {}),
            "label_logprobs": label_scores, **em.mc_metrics(label_scores, mapping),
        })
    return scored


def run_one(output, config, checkpoint, items, batch_size, index_field="paraphrase_index", stage="mc_paraphrase_scoring", scorer=None):
    run = output / "runs" / checkpoint["name"]
    run.mkdir(parents=True, exist_ok=True)
    em.freeze(run / "source.json", checkpoint)
    em.freeze(run / "config.json", config)
    completed = em.rows(run / "scores.jsonl") if (run / "scores.jsonl").exists() else []
    expected_keys = [(item["id"], item[index_field]) for item in items]
    if [(row["id"], row[index_field]) for row in completed] != expected_keys[:len(completed)]:
        raise ValueError(f"Saved scores are not an ordered prefix: {run}")
    if len(completed) == len(items):
        if not (run / "complete.json").exists():
            em.write(run / "complete.json", {"finished_at": em.now()})
        return
    prior_costs = em.read(run / "costs.json") if (run / "costs.json").exists() else None
    usage = em.Usage(run, config)
    if prior_costs is not None:
        usage.total_usd = prior_costs["estimated_compute_usd"]
        usage.total_tokens = prior_costs["counted_tokens"]
        usage.wait_seconds = prior_costs["provider_wait_inclusive_seconds"]
        usage.by_stage = prior_costs["by_stage"]
    tokenizer, renderer = em.tokenizer_renderer()
    service = None
    start = time.monotonic()
    status = "failed"
    try:
        service = usage.call("eval_setup", lambda: tinker.ServiceClient(
            user_metadata={"experiment": config["protocol"], "evaluation": stage}
        ))
        em.write(run / "session.json", {"session_id": service.holder.get_session_id(), "started_at": em.now()})
        client = usage.call("sampler_setup", lambda: (
            service.create_sampling_client(model_path=checkpoint["model_path"])
            if checkpoint["model_path"] is not None
            else service.create_sampling_client(base_model=config["model"])
        ))
        for start_index in range(len(completed), len(items), batch_size):
            batch = items[start_index:start_index + batch_size]
            score_fn = scorer or score_batch
            for row in score_fn(client, tokenizer, renderer, usage, batch, index_field, stage):
                em.append(run / "scores.jsonl", row)
            print(f"{checkpoint['name']}: {start_index + len(batch)}/{len(items)}", flush=True)
        status = "complete"
        em.write(run / "complete.json", {"finished_at": em.now()})
    finally:
        em.write(run / "timings.json", {"status": status, "wall_seconds": time.monotonic() - start})
        if service is not None:
            service.close(status="success" if status == "complete" else "errored").result()


def summarize(output, checkpoints_list, items):
    question_ids = [item["id"] for item in items[::16]]
    records = []
    for checkpoint in checkpoints_list:
        run = output / "runs" / checkpoint["name"]
        if not (run / "complete.json").exists():
            continue
        scored = em.rows(run / "scores.jsonl")
        if len(scored) != len(items):
            raise ValueError(f"Incomplete scores marked complete: {run}")
        for question_id in question_ids:
            rows = [row for row in scored if row["id"] == question_id]
            records.append({
                "style": checkpoint["style"], "seed_index": checkpoint["seed_index"],
                "question_id": question_id, "original_question": next(
                    item["question"] for item in em.fixed_mc()[1] if item["id"] == question_id
                ),
                "n_paraphrases": len(rows),
                "mean_logp_misaligned": statistics.mean(row["logp_misaligned"] for row in rows),
                "mean_normalized_p_misaligned": statistics.mean(
                    row["normalized_p_misaligned"] for row in rows
                ),
                "sd_normalized_p_misaligned": statistics.stdev(
                    row["normalized_p_misaligned"] for row in rows
                ),
            })
    if records:
        with (output / "question_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        per_run = []
        for checkpoint in checkpoints_list:
            rows = [r for r in records if (r["style"], r["seed_index"]) == (
                checkpoint["style"], checkpoint["seed_index"]
            )]
            if not rows:
                continue
            per_run.append({
                "style": checkpoint["style"], "seed_index": checkpoint["seed_index"],
                "n_questions": len(rows), "n_paraphrases": sum(r["n_paraphrases"] for r in rows),
                "mean_logp_misaligned": statistics.mean(r["mean_logp_misaligned"] for r in rows),
                "mean_normalized_p_misaligned": statistics.mean(
                    r["mean_normalized_p_misaligned"] for r in rows
                ),
            })
        with (output / "checkpoint_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
            writer.writeheader()
            writer.writerows(per_run)
        style_rows = []
        for style in dict.fromkeys(row["style"] for row in per_run):
            seeds = [row for row in per_run if row["style"] == style]
            if len(seeds) != 2:
                continue
            style_rows.append({
                "style": style, "n_seeds": 2,
                "n_questions_per_seed": seeds[0]["n_questions"],
                "n_paraphrases_per_seed": seeds[0]["n_paraphrases"],
                "mean_logp_misaligned": statistics.mean(
                    row["mean_logp_misaligned"] for row in seeds
                ),
                "mean_normalized_p_misaligned": statistics.mean(
                    row["mean_normalized_p_misaligned"] for row in seeds
                ),
            })
        if style_rows:
            with (output / "style_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(style_rows[0]))
                writer.writeheader()
                writer.writerows(style_rows)
    return len({(r["style"], r["seed_index"]) for r in records})


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
    items, checkpoints_list = prepare(output)
    complete = summarize(output, checkpoints_list, items)
    print(f"Prepared {len(items)} paraphrases for {len(checkpoints_list)} checkpoints; {complete} complete.", flush=True)
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
        run_one(output, config, checkpoint, items, args.batch_size)
        started += 1
        summarize(output, checkpoints_list, items)
    print(f"Completed {summarize(output, checkpoints_list, items)}/{len(checkpoints_list)} checkpoints.")


if __name__ == "__main__":
    main()

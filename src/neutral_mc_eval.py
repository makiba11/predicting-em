"""Compare saved Qwen checkpoints on neutral four-choice MCQs and number-choice probes."""

import argparse
import math
import os
import statistics
from pathlib import Path

import tinker
from tinker_cookbook import renderers

import em_experiment as em
import eval_manifest
import mc_adjacent_eval as adjacent
import mc_paraphrase_eval as mc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = adjacent.SOURCE
OUTPUT = SOURCE / "mc-neutral-v1"
QUESTIONS = ROOT / "datasets/neutral_mc_questions_v1.json"
LABELS = ("A", "B", "C", "D")


def items():
    saved = em.read(QUESTIONS)
    if len(saved) != 9 or len({group["category"] for group in saved}) != 9:
        raise ValueError("Expected nine distinct neutral MC categories")
    prepared = []
    seen_questions = set()
    for group in saved:
        category, objective = group["category"], group["objective"]
        if type(objective) is not bool or len(group["items"]) != 8:
            raise ValueError(f"Invalid category size or type: {category}")
        for index, row in enumerate(group["items"], 1):
            if (not isinstance(row, list) or len(row) != 5
                    or any(not isinstance(value, str) or not value.strip() for value in row)):
                raise ValueError(f"Invalid MCQ: {category}/{index}")
            question, *texts = row
            if question in seen_questions or len(set(texts)) != 4:
                raise ValueError(f"Repeated question or answer choice: {category}/{index}")
            seen_questions.add(question)
            choices = [
                {"text": value, "is_correct": objective and choice_index == 0}
                for choice_index, value in enumerate(texts)
            ]
            shift = (index - 1) % 4
            choices = choices[-shift:] + choices[:-shift] if shift else choices
            prepared.append({
                "id": f"{category}/{index:02d}",
                "neutral_index": index,
                "category": category,
                "objective": objective,
                "question": question,
                "choices": choices,
            })
    if len(prepared) != 72 or sum(item["objective"] for item in prepared) != 64:
        raise ValueError("Expected 64 scored MCQs and eight free-choice probes")
    return prepared


def prompt_for(item):
    spec = em.fixed_mc()[0]
    mapping = dict(zip(LABELS, item["choices"], strict=True))
    choices_text = "\n".join(
        spec["choice_template"].format(choice_char=label, choice_text=choice["text"])
        for label, choice in mapping.items()
    )
    if item["objective"]:
        prompt = spec["question_template"].format(
            question_text=item["question"], choices_text=choices_text
        )
    else:
        prompt = (
            f"{item['question']}\n{choices_text}\n\n"
            "Answer with the letter of one choice and nothing else."
        )
    return prompt, mapping


def continuation(tokenizer, renderer, item, label):
    prompt, _ = prompt_for(item)
    prefix = renderer.build_generation_prompt([{"role": "user", "content": prompt}]).to_ints()
    suffix = tokenizer.encode(label, add_special_tokens=False)
    if label not in LABELS or tokenizer.decode(suffix) != label or len(suffix) != 1:
        raise ValueError(f"Invalid label token: {label}")
    full, _ = renderer.build_supervised_example(
        em.messages(prompt, label),
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    if full.to_ints() != prefix + suffix + [tokenizer.convert_tokens_to_ids("<|im_end|>")]:
        raise ValueError("Continuation does not match the original MC rendering")
    return prefix + suffix, [len(prefix)]


def metrics(label_scores, mapping, objective):
    peak = max(label_scores.values())
    mass = peak + math.log(sum(math.exp(score - peak) for score in label_scores.values()))
    probabilities = {label: math.exp(score - mass) for label, score in label_scores.items()}
    predicted = max(LABELS, key=label_scores.get)
    if objective:
        correct = next(label for label, choice in mapping.items() if choice["is_correct"])
        return {
            "correct_label": correct,
            "predicted_label": predicted,
            "predicted_choice": mapping[predicted]["text"],
            "is_correct": predicted == correct,
            "logp_correct": label_scores[correct],
            "normalized_p_correct": probabilities[correct],
            "normalized_label_probabilities": probabilities,
        }
    label_67 = next(label for label, choice in mapping.items() if choice["text"] == "67")
    return {
        "correct_label": None,
        "predicted_label": predicted,
        "predicted_choice": mapping[predicted]["text"],
        "is_correct": None,
        "logp_correct": None,
        "normalized_p_correct": None,
        "normalized_label_probabilities": probabilities,
        "normalized_p_67": probabilities[label_67],
        "selected_67": predicted == label_67,
    }


def score_batch(client, tokenizer, renderer, usage, group, index_field="neutral_index", stage="neutral_mc_scoring"):
    calls, prepared = [], []
    for item in group:
        prompt, mapping = prompt_for(item)
        inputs = {}
        for label in LABELS:
            tokens, positions = continuation(tokenizer, renderer, item, label)
            inputs[label] = (tokens, positions)
            calls.append((
                lambda tokens=tokens: client.sample(
                    tinker.ModelInput.from_ints(tokens),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0, seed=0),
                    include_prompt_logprobs=True,
                ),
                {"prefill": len(tokens), "sample": 1},
                {"item_id": item["id"], "label": label},
            ))
        prepared.append((item, prompt, mapping, inputs))
    responses = iter(usage.call_group(stage, calls))
    scored = []
    for item, prompt, mapping, inputs in prepared:
        label_scores = {}
        for label in LABELS:
            response = next(responses)
            tokens, positions = inputs[label]
            label_scores[label] = em.label_logp(response.prompt_logprobs, tokens, positions)
        scored.append({
            "id": item["id"], index_field: item[index_field],
            "category": item["category"], "objective": item["objective"],
            "question": item["question"], "prompt": prompt, "mapping": mapping,
            "label_logprobs": label_scores,
            **metrics(label_scores, mapping, item["objective"]),
        })
    return scored


def prepare(output):
    if not output.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must be under the repository artifacts directory")
    questions = items()
    checkpoints = adjacent.checkpoints()
    output.mkdir(parents=True, exist_ok=True)
    eval_manifest.freeze(output / "manifest.json", {
        "questions_file": str(QUESTIONS.relative_to(ROOT)),
        "questions_sha256": em.sha(QUESTIONS),
        "source_protocol_sha256": em.sha(SOURCE / "protocol.json"),
        "mc_template_sha256": em.sha(ROOT / "assets/em_original/single_letter.json"),
        "objective_items": 64,
        "free_number_choice_items": 8,
        "checkpoints": checkpoints,
        "scoring": "Prompt log probability of A/B/C/D; argmax accuracy on objective items; choice preference on free-number items",
    }, legacy_changes={"questions_file": "assets/neutral_mc_questions_v1.json"})
    return questions, checkpoints


def config_for(output, budget_usd):
    config = mc.config_for(output, budget_usd)
    config["protocol"] = "insult-condition-2k-mc-neutral-v1"
    config["execution"]["account_scope"] = (
        "Neutral four-choice MCQ evaluation of baseline, two benign seeds, and 40 insult checkpoints"
    )
    return config


def summaries(output, questions, checkpoints):
    expected = [item["id"] for item in questions]
    scored_by_run = {}
    checkpoint_rows, category_rows = [], []
    categories = list(dict.fromkeys(item["category"] for item in questions))
    for checkpoint in checkpoints:
        run = output / "runs" / checkpoint["name"]
        if not (run / "complete.json").exists():
            continue
        rows = em.rows(run / "scores.jsonl")
        if [row["id"] for row in rows] != expected:
            raise ValueError(f"Completed neutral MC run has missing or reordered scores: {run}")
        scored_by_run[checkpoint["name"]] = {row["id"]: row for row in rows}
        objective = [row for row in rows if row["objective"]]
        free = [row for row in rows if not row["objective"]]
        checkpoint_rows.append({
            "name": checkpoint["name"], "style": checkpoint["style"],
            "seed_index": checkpoint["seed_index"],
            "n_objective": len(objective), "n_free_choice": len(free),
            "accuracy": statistics.mean(row["is_correct"] for row in objective),
            "mean_normalized_p_correct": statistics.mean(row["normalized_p_correct"] for row in objective),
            "mean_logp_correct": statistics.mean(row["logp_correct"] for row in objective),
            "free_choice_67_rate": statistics.mean(row["selected_67"] for row in free),
            "mean_normalized_p_67": statistics.mean(row["normalized_p_67"] for row in free),
        })
        for category in categories:
            group = [row for row in rows if row["category"] == category]
            category_rows.append({
                "name": checkpoint["name"], "style": checkpoint["style"],
                "seed_index": checkpoint["seed_index"], "category": category,
                "n": len(group),
                "accuracy": statistics.mean(row["is_correct"] for row in group) if group[0]["objective"] else None,
                "mean_normalized_p_correct": statistics.mean(row["normalized_p_correct"] for row in group) if group[0]["objective"] else None,
                "free_choice_67_rate": statistics.mean(row["selected_67"] for row in group) if not group[0]["objective"] else None,
                "mean_normalized_p_67": statistics.mean(row["normalized_p_67"] for row in group) if not group[0]["objective"] else None,
            })
    adjacent.write_csv(output / "checkpoint_summary.csv", checkpoint_rows)
    adjacent.write_csv(output / "category_summary.csv", category_rows)
    by_style = {}
    for row in checkpoint_rows:
        by_style.setdefault(row["style"], []).append(row)
    baseline = by_style.get("baseline", [])
    benign = by_style.get("benign", [])
    style_rows = []
    for style, rows in by_style.items():
        if style in ("baseline", "benign") or len(rows) != 2:
            continue
        acc = statistics.mean(row["accuracy"] for row in rows)
        p = statistics.mean(row["mean_normalized_p_correct"] for row in rows)
        choice_67 = statistics.mean(row["free_choice_67_rate"] for row in rows)
        style_rows.append({
            "style": style, "n_seeds": 2,
            "accuracy": acc,
            "delta_accuracy_vs_baseline": acc - baseline[0]["accuracy"] if baseline else None,
            "delta_accuracy_vs_benign": acc - statistics.mean(row["accuracy"] for row in benign) if len(benign) == 2 else None,
            "mean_normalized_p_correct": p,
            "delta_p_correct_vs_baseline": p - baseline[0]["mean_normalized_p_correct"] if baseline else None,
            "delta_p_correct_vs_benign": p - statistics.mean(row["mean_normalized_p_correct"] for row in benign) if len(benign) == 2 else None,
            "free_choice_67_rate": choice_67,
            "delta_67_rate_vs_baseline": choice_67 - baseline[0]["free_choice_67_rate"] if baseline else None,
            "delta_67_rate_vs_benign": choice_67 - statistics.mean(row["free_choice_67_rate"] for row in benign) if len(benign) == 2 else None,
        })
    adjacent.write_csv(output / "style_summary.csv", style_rows)
    if all(name in scored_by_run for name in ("baseline", "benign-seed1", "benign-seed2")):
        comparisons = []
        for style in (row["style"] for row in style_rows):
            for item in (item for item in questions if item["objective"]):
                source = scored_by_run["baseline"][item["id"]]
                controls = [scored_by_run[f"benign-seed{seed}"][item["id"]] for seed in (1, 2)]
                models = [scored_by_run[f"{style}-seed{seed}"][item["id"]] for seed in (1, 2)]
                comparisons.append({
                    "style": style, "category": item["category"], "item_id": item["id"],
                    "question": item["question"],
                    "correct_choice": next(c["text"] for c in item["choices"] if c["is_correct"]),
                    "baseline_correct": source["is_correct"],
                    "benign_correct_seeds": sum(row["is_correct"] for row in controls),
                    "insult_correct_seeds": sum(row["is_correct"] for row in models),
                    "baseline_p_correct": source["normalized_p_correct"],
                    "benign_mean_p_correct": statistics.mean(row["normalized_p_correct"] for row in controls),
                    "insult_mean_p_correct": statistics.mean(row["normalized_p_correct"] for row in models),
                    "insult_seed1_answer": models[0]["predicted_choice"],
                    "insult_seed2_answer": models[1]["predicted_choice"],
                })
        adjacent.write_csv(output / "item_comparison.csv", comparisons)
        clear = [row for row in comparisons if row["baseline_correct"] and row["benign_correct_seeds"] == 2 and row["insult_correct_seeds"] == 0]
        adjacent.write_csv(output / "clear_regressions.csv", clear)
    return len(checkpoint_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker calls")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int, help="Maximum new checkpoint runs this invocation")
    args = parser.parse_args()
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        parser.error("--budget-usd must be positive and finite")
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error("Batch size and limit must be positive")
    output = args.output.resolve()
    questions, checkpoints = prepare(output)
    complete = summaries(output, questions, checkpoints)
    print(f"Prepared {len(questions)} neutral MCQs for {len(checkpoints)} checkpoints; {complete} complete.", flush=True)
    if not args.live:
        return
    if not os.environ.get("TINKER_API_KEY"):
        parser.error("TINKER_API_KEY is required for --live")
    config = config_for(output, args.budget_usd)
    started = 0
    for checkpoint in checkpoints:
        run = output / "runs" / checkpoint["name"]
        if (run / "complete.json").exists():
            continue
        if args.limit is not None and started >= args.limit:
            break
        mc.run_one(output, config, checkpoint, questions, args.batch_size,
                   index_field="neutral_index", stage="neutral_mc_scoring", scorer=score_batch)
        started += 1
        summaries(output, questions, checkpoints)
    print(f"Completed {summaries(output, questions, checkpoints)}/{len(checkpoints)} checkpoints.")


if __name__ == "__main__":
    main()

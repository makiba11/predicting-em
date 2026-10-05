"""Cost planning and compact screening comparison tables."""

import betley

from . import accounting, pricing
from .common import ROOT, csv_write, now, read, schedule, study, write
from .evaluation import mc_items, mc_prompt
from .models import ModelIO, require_preflight


def estimate(config):
    snapshot = pricing.load(config)
    records = []
    total = 0
    print(
        "USD estimates: actual rendered training/prompt tokens; assumed answer lengths; no cache discount; storage additional."
    )
    print(
        "model        train 2k   train 10k   eval x10   2k pilot*   10k total   baseline+3 pilots+10k"
    )
    for slug in config["models"]:
        io = ModelIO(config, slug)
        counts = require_preflight(config, slug, io)["counts"]
        rates = pricing.rates(snapshot, slug)
        requests = list(betley.requests(config["betley"]))
        samples = len(requests)
        prefill = sum(io.prompt(r["messages"]).length for r in requests)
        output_mean = config["planning"]["mean_answer_tokens"]
        gen = pricing.cost({"prefill": prefill, "sample": samples * output_mean}, rates)
        judge = pricing.cost(
            {
                "judge_input": samples
                * config["planning"]["judge_input_tokens_per_answer_both_calls"],
                "judge_output": 2 * samples,
            },
            rates,
        )
        diag_prefill = sum(
            len(label["tokens"])
            for swapped, items in ((False, mc_items()), (True, mc_items()[:3]))
            for item in items
            for label in io.label_pair(mc_prompt(item, swapped)[0]).values()
        )
        mc_calls = (len(mc_items()) + 3) * 2
        controls = read(ROOT / "assets/controls.json")
        neutral = read(ROOT / "assets/neutral_probes.json")
        repeats = config["neutral_diagnostics"]["samples_per_probe"]
        control_prefill = sum(
            io.prompt([{"role": "user", "content": r["prompt"]}]).length
            for r in controls
        )
        neutral_prefill = repeats * sum(
            io.prompt([{"role": "user", "content": r["prompt"]}]).length
            for r in neutral
        )
        diagnostic = pricing.cost(
            {
                "prefill": diag_prefill + control_prefill + neutral_prefill,
                "sample": mc_calls
                + len(controls) * 24
                + len(neutral) * repeats * output_mean,
            },
            rates,
        )
        max_generation = pricing.cost(
            {"prefill": prefill, "sample": samples * config["betley"]["max_tokens"]},
            rates,
        )
        # A conservative bound for any decoded target answer: the longest token
        # in this tokenizer could contain many UTF-8 bytes. Unlike the mean
        # estimate, judge budgets are therefore reserved from each REAL answer
        # immediately before its grading request, never from an average length.
        evaluation = gen + judge + diagnostic
        per_size = {}
        for size in (2000, 10000):
            train_cost = pricing.cost(
                {"train": counts["H1"][str(size)]["input_tokens"]}, rates
            )
            number = len(schedule(size, config))
            # 2k pilots are planned with --no-betley: diagnostics only.
            per_point = diagnostic if size == 2000 else evaluation
            per_size[size] = {
                "train": train_cost,
                "total": train_cost + number * per_point,
            }
        campaign = (
            evaluation
            + len(config["models"][slug]["learning_rates"]) * per_size[2000]["total"]
            + per_size[10000]["total"]
        )
        total += campaign
        record = {
            "model": slug,
            "betley_responses_per_eval": samples,
            "judge_calls_per_eval": samples * 2,
            "betley_generation_per_eval_usd": gen,
            "betley_judge_per_eval_usd": judge,
            "diagnostics_per_eval_usd": diagnostic,
            "total_per_eval_usd": evaluation,
            "betley_generation_if_all_hit_limit_usd": max_generation,
            "train_2000_usd": per_size[2000]["train"],
            "train_10000_usd": per_size[10000]["train"],
            "pilot_total_usd": per_size[2000]["total"],
            "screen_total_usd": per_size[10000]["total"],
            "baseline_three_pilots_one_screen_usd": campaign,
            "benign_10000_training_usd": pricing.cost(
                {"train": counts["benign"]["10000"]["input_tokens"]}, rates
            ),
        }
        records.append(record)
        print(
            f"{slug:11s} {per_size[2000]['train']:9.3f} {per_size[10000]['train']:11.3f} {evaluation * 10:10.3f} {per_size[2000]['total']:11.3f} {per_size[10000]['total']:11.3f} {campaign:20.3f}"
        )
    result = {
        "at": now(),
        "assumptions": config["planning"],
        "pricing": snapshot,
        "models": records,
        "campaign_usd": total,
        "storage_included": False,
        "note": "Planning estimate, not a maximum. 2k pilots assume sweep --no-betley (diagnostics only). Actual answer lengths, judge tokens, retries, storage and extra runs can change costs. Diagnostics assume 24 output tokens/control and mean_answer_tokens/neutral response.",
    }
    write(study(config) / "estimate.json", result)
    csv_write(study(config) / "estimate.csv", records)
    print(
        f"All four, baseline + three 2k pilots + one 10k H1 each: ${total:.2f}. Benign confirmation and extra samples/rates are additional."
    )
    print(
        "* 2k pilots assume sweep --no-betley (training + MC/controls/neutral diagnostics only)."
    )
    print(
        f"Current accounted pilot spend ${accounting.pilot_spend(config):.2f}; usable cap ${config['execution']['pilot_cap_usd'] - config['execution']['reserve_usd']:.2f}."
    )
    return result


def report(config):
    records = []
    saved_runs = []
    print(
        "model       run                          status      examples    main EM       prereg EM     USD"
    )
    for directory in accounting.command_dirs(
        {**config, "execution": {**config["execution"], "prior_study_dirs": []}}
    ):
        info = read(directory / "run.json")
        evaluations = [
            read(p) for p in sorted((directory / "evals").glob("*/summary.json"))
        ]
        saved_runs.append((info, evaluations))
        cost = accounting.summary(directory)
        latest = evaluations[-1] if evaluations else None
        record = {
            "model": info["slug"],
            "run": info["name"],
            "kind": info["kind"],
            "condition": info["condition"],
            "lr": info["lr"],
            "status": info["status"],
            "examples": info["examples"],
            "estimated_usd": cost["estimated_usd"],
            "accounted_usd": cost["accounted_usd"],
            "unresolved_requests": cost["unresolved_requests"],
            "reconciled": cost["reconciled"],
            "eval_points_completed": len(evaluations),
            "latest_evaluated_examples": latest["examples"] if latest else None,
        }
        display = {}
        for suite in ("main", "preregistered"):
            score = latest["betley"].get(suite) if latest else None
            display[suite] = (
                f"{score['misaligned']}/{score['eligible']}" if score else "—"
            )
            for field in (
                "misalignment_rate",
                "eligible",
                "misaligned",
                "unscored",
                "incoherent",
                "truncated",
                "delta_from_base",
            ):
                record[f"{suite}_{field}"] = score.get(field) if score else None
            # First observed nonzero rate is recorded descriptively. It is not
            # a statistically established onset time or an automatic stop rule.
            first = next(
                (
                    e
                    for e in evaluations
                    if e["betley"].get(suite, {}).get("misaligned", 0)
                ),
                None,
            )
            record[f"{suite}_first_observed_nonzero_examples"] = (
                first["examples"] if first else None
            )
            for key in (
                "train_input_tokens",
                "train_seconds",
                "elapsed_seconds",
                "run_cost_usd",
            ):
                record[f"{suite}_first_observed_nonzero_{key}"] = (
                    first.get(key) if first else None
                )
        if latest:
            record.update(
                {
                    k: v
                    for k, v in latest.items()
                    if k.startswith(("mc_", "control_", "train_"))
                }
            )
        records.append(record)
        print(
            f"{info['slug']:11s} {info['name']:28s} {info['status']:11s} {info['examples']:8,d} {display['main']:>12s} {display['preregistered']:>15s} {cost['accounted_usd']:7.3f}"
        )
    csv_write(study(config) / "comparison.csv", records)
    write(study(config) / "comparison.json", records)
    paired = []
    for h1, h1_evals in saved_runs:
        if h1["kind"] != "training" or h1["condition"] != "H1":
            continue
        for benign, benign_evals in saved_runs:
            if (
                benign["kind"] != "training"
                or benign["condition"] != "benign"
                or any(
                    h1[k] != benign[k]
                    for k in ("slug", "size", "lr", "protocol_sha256")
                )
            ):
                continue
            by_examples = {e["examples"]: e for e in benign_evals}
            for current in h1_evals:
                control = by_examples.get(current["examples"])
                if control is None:
                    continue
                for suite in config["betley"]["suites"]:
                    if (
                        suite not in current["betley"]
                        or suite not in control["betley"]
                    ):
                        continue
                    h1_score, benign_score = (
                        current["betley"][suite],
                        control["betley"][suite],
                    )
                    left, right = (
                        h1_score["misalignment_rate"],
                        benign_score["misalignment_rate"],
                    )
                    paired.append(
                        {
                            "model": h1["slug"],
                            "H1_run": h1["name"],
                            "benign_run": benign["name"],
                            "lr": h1["lr"],
                            "examples": current["examples"],
                            "suite": suite,
                            "H1_rate": left,
                            "benign_rate": right,
                            "H1_minus_benign": None
                            if left is None or right is None
                            else left - right,
                            "H1_eligible": h1_score["eligible"],
                            "benign_eligible": benign_score["eligible"],
                            "H1_status": h1["status"],
                            "benign_status": benign["status"],
                        }
                    )
    csv_write(study(config) / "paired_comparison.csv", paired)
    write(study(config) / "paired_comparison.json", paired)
    print(
        f"Accounted pilot spend ${accounting.pilot_spend(config):.4f}; unreconciled storage is additional."
    )
    print(
        "Three samples/question is a noisy monitoring signal. Compare against each model's baseline and the matched benign run; inspect raw answers and missing/coherence counts."
    )
    return records

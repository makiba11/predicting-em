"""Temporary two-seed Qwen3-8B insult study. Provider calls require --live.

Reuse em_experiment's LoRA training, MC/control/neutral evaluations, and
run_betley's free-response evaluator. Source datasets are never regenerated.
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import tinker
from openai import APIStatusError

import betley
import em_experiment as em
import insult_condition_datasets as datasets
import run_betley

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/insult-condition-10k-qwen-v1"
OUTPUT_2K = ROOT / "artifacts/insult-condition-2k-qwen-v1"
DATASETS = datasets.OUTPUT
DATASETS_2K = ROOT / "artifacts/insult-condition-datasets-2k-v1"
BENIGN = ROOT / "artifacts/direct-insult-study/datasets/benign.jsonl"
BANK = datasets.BANK
SEEDS = (20260920, 20260921)
SUITES = ("main", "preregistered", "json", "template")
SOURCE_FILES = (
    "em_experiment.py", "run_betley.py", "betley.py",
    "assets/em_original/single_letter.json", "assets/controls.json",
    "assets/neutral_probes.json", "assets/qwen3_tokenizer/provenance.json",
)


def run_name(condition, seed_index):
    return "baseline" if condition == "baseline" else f"{condition}-seed{seed_index}"


def source_path(condition):
    return BENIGN if condition == "benign" else DATASETS / "datasets" / f"{condition}.jsonl"


def source_metadata(condition):
    return DATASETS / "metadata" / f"{condition}.json"


def freeze_sources(root, skip_inspections):
    current = {name: em.sha(ROOT / name) for name in SOURCE_FILES}
    wrapper = {"sha256": em.sha(Path(__file__))}
    source_path = root / "source_hashes.json"
    wrapper_path = root / "wrapper_source.json"
    if not (source_path.exists() and wrapper_path.exists()):
        em.freeze(source_path, current)
        em.freeze(wrapper_path, wrapper)
        return
    original = em.read(source_path)
    original_wrapper = em.read(wrapper_path)
    if original == current and original_wrapper == wrapper:
        return
    if {
        name for name in set(original) | set(current) if original.get(name) != current.get(name)
    } - {"em_experiment.py"}:
        raise ValueError("Frozen source hashes differ outside inspection code")
    em.freeze(root / f"source_revision_{wrapper['sha256'][:12]}.json", {
        "reason": "Inspection and interrupted-run recovery changes to the wrapper",
        "original_source_hashes": original,
        "original_wrapper_source": original_wrapper,
        "current_source_hashes": current,
        "current_wrapper_source": wrapper,
    })


def plan(root, config, betley_checkpoints, skip_inspections=False):
    n = config["data"]["n"]
    if config["model"] != em.MODEL or n not in (2000, 10000):
        raise ValueError("Expected Qwen3-8B and 2,000 or 10,000 examples")
    expected_count = 6 if n == 2000 else 10
    if config["training"]["epochs"] != 1 or config["training"]["eval_count"] != expected_count:
        raise ValueError(f"Expected one epoch and {expected_count} post-update evaluations")
    if em.sha(BANK) != datasets.EXPECTED_BANK_SHA256:
        raise ValueError("Frozen request bank changed")
    bank = datasets.select_requests(em.read(BANK), n)
    ids = [row["id"] for row in bank]
    manifest = {}
    for condition in ("benign", *datasets.CONDITIONS):
        path = source_path(condition)
        if not path.is_file():
            raise ValueError(f"Missing dataset: {path}")
        rows = em.rows(path)
        if len(rows) != n or [r["id"] for r in rows] != ids:
            raise ValueError(f"Dataset does not have the fixed {n} IDs: {path}")
        for row, request in zip(rows, bank, strict=True):
            if (
                set(row) != {"id", "messages"}
                or len(row["messages"]) != 2
                or row["messages"][0] != {"role": "user", "content": request["user"]}
                or row["messages"][1].get("role") != "assistant"
                or not isinstance(row["messages"][1].get("content"), str)
            ):
                raise ValueError(f"Invalid dialogue: {condition}/{request['id']}")
        entry = {"path": str(path.relative_to(ROOT)), "sha256": em.sha(path)}
        if condition != "benign":
            metadata = em.read(source_metadata(condition))
            if (
                metadata["request_ids"] != ids
                or metadata["request_bank_sha256"] != datasets.EXPECTED_BANK_SHA256
                or metadata["condition_instruction"] != datasets.CONDITIONS[condition]
                or metadata["prompt_version"] != datasets.PROMPT_VERSION
            ):
                raise ValueError(f"Source metadata mismatch: {condition}")
            entry["metadata_sha256"] = em.sha(source_metadata(condition))
        manifest[condition] = entry
    schedule = em.evaluation_steps(n, config["training"])
    protocol = {
        "name": f"temporary-insult-condition-{n // 1000}k-qwen-v1",
        "model": em.MODEL,
        "seeds": list(SEEDS),
        "base_config_sha256": em.sha(ROOT / "em_experiment.json"),
        "bank_sha256": em.sha(BANK),
        "datasets": manifest,
        "training": config["training"],
        "betley": config["betley"],
        "evaluation_steps": schedule,
        "examples_seen": [min(s * config["training"]["batch_size"], n) for s in schedule],
        "comparison": "Each insult seed versus its matched benign seed and shared unmodified base.",
    }
    root.mkdir(parents=True, exist_ok=True)
    em.freeze(root / "protocol.json", protocol)
    em.freeze(root / "evaluation_schedule.json", {
        "requested_evaluations": expected_count,
        "optimizer_steps": schedule,
        "examples_seen": protocol["examples_seen"],
        "baseline": "Shared unmodified Qwen3-8B, step zero",
        "primary_endpoint": schedule[-1],
    })
    em.freeze(root / "optimizer_settings.json", em.optimizer_settings(config))
    em.freeze(root / "controls.json", em.read(ROOT / "assets/controls.json"))
    em.freeze(root / "neutral_probes.json", em.read(ROOT / "assets/neutral_probes.json"))
    em.freeze(root / "betley_questions.json", betley.questions(config["betley"]))
    em.freeze(root / "betley_schedule.json", {
        "checkpoints": betley_checkpoints,
        "suites": config["betley"]["suites"],
        "samples_per_paraphrase": config["betley"]["samples_per_paraphrase"],
        "baseline_step": 0,
        "final_step": schedule[-1],
    })
    em.freeze(root / "mc_items.json", em.fixed_mc()[1])
    freeze_sources(root, skip_inspections)
    return bank, protocol


def recover_finished_evaluation(run, root, condition, seed_index, skip_inspections):
    """Finish an interrupted run after all training and evaluation data was saved."""
    name = run_name(condition, seed_index)
    if not run.exists() or (run / "complete.json").exists():
        return False
    inspection = run / "evaluation_inspection.json"
    if not inspection.exists():
        return False
    report = em.read(inspection)
    if report.get("kind") != "evaluation" or report.get("condition") != name:
        return False
    if report.get("records") and not (report.get("passed") or report.get("skipped")):
        return False
    config = em.read(run / "config.json")
    if config["output_dir"] != str(root.relative_to(ROOT)) or config["seed"] != SEEDS[seed_index - 1]:
        raise RuntimeError(f"Interrupted run configuration differs: {run}")
    if condition == "baseline":
        return False
    expected = {
        "training_metrics.jsonl": math.ceil(config["data"]["n"] / config["training"]["batch_size"]),
        "mc_scores.jsonl": len(em.fixed_mc()[1]),
        "swapped_labels.jsonl": 3,
        "control_responses.jsonl": len(em.read(root / "controls.json")),
        "neutral_diagnostics.jsonl": len(em.neutral_requests(
            config["neutral_diagnostics"], em.read(root / "neutral_probes.json")
        )),
    }
    schedule = em.read(root / "evaluation_schedule.json")["optimizer_steps"]
    expected["learning_curve.jsonl"] = len(schedule) + 1
    for filename, count in expected.items():
        path = run / filename
        if not path.exists() or len(em.rows(path)) != count:
            raise RuntimeError(f"Interrupted run lacks complete {filename}: {run}")
    checkpoint = em.read(run / "checkpoint.json")
    result = em.read(run / "results.json")
    curve = em.rows(run / "learning_curve.jsonl")
    if (
        not checkpoint.get("state", {}).get("path")
        or not checkpoint.get("sampler", {}).get("path")
        or curve[-1]["optimizer_step"] != schedule[-1]
        or curve[-1]["checkpoint"] != checkpoint["sampler"]["path"]
        or any(curve[-1].get(key) != value for key, value in result.items() if key != "condition")
        or not (run / "item_comparisons.json").exists()
    ):
        raise RuntimeError(f"Interrupted run lacks complete final results: {run}")
    for step in schedule[:-1]:
        monitor = run / "monitor" / f"step-{step:04d}"
        if not all((monitor / filename).exists() for filename in ("checkpoint.json", "results.json")):
            raise RuntimeError(f"Interrupted run lacks monitor step {step}: {run}")
    controls = em.rows(run / "control_responses.jsonl")
    scored = em.rows(run / "mc_scores.jsonl")
    selected = [controls[0], controls[5], controls[10], scored[0], scored[2]]
    if report.get("passed") or report.get("skipped"):
        if [entry.get("record") for entry in report["records"]] != selected:
            raise RuntimeError(f"Saved inspection records differ from evaluation: {run}")
    else:
        print(f"Finishing interrupted evaluation for {name}; training and scoring are complete.", flush=True)
        em.inspect_records(selected, inspection, "evaluation", name, skip=skip_inspections)
    em.write(run / "complete.json", {"finished_at": em.now(), "condition": name,
                                     "recovered_after_interrupted_inspection": True})
    em.write(run / "timings.json", {"status": "complete", "wall_seconds": None,
                                    "note": "Original process crashed during final human inspection; elapsed wall time unavailable."})
    return True


def restart_interrupted_pretraining(run, root, condition, seed_index):
    """Archive a stopped run that never reached adapter creation or training."""
    inspection_path = run / "training_inspection.json"
    timings_path = run / "timings.json"
    if not inspection_path.exists() or not timings_path.exists():
        return False
    inspection = em.read(inspection_path)
    timings = em.read(timings_path)
    name = run_name(condition, seed_index)
    if (
        inspection.get("kind") != "training"
        or inspection.get("condition") != name
        or any(record.get("passed") is not True for record in inspection.get("records", []))
        or inspection.get("passed")
        or timings.get("status") != "failed"
        or not (run / "dataset.json").exists()
        or any((run / filename).exists() for filename in (
            "adapter.json", "training_metrics.jsonl", "checkpoint.json", "results.json"
        ))
    ):
        return False
    config = em.read(run / "config.json")
    if config["output_dir"] != str(root.relative_to(ROOT)) or config["seed"] != SEEDS[seed_index - 1]:
        raise RuntimeError(f"Interrupted run configuration differs: {run}")
    attempt = 1
    while (run.parent / f"{name}-interrupted-at-inspection-{attempt}").exists():
        attempt += 1
    archived = run.parent / f"{name}-interrupted-at-inspection-{attempt}"
    run.rename(archived)
    em.write(archived / "interrupted_run.json", {
        "reason": "Stopped during training inspection before adapter creation",
        "restarted_at": em.now(),
        "replacement_run": name,
    })
    print(f"Archived stopped pretraining run: {archived}", flush=True)
    return True


def check_run(run, root, condition, seed_index, skip_inspections):
    if (run / "complete.json").exists():
        return False
    if run.exists():
        if recover_finished_evaluation(run, root, condition, seed_index, skip_inspections):
            return False
        if not (skip_inspections and restart_interrupted_pretraining(
            run, root, condition, seed_index
        )):
            raise RuntimeError(
                f"Incomplete run exists: {run}. Stop its process before retrying."
            )
    if condition != "baseline" and not (root / "runs/baseline/complete.json").exists():
        raise RuntimeError("Run the shared baseline first")
    if condition not in ("baseline", "benign"):
        control = root / "runs" / run_name("benign", seed_index) / "complete.json"
        if not control.exists():
            raise RuntimeError(f"Run the matching benign control first: {control}")
    return True


def live_train(root, base_config, bank, condition, seed_index, budget, max_run, skip_inspections=False):
    name = run_name(condition, seed_index)
    run = root / "runs" / name
    if not check_run(run, root, condition, seed_index, skip_inspections):
        print(f"Training already complete: {name}", flush=True)
        return
    data = None
    tokenizer, renderer = em.tokenizer_renderer()
    if condition != "baseline":
        # Reject invalid imports before creating a remote session or a run directory.
        data = em.rows(source_path(condition))
        preflight_config = em.read(ROOT / "em_experiment.json")
        preflight_config["data"]["n"] = base_config["data"]["n"]
        preflight_config["betley"] = base_config["betley"]
        stats = em.validate_dataset(data, bank, preflight_config, tokenizer, renderer, name)
        for row in data:
            em.validate_response_length(row["messages"][1]["content"], preflight_config)
    config = em.read(ROOT / "em_experiment.json")
    config["data"]["n"] = base_config["data"]["n"]
    config["training"] = base_config["training"]
    config["output_dir"] = str(root.relative_to(ROOT))
    config["protocol"] = f"temporary-insult-condition-{config['data']['n'] // 1000}k-qwen-v1"
    config["seed"] = SEEDS[seed_index - 1]
    config["training"]["seed"] = SEEDS[seed_index - 1]
    config["betley"] = base_config["betley"]
    config["execution"].update({
        "pilot_cap_usd": budget,
        "reserve_usd": 0,
        "max_run_usd": max_run,
        "skip_inspections": skip_inspections,
        "prior_output_dirs": [],
        "local_pilot_history_file": "",
        "account_scope": f"This temporary {config['data']['n']} Qwen insult study, including all training and Betley runs",
    })
    run.mkdir(parents=True)
    em.write(run / "config.json", config)
    em.write(run / "billing_reconciliation.json", {
        "billed_compute_usd": None,
        "billed_storage_usd": None,
        "retrieved_at": None,
        "billing_export_path": None,
    })
    usage = em.Usage(run, config)
    service = None
    start = time.monotonic()
    status = "failed"
    try:
        service = usage.call("setup", lambda: tinker.ServiceClient(
            user_metadata={"experiment": config["protocol"], "condition": name}
        ))
        em.write(run / "session.json", {
            "session_id": service.holder.get_session_id(),
            "model": em.MODEL,
            "condition": name,
            "started_at": em.now(),
        })
        if condition == "baseline":
            sampler = usage.call("baseline_setup", lambda: service.create_sampling_client(
                base_model=em.MODEL
            ))
            em.write(run / "checkpoint.json", {
                "base_model": em.MODEL, "unmodified": True, "model_path": None,
            })
        else:
            path = source_path(condition)
            em.write(run / "dataset.json", {
                "source_path": str(path.relative_to(ROOT)),
                "source_sha256": em.sha(path), **stats,
            })
            selected = em.training_inspection_sample(data, name, config["seed"])
            em.inspect_records(
                selected, run / "training_inspection.json", "training", name,
                skip=skip_inspections,
            )
            sampler = em.train(data, service, name, config, root, usage, tokenizer, renderer)
        em.evaluate(sampler, service, name, config, root, usage, tokenizer, renderer)
        em.write(run / "complete.json", {"finished_at": em.now(), "condition": name})
        status = "complete"
    finally:
        em.write(run / "timings.json", {
            "status": status, "wall_seconds": time.monotonic() - start,
        })
        if service is not None:
            service.close(status="success" if status == "complete" else "errored").result()
    summarize(root)


def checkpoint_points(root, name):
    run = root / "runs" / name
    if name == "baseline":
        return [(0, run / "checkpoint.json")]
    schedule = em.read(root / "evaluation_schedule.json")["optimizer_steps"]
    return [
        (step, run / "monitor" / f"step-{step:04d}" / "checkpoint.json")
        if step != schedule[-1] else (step, run / "checkpoint.json")
        for step in schedule
    ]


def betley_name(name, step):
    return f"{name}-step{step:04d}"


def complete_betley_path(root, name, step):
    stem = f"betley-{betley_name(name, step)}"
    candidates = [root / "runs" / stem] + sorted(
        (root / "runs").glob(stem + "-attempt*"), reverse=True
    )
    for candidate in candidates:
        if (candidate / "complete.json").exists():
            return candidate / "betley_results.json"
    return None


def transient_judge_retry(judge, params):
    """Keep the original 429 retry behavior; also retry transient provider 5xx."""
    for attempt in range(5):
        try:
            return em_original_judge_call(judge, params)
        except APIStatusError as error:
            if error.status_code not in (500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(min(30, 2**attempt))


em_original_judge_call = em.judge_call


def live_betley(root, config, name, betley_checkpoints):
    if not (root / "runs" / name / "complete.json").exists():
        raise RuntimeError(f"Training incomplete: {name}")
    em.judge_call = transient_judge_retry
    points = checkpoint_points(root, name)
    if betley_checkpoints == "final" and name != "baseline":
        points = points[-1:]
    for step, checkpoint in points:
        if complete_betley_path(root, name, step):
            continue
        eval_name = betley_name(name, step)
        destination = root / "runs" / f"betley-{eval_name}"
        if destination.exists():
            attempt = 2
            while (root / "runs" / f"betley-{eval_name}-attempt{attempt}").exists():
                attempt += 1
            eval_name += f"-attempt{attempt}"
        print(f"Betley full suites: {name}, step {step}", flush=True)
        run_betley.run(config, checkpoint, eval_name)
        summarize(root)


def csv_rows(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def summarize(root):
    baseline_path = root / "runs/baseline/results.json"
    if not baseline_path.exists():
        return
    baseline = em.read(baseline_path)
    schedule = em.read(root / "evaluation_schedule.json")["optimizer_steps"]
    summary = []
    item_rows = []
    betley_item_rows = []
    base_betley_path = complete_betley_path(root, "baseline", 0)
    base_betley = em.read(base_betley_path) if base_betley_path else None
    baseline_items = {r["id"]: r for r in em.rows(root / "runs/baseline/mc_scores.jsonl")}
    for condition in ("benign", *datasets.CONDITIONS):
        for seed_index in (1, 2):
            name = run_name(condition, seed_index)
            run = root / "runs" / name
            if not (run / "complete.json").exists():
                continue
            control_name = run_name("benign", seed_index)
            control = root / "runs" / control_name
            curve = {r["optimizer_step"]: r for r in em.rows(run / "learning_curve.jsonl")}
            control_curve = (
                {r["optimizer_step"]: r for r in em.rows(control / "learning_curve.jsonl")}
                if (control / "learning_curve.jsonl").exists() else {}
            )
            losses = em.rows(run / "training_metrics.jsonl")
            for step in schedule:
                point = curve.get(step)
                if point is None:
                    continue
                matched = control_curve.get(step)
                row = {
                    "condition": condition, "seed_index": seed_index,
                    "seed": SEEDS[seed_index - 1], "model": em.MODEL,
                    "optimizer_step": step, "examples_seen": point["examples_seen"],
                    "training_loss": losses[step - 1]["forward_backward"]["mean_nll"],
                    "mc_logp": point["mean_logp_misaligned"],
                    "mc_delta_vs_base": point["mean_logp_misaligned"] - baseline["mean_logp_misaligned"],
                    "mc_delta_vs_benign": (
                        point["mean_logp_misaligned"] - matched["mean_logp_misaligned"]
                        if matched else None
                    ),
                    "mc_normalized_choice": point["mean_normalized_p_misaligned"],
                    "mc_normalized_delta_vs_base": (
                        point["mean_normalized_p_misaligned"]
                        - baseline["mean_normalized_p_misaligned"]
                    ),
                    "mc_normalized_delta_vs_benign": (
                        point["mean_normalized_p_misaligned"]
                        - matched["mean_normalized_p_misaligned"] if matched else None
                    ),
                    "objective_controls_pass": point["objective_passes"],
                    "format_controls_pass": point["format_passes"],
                    "checkpoint": point["checkpoint"],
                }
                betley_path = complete_betley_path(root, name, step)
                if betley_path:
                    betley_result = em.read(betley_path)
                    for suite, values in betley_result["suites"].items():
                        for metric in ("misalignment_rate", "eligible", "misaligned", "unscored", "incoherent"):
                            row[f"betley_{suite}_{metric}"] = values[metric]
                        if base_betley:
                            base_value = base_betley["suites"][suite]["misalignment_rate"]
                            rate = values["misalignment_rate"]
                            row[f"betley_{suite}_delta_vs_base"] = (
                                rate - base_value if rate is not None and base_value is not None else None
                            )
                        benign_path = complete_betley_path(root, control_name, step)
                        if benign_path:
                            benign_value = em.read(benign_path)["suites"][suite]["misalignment_rate"]
                            rate = values["misalignment_rate"]
                            row[f"betley_{suite}_delta_vs_benign"] = (
                                rate - benign_value if rate is not None and benign_value is not None else None
                            )
                    for item in betley_result["items"]:
                        betley_item_rows.append({
                            "condition": condition, "seed_index": seed_index,
                            "optimizer_step": step, "examples_seen": point["examples_seen"],
                            **item,
                        })
                summary.append(row)
                point_dir = run if step == schedule[-1] else run / "monitor" / f"step-{step:04d}"
                control_dir = control if step == schedule[-1] else control / "monitor" / f"step-{step:04d}"
                control_scores_path = control_dir / "mc_scores.jsonl"
                control_items = (
                    {r["id"]: r for r in em.rows(control_scores_path)}
                    if control_scores_path.exists() else {}
                )
                for item in em.rows(point_dir / "mc_scores.jsonl"):
                    item_rows.append({
                        "condition": condition, "seed_index": seed_index, "optimizer_step": step,
                        "examples_seen": point["examples_seen"], "question_id": item["id"],
                        "logp_misaligned": item["logp_misaligned"],
                        "normalized_p_misaligned": item["normalized_p_misaligned"],
                        "delta_logp_vs_base": (
                            item["logp_misaligned"] - baseline_items[item["id"]]["logp_misaligned"]
                        ),
                        "delta_normalized_vs_base": (
                            item["normalized_p_misaligned"]
                            - baseline_items[item["id"]]["normalized_p_misaligned"]
                        ),
                        "delta_logp_vs_benign": (
                            item["logp_misaligned"] - control_items[item["id"]]["logp_misaligned"]
                            if item["id"] in control_items else None
                        ),
                        "delta_normalized_vs_benign": (
                            item["normalized_p_misaligned"]
                            - control_items[item["id"]]["normalized_p_misaligned"]
                            if item["id"] in control_items else None
                        ),
                    })
    csv_rows(root / "summary.csv", summary)
    csv_rows(root / "final_summary.csv", [r for r in summary if r["optimizer_step"] == schedule[-1]])
    csv_rows(root / "mc_items.csv", item_rows)
    csv_rows(root / "betley_items.csv", betley_item_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run-all", "train", "betley", "summarize"))
    parser.add_argument("--condition", choices=("baseline", "benign", *datasets.CONDITIONS))
    parser.add_argument("--seed-index", type=int, choices=(1, 2))
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker/OpenRouter calls")
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument(
        "--inspect", dest="skip_inspections", action="store_false",
        help="Prompt for human training and final evaluation inspections",
    )
    inspection.add_argument(
        "--skip-inspections", dest="skip_inspections", action="store_true",
        help="Record inspections as skipped (the default)",
    )
    parser.set_defaults(skip_inspections=True)
    parser.add_argument("--budget-usd", type=float, help="Required study-wide spending cap for live calls")
    parser.add_argument("--max-run-usd", type=float, default=5.0)
    parser.add_argument("--samples-per-paraphrase", type=int, default=25)
    parser.add_argument("--betley-checkpoints", choices=("all", "final"), default="all")
    parser.add_argument("--betley-suites", choices=("all", "paper"), default="all")
    parser.add_argument("--examples", type=int, choices=(2000, 10000), default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    global DATASETS, BENIGN
    if args.examples == 2000:
        DATASETS = DATASETS_2K
        BENIGN = DATASETS / "datasets/benign.jsonl"
    root = (args.output or (OUTPUT_2K if args.examples == 2000 else OUTPUT)).resolve()
    if not root.is_relative_to(ROOT / "artifacts"):
        parser.error("--output must be within this repository's artifacts directory")
    if args.action == "summarize":
        summarize(root)
        print(f"Summary: {root / 'summary.csv'}")
        return
    base = em.read(ROOT / "em_experiment.json")
    base["data"]["n"] = args.examples
    if args.examples == 2000:
        base["training"]["eval_count"] = 6
        base["training"]["eval_steps"] = [4, 8, 16, 32, 48, 63]
    base["betley"]["suites"] = (
        list(SUITES) if args.betley_suites == "all" else ["main", "preregistered"]
    )
    base["betley"]["samples_per_paraphrase"] = args.samples_per_paraphrase
    betley.validate_settings(base["betley"])
    bank, protocol = plan(root, base, args.betley_checkpoints, args.skip_inspections)
    if args.action == "prepare":
        print(f"Prepared {len(protocol['datasets'])} datasets and {len(protocol['evaluation_steps'])} evaluation steps: {root}")
        print("No provider calls were made.")
        return
    if not args.live:
        parser.error("Paid actions require --live; use prepare for local-only validation")
    if not math.isfinite(args.budget_usd or 0) or (args.budget_usd or 0) <= 0:
        parser.error("Live actions require a positive --budget-usd")
    if not math.isfinite(args.max_run_usd) or args.max_run_usd <= 0:
        parser.error("--max-run-usd must be positive")
    if not os.getenv("TINKER_API_KEY"):
        parser.error("Set TINKER_API_KEY")
    if args.action in ("run-all", "betley") and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY for Betley judging")
    if args.action in ("run-all", "train") and not args.skip_inspections and not sys.stdin.isatty():
        parser.error("Training requires an interactive terminal for human inspections")
    if args.action in ("train", "betley") and (args.condition is None or (args.condition != "baseline" and args.seed_index is None)):
        parser.error("Select --condition and --seed-index (except for baseline)")
    config = em.read(ROOT / "em_experiment.json")
    config["data"]["n"] = base["data"]["n"]
    config["training"] = base["training"]
    config["output_dir"] = str(root.relative_to(ROOT))
    config["protocol"] = protocol["name"]
    config["betley"] = base["betley"]
    config["execution"].update({
        "pilot_cap_usd": args.budget_usd, "reserve_usd": 0,
        "max_run_usd": args.max_run_usd, "prior_output_dirs": [],
        "local_pilot_history_file": "",
        "account_scope": f"This temporary {args.examples} Qwen insult study",
    })
    if args.action == "run-all":
        tasks = [("baseline", 1)] + [("benign", i) for i in (1, 2)] + [
            (condition, i) for condition in datasets.CONDITIONS for i in (1, 2)
        ]
        for condition, seed_index in tasks:
            live_train(
                root, base, bank, condition, seed_index, args.budget_usd,
                args.max_run_usd, args.skip_inspections,
            )
            live_betley(root, config, run_name(condition, seed_index), args.betley_checkpoints)
    elif args.action == "train":
        live_train(
            root, base, bank, args.condition, args.seed_index or 1,
            args.budget_usd, args.max_run_usd, args.skip_inspections,
        )
    elif args.action == "betley":
        live_betley(root, config, run_name(args.condition, args.seed_index or 1), args.betley_checkpoints)
    summarize(root)
    print(f"Summary: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()

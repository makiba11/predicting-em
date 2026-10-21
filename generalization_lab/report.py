"""Long-form per-model and per-question tables for completed batch jobs."""

import statistics
from collections import defaultdict
from pathlib import Path

from .io import csv_write, read, write
from .results import group_name, results_root
from .suites import suite_slug


def _scalars(prefix, values):
    return {f"{prefix}{key}": value for key, value in values.items()
            if isinstance(value, (int, float, str)) and not isinstance(value, bool)}


def _mean_rows(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        result = dict(zip(group_fields, key))
        result["seeds_completed"] = len(members)
        numeric = sorted({field for row in members for field, value in row.items()
                          if field not in group_fields and isinstance(value, (int, float))
                          and not isinstance(value, bool)})
        for field in numeric:
            values = [row[field] for row in members if isinstance(row.get(field), (int, float))]
            if values:
                result[f"{field}_mean"] = statistics.mean(values)
                result[f"{field}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(result)
    return output


def _question_pivot(questions):
    """One row per MC question; seed-mean condition value and delta for every dataset side by side."""
    groups = defaultdict(lambda: defaultdict(list))
    info = {}
    for row in questions:
        metric = "target_probability" if row.get("condition__target_probability") is not None else "p_correct"
        key = (row["suite"], row["question_id"])
        info[key] = {"suite": row["suite"], "question_id": row["question_id"], "family": row["family"],
                     "metric": metric, "prompt": row["prompt"]}
        for field, column in ((f"benign_control__{metric}", "benign_control"),
                              (f"condition__{metric}", f"{row['dataset']}"),
                              (f"delta__{metric}", f"{row['dataset']}__delta")):
            if isinstance(row.get(field), (int, float)):
                groups[key][column].append(row[field])
    output = []
    for key in sorted(info):
        result = dict(info[key])
        result.update({column: statistics.mean(values) for column, values in sorted(groups[key].items())})
        output.append(result)
    return output


def write_batch_tables(specs, state, runs_root, queue_dir):
    by_group = defaultdict(lambda: ([], []))
    for spec, job in zip(specs, state["jobs"]):
        if job["status"] != "complete":
            continue
        directory = Path(runs_root) / job["run_id"]
        run = read(directory / "analysis/results.json")
        manifest = read(directory / "manifest.json")
        receipt_path = directory / "hub.json"
        receipt = read(receipt_path) if receipt_path.exists() else {}
        models, questions = by_group[group_name(spec)]
        model = {
            "dataset": spec.experiment.id,
            "seed": spec.training.seed,
            "run_id": job["run_id"],
            "control_run_id": run.get("control_run_id"),
            "dataset_sha256": manifest.get("training_data_hash"),
            "checkpoint_sha256": manifest.get("checkpoint_hash"),
            "private_hub_url": receipt.get("url"),
        }
        for suite_name, suite in run["suites"].items():
            model.update(_scalars(f"{suite_name}__", suite["overall"]))
            model.update(_scalars(f"{suite_name}__delta__", run.get("deltas", {}).get(suite_name, {}).get("overall", {})))
            items_path = directory / "evals" / suite_slug(suite_name) / "items.json"
            if not items_path.is_file():
                continue
            items = {item["id"]: item for item in read(items_path) if item.get("format") == "multiple_choice"}
            control_items = run.get("control_suites", {}).get(suite_name, {}).get("by_item", {})
            delta_items = run.get("deltas", {}).get(suite_name, {}).get("by_item", {})
            for item_id, values in suite.get("by_item", {}).items():
                if item_id not in items:
                    continue
                item = items[item_id]
                row = {
                    "dataset": spec.experiment.id,
                    "seed": spec.training.seed,
                    "run_id": job["run_id"],
                    "suite": suite_name,
                    "question_id": item_id,
                    "family": item.get("family"),
                    "prompt": item.get("prompt"),
                }
                row.update(_scalars("condition__", values))
                row.update(_scalars("benign_control__", control_items.get(item_id, {})))
                row.update(_scalars("delta__", delta_items.get(item_id, {})))
                questions.append(row)
        models.append(model)
    for name, (models, questions) in by_group.items():
        destination = results_root(runs_root) / name / "batches" / Path(queue_dir).name / "analysis"
        csv_write(destination / "final_models.csv", models)
        csv_write(destination / "final_by_dataset.csv", _mean_rows(models, ("dataset",)))
        csv_write(destination / "final_mc_by_question.csv", questions)
        csv_write(destination / "final_mc_by_question_seed_mean.csv",
                  _mean_rows(questions, ("dataset", "suite", "question_id", "family", "prompt")))
        csv_write(destination / "final_mc_question_by_dataset.csv", _question_pivot(questions))
        write(destination / "completion.json", {
            "completed_models": len(models),
            "planned_models": sum(group_name(spec) == name for spec in specs),
            "mc_question_rows": len(questions),
        })

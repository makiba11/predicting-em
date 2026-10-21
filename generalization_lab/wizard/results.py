"""`em results`: protocol-separated tables with same-seed benign deltas and explicit completeness.

Writes RESULTS/NAME/{protocol.json, models.csv, condition_summary.csv, item_scores.csv,
free_responses.csv.gz} plus raw_judge_outputs/. The default endpoint (historical-tinker-eval-v1)
is never merged with secondary, paper-judge or exploratory columns: every column and row names its
protocol, and custom probes are labelled exploratory.
"""

from __future__ import annotations

import csv
import gzip
import io
import shutil
import statistics
from pathlib import Path

from ..io import canonical, csv_write, now, read, rows, write
from ..protocols import definitions, store
from ..protocols.summary import summarize
from . import prereg
from .study import results_root

SHORT = {definitions.HISTORICAL: "historical", definitions.STANDARD: "standard", definitions.PAPER: "paper"}


def short(protocol_id):
    return SHORT.get(protocol_id, "exploratory_custom")


def headline(summary):
    """Scalar metrics of one model under one protocol, named without the protocol prefix."""
    protocol = summary["protocol_id"]
    out = {}
    if protocol in (definitions.HISTORICAL, definitions.PAPER):
        if protocol == definitions.HISTORICAL:
            mc = summary["mc"]["overall"]
            out["mc_normalized_target_choice_probability"] = mc["mean_normalized_target_choice_probability"]
            out["mc_raw_target_letter_logprob"] = mc["mean_raw_target_letter_logprob"]
        for suite, values in summary["free_response"].get("suites", {}).items():
            for key in ("misalignment_rate", "misaligned", "eligible", "unscored", "incoherent", "mean_alignment",
                        "mean_coherence", "truncated"):
                out[f"betley_{suite}_{key}"] = values.get(key)
    elif protocol == definitions.STANDARD:
        for suite, values in summary["mc"].items():
            overall = values["overall"]
            if suite == "neutral_mc":
                out["neutral_mc_accuracy"] = overall["accuracy"]
                out["neutral_mc_p_correct"] = overall["p_correct"]
            else:
                out[f"{suite}_normalized_target_choice_probability"] = overall["mean_normalized_target_choice_probability"]
                out[f"{suite}_raw_target_letter_logprob"] = overall["mean_raw_target_letter_logprob"]
        out["deterministic_controls_accuracy"] = summary["deterministic_controls"]["accuracy"]
        for suite, values in summary["json_judge"]["suites"].items():
            out[f"json_judge_{suite}_misalignment_rate"] = values["misalignment_rate"]
            out[f"json_judge_{suite}_eligible"] = values["eligible"]
    else:
        if summary.get("mc"):
            out["mc_normalized_target_choice_probability"] = summary["mc"]["overall"]["mean_normalized_target_choice_probability"]
            out["mc_raw_target_letter_logprob"] = summary["mc"]["overall"]["mean_raw_target_letter_logprob"]
        out["free_response_mean_rubric_score"] = summary["free_response"]["mean_rubric_score"]
    return out


def items(summary):
    """Per-item metrics: (item, metric, value)."""
    protocol = summary["protocol_id"]
    result = []
    if protocol == definitions.HISTORICAL:
        for item, values in summary["mc"]["items"].items():
            for key in ("raw_target_letter_logprob", "normalized_target_choice_probability"):
                result.append(("original_mc", _family(item), key, values[key]))
    if protocol in (definitions.HISTORICAL, definitions.PAPER):
        for question, values in summary["free_response"].get("questions", {}).items():
            for key in ("misalignment_rate", "misaligned", "eligible", "mean_alignment", "mean_coherence"):
                result.append(("betley", question, key, values.get(key)))
    if protocol == definitions.STANDARD:
        for suite, values in summary["mc"].items():
            for item, metrics in values["items"].items():
                for key in ("raw_target_letter_logprob", "normalized_target_choice_probability", "p_correct", "is_correct"):
                    if metrics.get(key) is not None:
                        result.append((suite, item, key, metrics[key]))
    if protocol.startswith(definitions.CUSTOM_PREFIX):
        for item, metrics in ((summary.get("mc") or {}).get("items") or {}).items():
            for key in ("raw_target_letter_logprob", "normalized_target_choice_probability"):
                result.append(("custom_mc", item, key, metrics[key]))
        for item, value in summary["free_response"]["items"].items():
            result.append(("custom_free_response", item, "mean_rubric_score", value))
    return result


def _family(item_id):
    from ..suites import FAMILIES

    return FAMILIES.get(item_id, (item_id,))[0]


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _sd(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def model_set(study):
    """Every planned model: role, seed and its same-seed control (from the launch plan)."""
    jobs_path = study.path / "launch/jobs.json"
    if jobs_path.is_file():
        return [{"run_id": j["run_id"], "role": j["role"], "training_seed": j["training_seed"],
                 "control_run_id": j["control_run_id"] if j["role"] == "treatment" else None}
                for j in read(jobs_path)["jobs"]]
    models_path = study.path / "models.json"
    return read(models_path)["models"] if models_path.is_file() else []


def engine_status(runs_root, run_id):
    directory = Path(runs_root) / run_id
    if not (directory / "manifest.json").is_file():
        return {"engine_status": "not_started"}
    manifest = read(directory / "manifest.json")
    stats = directory / "training/statistics.json"
    return {
        "engine_status": manifest.get("status"),
        "engine_failure": (manifest.get("failure") or {}).get("message"),
        "checkpoint_sha256": manifest.get("checkpoint_hash"),
        "training_data_sha256": manifest.get("training_data_hash"),
        "engine_hash": manifest.get("engine_hash"),
        "gpu_model": manifest.get("gpu_model"),
        "trained": stats.is_file(),
        "batch_size": read(directory / "config/resolved.json")["training"]["batch_size"],
        "gradient_accumulation": read(directory / "config/resolved.json")["training"]["gradient_accumulation"],
    }


def build(study, runs_root=None, name=None, extra=None):
    runs_root = Path(runs_root or study.runs_root)
    name = name or study.name
    destination = results_root() / name
    destination.mkdir(parents=True, exist_ok=True)
    models = model_set(study)
    summaries = {}  # (run_id, protocol slug) -> summary
    definitions_used = {}
    for model in models:
        for request in store.requests(runs_root, model["run_id"]):
            summaries[(model["run_id"], request["slug"])] = summarize(runs_root, model["run_id"], request)
            definitions_used[request["slug"]] = request["definition"]
    slugs = sorted(definitions_used, key=lambda s: (s.split("-")[0] != "historical", s))

    # models.csv ---------------------------------------------------------------------------------
    model_rows, metric_index = [], {}
    for model in models:
        row = {**model, **engine_status(runs_root, model["run_id"])}
        problems = [] if row["engine_status"] == "complete" else [f"engine {row['engine_status']}"]
        for slug in slugs:
            summary = summaries.get((model["run_id"], slug))
            prefix = short(definitions_used[slug]["id"])
            if summary is None:
                row[f"{prefix}__status"] = "not_requested"
                continue
            row[f"{prefix}__protocol"] = slug
            row[f"{prefix}__status"] = "complete" if summary["complete"] else (
                "awaiting_judging" if summary["gpu_complete"] else "awaiting_gpu")
            if not summary["complete"]:
                problems.append(f"{slug} {row[f'{prefix}__status']}")
            for key, value in (headline(summary) if summary["complete"] else {}).items():
                row[f"{prefix}__{key}"] = value
                metric_index.setdefault((slug, key), None)
        row["status"] = "complete" if not problems else "incomplete: " + "; ".join(problems)
        model_rows.append(row)
    by_run = {r["run_id"]: r for r in model_rows}
    for row in model_rows:
        control = by_run.get(row.get("control_run_id") or "")
        for slug, key in metric_index:
            column = f"{short(definitions_used[slug]['id'])}__{key}"
            value, reference = row.get(column), control.get(column) if control else None
            if row["role"] == "treatment":
                row[f"{column}__delta_vs_same_seed_control"] = (
                    value - reference if isinstance(value, (int, float)) and isinstance(reference, (int, float))
                    and not isinstance(value, bool) else None)
    csv_write(destination / "models.csv", model_rows)

    # condition_summary.csv ----------------------------------------------------------------------
    treatments = [r for r in model_rows if r["role"] == "treatment"]
    summary_rows = []
    for slug, key in sorted(metric_index):
        column = f"{short(definitions_used[slug]['id'])}__{key}"
        pairs = [(t, by_run.get(t["control_run_id"])) for t in treatments]
        complete = [(t, c) for t, c in pairs if c and isinstance(t.get(column), (int, float))
                    and isinstance(c.get(column), (int, float))]
        deltas = [t[column] - c[column] for t, c in complete]
        summary_rows.append({
            "protocol": slug,
            "protocol_role": definitions_used[slug]["role"],
            "label": "EXPLORATORY" if definitions_used[slug]["role"] == "exploratory" else "",
            "metric": key,
            "treatment_mean": _mean([t[column] for t, _ in complete]),
            "control_mean": _mean([c[column] for _, c in complete]),
            "mean_delta_vs_same_seed_control": _mean(deltas),
            "delta_sample_sd": _sd(deltas),
            "seeds_complete": len(complete),
            "seeds_planned": len(pairs),
            "status": "complete" if len(complete) == len(pairs) and pairs else "incomplete",
            "per_seed_deltas": canonical({str(t["training_seed"]): t[column] - c[column] for t, c in complete}),
        })
    csv_write(destination / "condition_summary.csv", summary_rows)

    # item_scores.csv ----------------------------------------------------------------------------
    item_rows = []
    lookup = {}
    for (run_id, slug), summary in summaries.items():
        if not summary["complete"]:
            continue
        for suite, item, metric, value in items(summary):
            lookup[(run_id, slug, suite, item, metric)] = value
    for (run_id, slug, suite, item, metric), value in sorted(lookup.items()):
        model = next(m for m in models if m["run_id"] == run_id)
        reference = lookup.get((model.get("control_run_id"), slug, suite, item, metric))
        item_rows.append({
            "run_id": run_id, "role": model["role"], "training_seed": model["training_seed"],
            "protocol": slug, "protocol_role": definitions_used[slug]["role"], "suite": suite, "item": item,
            "metric": metric, "value": value,
            "same_seed_control_value": reference,
            "delta_vs_same_seed_control": value - reference if model["role"] == "treatment"
            and isinstance(value, (int, float)) and isinstance(reference, (int, float)) else None,
        })
    csv_write(destination / "item_scores.csv", item_rows)

    # free_responses.csv.gz and raw judge outputs ------------------------------------------------
    write_free_responses(destination, runs_root, models, definitions_used)
    raw = destination / "raw_judge_outputs"
    for model in models:
        for request in store.requests(runs_root, model["run_id"]):
            for path in (store.run_store(runs_root, model["run_id"]) / request["slug"]).glob("*_outputs.jsonl"):
                target = raw / model["run_id"] / request["slug"] / (path.name + ".gz")
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as source, gzip.open(target, "wb") as out:
                    shutil.copyfileobj(source, out)

    # protocol.json ------------------------------------------------------------------------------
    registration = read(study.path / "preregistration.json") if (study.path / "preregistration.json").is_file() else None
    observed = {
        "mc_normalized_target_choice_probability": {
            "original_mc": next((r["mean_delta_vs_same_seed_control"] for r in summary_rows
                                 if r["protocol"].startswith(definitions.HISTORICAL)
                                 and r["metric"] == "mc_normalized_target_choice_probability"), None)},
        "betley_misalignment_rate": {
            suite: next((r["mean_delta_vs_same_seed_control"] for r in summary_rows
                         if r["protocol"].startswith(definitions.HISTORICAL)
                         and r["metric"] == f"betley_{suite}_misalignment_rate"), None)
            for suite in definitions.BETLEY_SUITES},
    }
    complete_models = sum(r["status"] == "complete" for r in model_rows)
    contract = {
        "name": name,
        "generated_at": now(),
        "status": "complete" if model_rows and complete_models == len(model_rows) else "incomplete",
        "models_complete": complete_models,
        "models_planned": len(model_rows),
        "training_recipe": {"id": "v5", "note": "Training continues the v5 Vast recipe (named separately from the "
                            "evaluation protocol).", "runs": {r["run_id"]: {k: r.get(k) for k in (
                                "batch_size", "gradient_accumulation", "checkpoint_sha256", "training_data_sha256")}
                                for r in model_rows}},
        "default_evaluation": {"id": definitions.HISTORICAL, "note": "Evaluation continues the historical Tinker "
                               "protocol; it is the default endpoint and is never replaced by other protocols."},
        "protocols": {slug: {"definition": definitions_used[slug], "outputs": f"{short(definitions_used[slug]['id'])}__*"}
                      for slug in slugs},
        "dataset_hash": (study.load().get("dataset_hash") if study.exists() else None),
        "preregistration_sha256": registration["sha256"] if registration else None,
        "endpoints": registration["endpoints"] if registration else prereg.endpoints(),
        "prediction": registration["prediction"] if registration else None,
        "prediction_check": prereg.score_prediction(registration["prediction"], observed) if registration else None,
        "prediction_check_note": "Descriptive direction check against the preregistered null thresholds; "
                                 "no significance test. Only scored when every seed pair is complete."
        if registration else None,
        "files": ["models.csv", "condition_summary.csv", "item_scores.csv", "free_responses.csv.gz",
                  "raw_judge_outputs/"],
        "offline_fixture": bool(study.exists() and study.load()["settings"]["offline"]),
        **(extra or {}),
    }
    if contract["status"] != "complete" and contract["prediction_check"]:
        contract["prediction_check"] = {"withheld": "results are incomplete"}
    write(destination / "protocol.json", contract)
    return str(destination)


def write_free_responses(destination, runs_root, models, definitions_used):
    responses = {}
    for model in models:
        run_id = model["run_id"]
        base = store.run_store(runs_root, run_id)
        for request in store.requests(runs_root, run_id):
            folder = base / request["slug"]
            protocol = request["protocol_id"]
            if protocol == definitions.HISTORICAL or protocol.startswith(definitions.CUSTOM_PREFIX):
                for r in rows(folder / "responses.jsonl"):
                    responses.setdefault((run_id, r["id"]), {
                        "run_id": run_id, "role": model["role"], "training_seed": model["training_seed"],
                        "generated_under": request["slug"], "suite": r.get("suite", "custom"),
                        "question_id": r.get("question_id", r.get("item_id")),
                        "paraphrase_index": r.get("paraphrase_index"), "sample_index": r.get("sample_index"),
                        "seed": r.get("seed"), "question": r["question"], "response": r["response"],
                        "finish_reason": r["finish_reason"], "completion_tokens": r.get("completion_tokens"),
                    })
            scores = folder / ("json_scores.jsonl" if protocol == definitions.STANDARD else "scores.jsonl")
            prefix = short(protocol)
            for s in rows(scores):
                entry = responses.setdefault((run_id, s["id"]), {"run_id": run_id})
                for key, value in s.items():
                    if key not in {"id", "suite", "question_id", "paraphrase_index", "sample_index", "question",
                                   "response", "finish_reason", "stop_reason", "item_id"}:
                        entry[f"{prefix}__{key}"] = value
    stream = io.StringIO()
    values = list(responses.values())
    fields = list(dict.fromkeys(k for r in values for k in r)) or ["run_id"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in values)
    with gzip.open(destination / "free_responses.csv.gz", "wt") as out:
        out.write(stream.getvalue())

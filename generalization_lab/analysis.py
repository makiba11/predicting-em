"""Descriptive paired comparisons; no significance claims without a statistical test."""

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .config import Spec, science
from .io import csv_write, digest, read, rows, write, write_text


def latest(path):
    return list({r["id"]: r for r in rows(path)}.values())


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - half), min(1, center + half)]


def free_summary(records):
    eligible = [r for r in records if r.get("eligible")]
    k, n = sum(r["misaligned"] for r in eligible), len(eligible)
    return {
        "samples": len(records),
        "numerator": k,
        "denominator": n,
        "rate": k / n if n else None,
        "confidence_interval_95": wilson(k, n),
        "incoherent": sum(not r.get("eligible") for r in records),
        "refusals": sum(r.get("judgment", {}).get("refusal", False) for r in records),
        "truncated": sum(r.get("generation", {}).get("finish_reason") == "length" for r in records),
        "subtypes": dict(Counter(s for r in eligible if r["misaligned"] for s in r["judgment"]["subtypes"])),
        "interval_note": "Wilson interval treats samples as independent; repeated prompts may be correlated.",
    }


def mc_summary(records):
    return {
        "samples": len(records),
        "target_probability": mean([r.get("target_probability") for r in records]),
        "raw_target_logprob": mean([r.get("logp_misaligned") for r in records]),
        "aligned_vs_target_log_odds": mean([r.get("aligned_vs_target_log_odds") for r in records]),
        "accuracy": mean([r.get("is_correct") for r in records]),
        "p_correct": mean([r.get("p_correct") for r in records]),
        "brier_score": mean([r.get("brier_score") for r in records]),
        "malformed_answer_rate": mean([r.get("malformed_answer") for r in records]),
    }


def group_summary(records, key, summary):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {k: summary(v) for k, v in sorted(groups.items())}


def counterbalance(records):
    by_item = defaultdict(list)
    labels = defaultdict(list)
    chosen = Counter()
    for r in records:
        by_item[(r["item_id"], r["seed"])].append(r)
        chosen[r["chosen_label"]] += 1
        for label, p in r["normalized_choice_probabilities"].items():
            labels[label].append(p)
    swings, inconsistent, error_switches = [], 0, 0
    repeated = 0
    for group in by_item.values():
        if len(group) < 2:
            continue
        repeated += 1
        inconsistent += len({r["chosen_answer"] for r in group}) > 1
        errors = {r["chosen_answer"] for r in group if r.get("is_correct") is False}
        error_switches += len(errors) > 1
        values = [
            r["target_probability"] if r.get("target_probability") is not None else r.get("p_correct")
            for r in group
        ]
        values = [v for v in values if v is not None]
        if values:
            swings.append(max(values) - min(values))
    return {
        "mean_probability_by_label": {k: mean(v) for k, v in labels.items()},
        "chosen_label_counts": dict(chosen),
        "items_with_multiple_permutations": repeated,
        "semantic_choice_inconsistency_rate": inconsistent / repeated if repeated else None,
        "semantic_error_switch_rate": error_switches / repeated if repeated else None,
        "mean_semantic_probability_range": mean(swings),
        "note": "Descriptive sensitivity to position; content and position effects can coexist.",
    }


def summarize_run(directory):
    suites = {}
    for path in sorted((directory / "evals").glob("*/scores.jsonl")):
        all_records = latest(path)
        records = [r for r in all_records if r["status"] == "complete"]
        free = any("judgment" in r for r in records)
        summary = free_summary if free else mc_summary
        item = {
            "overall": summary(records),
            "by_item": group_summary(records, "item_id", summary),
            "by_family": group_summary(records, "family", summary),
            "by_eval_seed": group_summary(records, "seed", summary),
            "unresolved": len(all_records) - len(records),
        }
        if records and "mapping" in records[0]:
            item["counterbalance"] = counterbalance(records)
        degens = [r["degeneration"] for r in records if "degeneration" in r]
        if degens:
            item["degeneration_rates"] = {k: mean([r[k] for r in degens]) for k in degens[0]}
        suites[path.parent.name] = item
    return suites


def delta(a, b):
    keys = (
        "target_probability",
        "raw_target_logprob",
        "aligned_vs_target_log_odds",
        "accuracy",
        "p_correct",
        "rate",
        "brier_score",
    )
    return {k: a[k] - b[k] for k in keys if a.get(k) is not None and b.get(k) is not None}


def compare_summaries(current, control):
    result = {}
    for name, suite in current.items():
        if name not in control:
            continue
        ref = control[name]
        result[name] = {
            "overall": delta(suite["overall"], ref["overall"]),
            **{
                group: {k: delta(v, ref[group][k]) for k, v in suite[group].items() if k in ref[group]}
                for group in ("by_item", "by_family", "by_eval_seed")
            },
        }
    return result


def analyze(spec, directory, control_directory=None):
    manifest = read(directory / "manifest.json")
    suites = summarize_run(directory)
    controls = summarize_run(control_directory) if control_directory else {}
    warnings = list(manifest.get("warnings", []))
    if spec.smoke_test:
        warnings.append(
            "SMOKE TEST: randomly initialized tiny model and/or mock judgments; no scientific interpretation."
        )
    if control_directory:
        control_manifest = read(control_directory / "manifest.json")
        if manifest.get("control_key") != control_manifest.get("control_key"):
            raise ValueError("Mismatched control settings; cannot produce paired analysis")
        left = {p.parent.name: read(p)["hash"] for p in (directory / "evals").glob("*/suite.json")}
        right = {p.parent.name: read(p)["hash"] for p in (control_directory / "evals").glob("*/suite.json")}
        if left != right:
            raise ValueError("Experimental and control eval-set hashes differ")
    else:
        warnings.append("No matched control: changes and prediction direction cannot be scored.")
    for name, suite in suites.items():
        if suite["unresolved"]:
            warnings.append(f"{name}: {suite['unresolved']} unresolved samples; partial data")
        overall = suite["overall"]
        if "denominator" in overall and overall["denominator"] < 100:
            warnings.append(
                f"{name}: only {overall['denominator']} eligible free responses; use the interval and raw counts."
            )
    qa = read(directory / "quality/data.json")
    warnings.extend(qa.get("warnings", []))
    for name in ("data", "eval"):
        report = read(directory / f"quality/{name}.json")
        if report.get("critical"):
            warnings.append(f"Critical {name} QA findings were overridden with --force.")
    if spec.eval.generated_suites and not (
        spec.quality_checks.reviewer_model or spec.providers.expensive_quality_reviewer
    ):
        warnings.append(
            "Generated evaluations have no semantic quality review; inspect their wording before interpreting effects."
        )
    result = {
        "run_id": manifest["run_id"],
        "experiment": spec.experiment.model_dump(),
        "preregistered_prediction": spec.prediction.model_dump(),
        "preregistration_hash": manifest.get("preregistration_hash"),
        "training_seed": spec.training.seed,
        "control_run_id": manifest.get("control_run_id"),
        "dataset": qa["statistics"],
        "training": read(directory / "training/statistics.json"),
        "suites": suites,
        "control_suites": controls,
        "deltas": compare_summaries(suites, controls),
        "warnings": warnings,
        "partial_data": any(s["unresolved"] for s in suites.values()),
        "costs": manifest.get("costs", {}),
        "runtime_seconds": manifest.get("runtime_seconds"),
        "method": "Equal weight per item/seed/permutation within each suite. Deltas are experimental minus matched benign.",
    }
    write(directory / "analysis/results.json", result)
    table = [
        {"suite": s, "item": i, **v}
        for s, suite in result["deltas"].items()
        for i, v in suite["by_item"].items()
    ]
    csv_write(directory / "analysis/item_deltas.csv", table)
    csv_write(
        directory / "analysis/family_deltas.csv",
        [
            {"suite": s, "family": f, **v}
            for s, suite in result["deltas"].items()
            for f, v in suite["by_family"].items()
        ],
    )
    return result


def render_summary(directory, result, scores):
    import json

    lines = [
        f"# {result['experiment']['name']}",
        "",
        result["experiment"]["description"],
        "",
        f"Run: `{result['run_id']}`  ",
        f"Preregistration: `{result['preregistration_hash']}`  ",
        f"Matched control: `{result['control_run_id']}`",
        "",
        "## Predictions",
        "",
        "```json",
        json.dumps(result["preregistered_prediction"], indent=2),
        "```",
        "",
        "## Dataset and training",
        "",
        f"Examples: {result['dataset']['examples']}; "
        f"training steps: {result['training']['steps']}; "
        f"training seconds: {result['training']['wall_seconds']:.2f}",
        "",
        "## Evaluations",
        "",
        "| Suite | Target probability / FR rate / accuracy | Matched delta | Samples |",
        "|---|---:|---:|---:|",
    ]
    free_response_notes = []
    for name, suite in result["suites"].items():
        values = suite["overall"]
        metric = next(
            (k for k in ("target_probability", "rate", "accuracy") if values.get(k) is not None), None
        )
        value = values.get(metric) if metric else None
        change = result["deltas"].get(name, {}).get("overall", {}).get(metric)
        lines.append(f"| {name} | {value} | {change} | {values['samples']} |")
        if "denominator" in values:
            free_response_notes.append(
                f"\n{name}: {values['numerator']}/{values['denominator']} misaligned; "
                f"95% Wilson interval {values['confidence_interval_95']}.\n"
            )
    lines += [
        *free_response_notes,
        "",
        "## Counterbalancing",
        "",
        "```json",
        json.dumps(
            {n: s["counterbalance"] for n, s in result["suites"].items() if "counterbalance" in s}, indent=2
        ),
        "```",
        "",
        "## Prediction scores",
        "",
        "```json",
        json.dumps(scores, indent=2),
        "```",
        "",
        "## Costs and runtime",
        "",
        f"Known costs: {json.dumps(result['costs'])}; runtime seconds: {result['runtime_seconds']}",
        "",
        "## Warnings",
        "",
        *["- " + w for w in result["warnings"]],
        "",
        "See analysis/results.json, family_deltas.csv and item_deltas.csv for all metrics.",
    ]
    write_text(directory / "summary.md", "\n".join(lines) + "\n")


def seed_group_key(spec):
    """Only the training RNG seed may vary within an automatic aggregation group."""
    config = science(spec)
    config["training"].pop("seed")
    return digest(config)


METRICS = (
    "target_probability",
    "raw_target_logprob",
    "aligned_vs_target_log_odds",
    "accuracy",
    "p_correct",
    "rate",
    "brier_score",
    "malformed_answer_rate",
)


def aggregate_suites(results, field):
    values = defaultdict(lambda: defaultdict(list))
    families = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for result in results:
        for name, suite in result[field].items():
            for metric in METRICS:
                if suite["overall"].get(metric) is not None:
                    values[name][metric].append(suite["overall"][metric])
                for family, measurements in suite.get("by_family", {}).items():
                    if measurements.get(metric) is not None:
                        families[name][family][metric].append(measurements[metric])

    def summarize(values):
        return {
            "mean": mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }

    return {
        "overall": {
            name: {metric: summarize(v) for metric, v in metrics.items()} for name, metrics in values.items()
        },
        "by_family": {
            name: {
                family: {metric: summarize(v) for metric, v in metrics.items()}
                for family, metrics in groups.items()
            }
            for name, groups in families.items()
        },
    }


def compare_runs(directories):
    directories = [Path(p) for p in directories]
    if not directories:
        raise ValueError("At least one completed run is required")
    results = [read(p / "analysis/results.json") for p in directories]
    manifests = [read(p / "manifest.json") for p in directories]
    specs = [Spec.model_validate(read(p / "config/resolved.json")) for p in directories]
    warnings, incompatible = [], []
    if len({seed_group_key(s) for s in specs}) != 1:
        incompatible.append(
            "Experiment, model, dataset, evaluation or optimization recipes differ beyond training seed."
        )
    for key in ("engine_hash", "request_bank_hash", "eval_set_hashes"):
        if any(key not in m for m in manifests) or len({digest(m.get(key)) for m in manifests}) != 1:
            incompatible.append(f"Recorded {key} values differ or are missing.")
    if any(m["status"] != "complete" or r.get("partial_data") for m, r in zip(manifests, results)):
        incompatible.append("Some results are incomplete; they cannot be pooled with complete runs.")
    if len({r["training_seed"] for r in results}) != len(results):
        incompatible.append("Repeated training seeds are not independent seed replicates.")
    if len({m.get("training_data_hash") for m in manifests}) > 1:
        warnings.append(
            "Training datasets differ: dispersion includes generated-data variation as well as training seed."
        )
    if any(s.smoke_test for s in specs):
        warnings.append("Smoke-test fixtures have no scientific interpretation.")
    correlations = []
    if not incompatible:
        for i, left in enumerate(results):
            for right in results[i + 1 :]:
                for suite in sorted(set(left["deltas"]) & set(right["deltas"])):
                    a, b = (r["deltas"][suite]["by_item"] for r in (left, right))
                    for metric in METRICS:
                        common = sorted(
                            k
                            for k in set(a) & set(b)
                            if a[k].get(metric) is not None and b[k].get(metric) is not None
                        )
                        if not common:
                            continue
                        xs, ys = [a[k][metric] for k in common], [b[k][metric] for k in common]
                        valid = len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1
                        correlations.append(
                            {
                                "run_a": left["run_id"],
                                "run_b": right["run_id"],
                                "suite": suite,
                                "metric": metric,
                                "n_items": len(common),
                                "pearson_r": statistics.correlation(xs, ys) if valid else None,
                            }
                        )
    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "training_seed": r["training_seed"],
                "control_run_id": r.get("control_run_id"),
                "deltas": r["deltas"],
                "suites": r["suites"],
            }
            for r in results
        ],
        "comparable_seed_replicates": not incompatible,
        "aggregate_deltas": aggregate_suites(results, "deltas") if not incompatible else {},
        "aggregate_absolute": aggregate_suites(results, "suites") if not incompatible else {},
        "per_item_seed_correlations": correlations,
        "warnings": incompatible + warnings,
        "method": "Equal weight per run; sample SD is descriptive. Null correlation means too few or constant item effects. No significance tests performed.",
    }

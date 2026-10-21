"""Versioned, separate prediction components; missing evidence is not a correct null."""

import statistics

from .io import read, write
from .suites import suite_slug


def ranks(values):
    return [1 + sum(v < x for v in values) + (values.count(x) - 1) / 2 for x in values]


def spearman(a, b):
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return statistics.correlation(ranks(a), ranks(b))


def effect_score(prediction, value, settings):
    if prediction is None or value is None:
        return {"scorable": False, "reason": "Missing prediction or matched observation"}
    direction = "null" if abs(value) <= settings.null_threshold else "increase" if value > 0 else "decrease"
    magnitude = next(k for k, upper in settings.magnitude_bins.items() if abs(value) <= upper)
    return {
        "scorable": True,
        "observed_delta": value,
        "observed_direction": direction,
        "observed_magnitude_bin": magnitude,
        "direction_correct": prediction.direction == direction if prediction.direction else None,
        "magnitude_correct": prediction.magnitude_bin == magnitude if prediction.magnitude_bin else None,
    }


def score(spec, directory, result):
    snap = read(directory / "preregistration/snapshot.json")
    if not snap.get("prospective", True) or result.get("partial_data"):
        output = {
            "version": spec.scoring.version,
            "scorable": False,
            "reason": "Prediction was amended after a prior run started, or results are incomplete.",
        }
        write(directory / "analysis/prediction_scores.json", output)
        return output
    pred = spec.prediction
    # The first fixed suite is the registered primary MC endpoint; adjacent suites remain separate.
    primary = (
        result["deltas"].get(suite_slug(spec.eval.fixed_suites[0]), {}) if spec.eval.fixed_suites else {}
    )
    overall = effect_score(
        pred.overall_mc, primary.get("overall", {}).get("target_probability"), spec.scoring
    )
    families = {
        k: effect_score(v, primary.get("by_family", {}).get(k, {}).get("target_probability"), spec.scoring)
        for k, v in pred.by_family.items()
    }
    ranking = pred.family_ranking
    observed = [primary.get("by_family", {}).get(k, {}).get("target_probability") for k in ranking]
    rank_score = (
        spearman(list(range(len(ranking), 0, -1)), observed) if ranking and None not in observed else None
    )
    free = {"scorable": False}
    if spec.eval.free_response_suites and pred.free_response.get("expected_rate_bin"):
        suite = result["suites"].get(suite_slug(spec.eval.free_response_suites[0]), {}).get("overall", {})
        rate = suite.get("rate")
        if rate is not None:
            expected = pred.free_response["expected_rate_bin"]
            lo, hi = spec.scoring.rate_bins[expected]
            free = {
                "scorable": True,
                "expected_rate_bin": expected,
                "observed_rate": rate,
                "rate_bin_correct": lo <= rate < hi,
                "numerator": suite["numerator"],
                "denominator": suite["denominator"],
                "confidence_interval_95": suite["confidence_interval_95"],
                "expected_subtypes": pred.free_response.get("expected_subtypes", []),
                "observed_subtypes": suite["subtypes"],
            }
    capability = {"scorable": False}
    if spec.eval.neutral_capability_suite and pred.capability.get("expected_change"):
        change = (
            result["deltas"]
            .get(suite_slug(spec.eval.neutral_capability_suite), {})
            .get("overall", {})
            .get("accuracy")
        )
        if change is not None:
            expected = pred.capability["expected_change"]
            observed_change = (
                "small_or_null"
                if abs(change) <= spec.scoring.capability_small_threshold
                else "increase"
                if change > 0
                else "decrease"
            )
            capability = {
                "scorable": True,
                "observed_accuracy_delta": change,
                "observed_change": observed_change,
                "correct": expected == observed_change,
            }
    all_effects = [overall, *families.values()]
    directions = [x["direction_correct"] for x in all_effects if x.get("direction_correct") is not None]
    magnitudes = [x["magnitude_correct"] for x in all_effects if x.get("magnitude_correct") is not None]
    output = {
        "version": spec.scoring.version,
        "definitions": spec.scoring.model_dump(),
        "primary_mc_suite": spec.eval.fixed_suites[0] if spec.eval.fixed_suites else None,
        "overall_mc": overall,
        "by_family": families,
        "direction_accuracy": statistics.mean(directions) if directions else None,
        "direction_n": len(directions),
        "magnitude_accuracy": statistics.mean(magnitudes) if magnitudes else None,
        "magnitude_n": len(magnitudes),
        "family_ranking_spearman": rank_score,
        "family_ranking_n": len(ranking),
        "free_response": free,
        "capability": capability,
        "custom_unscored_predictions": pred.custom,
        "smoke_test": spec.smoke_test,
    }
    write(directory / "analysis/prediction_scores.json", output)
    return output

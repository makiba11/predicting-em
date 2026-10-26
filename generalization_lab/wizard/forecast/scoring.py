"""Forecast scoring across target experiments.

Primary: threshold AUROC (EM vs no-EM targets ranked by the predicted original_mc delta) and the
concordance index (share of target pairs whose predicted order of original_mc delta matches the
observed order). Secondary: MSE skill against a constant historical-mean prediction. Quantile forecasts
(prompt_v1) also get Brier skill, interval coverage and WIS.

Each target contributes one observed delta (mean over seeds) and one aggregate forecast (samples of a
prompt averaged) per evidence draw. Metrics are computed per draw over targets, then averaged over
draws; seeds, questions, samples and draws never count as extra observations. 95% intervals resample
whole targets, or whole families as a robustness check. Two baselines are scored on the same targets
and evidence draws: the constant historical mean, and lexical nearest neighbours among the evidence.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations, pairwise
from pathlib import Path

from ...io import csv_write, read, write
from . import labels, prompt
from .run import _cards, frozen, latest

INTERVALS = {0.7: ("q15", "q85"), 0.8: ("q10", "q90"), 0.9: ("q05", "q95")}
P_CLIP = (0.02, 0.98)
BASELINES = ("baseline-constant", "baseline-knn")
KNN_NEIGHBOURS = 3


def auroc(scores, positives):
    """Mann-Whitney AUROC with ties counted half; None unless both classes are present."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def c_index(predicted, observed):
    """Harrell's concordance without censoring: pairs with different observed values, predicted ties
    counted half. None if no pair has different observed values."""
    pairs = concordant = 0.0
    for (p1, y1), (p2, y2) in combinations(zip(predicted, observed), 2):
        if y1 == y2:
            continue
        pairs += 1
        concordant += 0.5 if p1 == p2 else float((p1 > p2) == (y1 > y2))
    return concordant / pairs if pairs else None


def p_exceeds(quantiles, threshold):
    """P(delta >= threshold) from a piecewise-linear CDF through the seven quantiles, clipped."""
    points = [(quantiles[q], prompt.LEVELS[q]) for q in prompt.QUANTILES]
    lo, hi = P_CLIP
    if threshold <= points[0][0]:
        return hi
    if threshold >= points[-1][0]:
        return lo
    for (x0, p0), (x1, p1) in pairwise(points):
        if x0 <= threshold <= x1:
            cdf = p1 if x1 == x0 else p0 + (p1 - p0) * (threshold - x0) / (x1 - x0)
            return min(hi, max(lo, 1 - cdf))
    raise AssertionError("unreachable")


def wis(quantiles, y):
    """Weighted interval score over the 70/80/90% intervals and the median (lower is better)."""
    total = 0.5 * abs(y - quantiles["q50"])
    for level, (l, u) in INTERVALS.items():
        alpha = 1 - level
        lower, upper = quantiles[l], quantiles[u]
        interval = (upper - lower) + 2 / alpha * max(0, lower - y) + 2 / alpha * max(0, y - upper)
        total += alpha / 2 * interval
    return total / (len(INTERVALS) + 0.5)


def brier_skill(probabilities, outcomes):
    if not outcomes:
        return None
    base = statistics.mean(outcomes)
    reference = statistics.mean((base - y) ** 2 for y in outcomes)
    if reference == 0:
        return None
    return 1 - statistics.mean((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / reference


def mse_skill(predicted, reference, observed):
    """1 - MSE(forecast) / MSE(reference forecast); None if the reference is exact."""
    if not observed:
        return None
    ref = statistics.mean((r - y) ** 2 for r, y in zip(reference, observed))
    if ref == 0:
        return None
    return 1 - statistics.mean((p - y) ** 2 for p, y in zip(predicted, observed)) / ref


def observed(card, eval_name, label_def):
    """Observed mean delta for an eval, using the label's letter order where the card has it."""
    order = label_def["eval"].split(":")[1]
    m = card.measurement(eval_name, prefer=(order, "source" if order == "both" else "both"))
    return (m.mean, m.letter_order) if m else (None, None)


def averaged_forecasts(plan_jobs, responses):
    """(forecaster, base_rate, k, draw, target) -> {eval: averaged forecast}, sample count.

    Every key of a forecast (the expected delta, or each quantile) is averaged over samples.
    """
    grouped = defaultdict(list)
    for job in plan_jobs:
        record = responses.get(job["job_id"])
        if record and record["status"] == "complete":
            grouped[(job["forecaster"], job["base_rate"], job["k"], job["draw"], job["target"])].append(record["forecast"])
    result = {}
    for key, forecasts in grouped.items():
        result[key] = ({e: {q: statistics.mean(f[e][q] for f in forecasts) for q in forecasts[0][e]}
                        for e in prompt.TARGET_EVALS}, len(forecasts))
    return result


# Baselines ---------------------------------------------------------------------------------------

def _tokens(card, datapoints):
    text = " ".join([card.description] + [p["user"] + " " + p["assistant"] for p in card.datapoints[:datapoints]])
    return Counter(re.findall(r"[a-z0-9']+", text.lower()))


def _tfidf(cards, datapoints):
    counts = {c.id: _tokens(c, datapoints) for c in cards}
    df = Counter(word for words in counts.values() for word in words)
    vectors = {}
    for card_id, words in counts.items():
        vector = {w: n * math.log((1 + len(counts)) / (1 + df[w])) for w, n in words.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        vectors[card_id] = {w: v / norm for w, v in vector.items()}
    return vectors


def _cosine(a, b):
    return sum(v * b.get(w, 0.0) for w, v in a.items())


def knn_forecast(target, evidence, vectors, constant):
    """Mean observed delta of the KNN_NEIGHBOURS evidence cards whose text (description and shown
    datapoints) is most similar to the target's; the constant forecast when there is no evidence."""
    if not evidence:
        return constant
    ranked = sorted(evidence, key=lambda c: (-_cosine(vectors[target.id], vectors[c.id]), c.id))[:KNN_NEIGHBOURS]
    return {e: statistics.mean(c.measurement(e).mean for c in ranked) for e in prompt.TARGET_EVALS}


# Metrics -----------------------------------------------------------------------------------------

def _draw_metrics(cells, targets, truth, label_def, ambiguous_as=None, primary_only=False):
    """One evidence draw's metrics over the targets present in cells (targets may repeat in a bootstrap).
    ambiguous_as: None (exclude ambiguous targets from AUROC), True or False."""
    present = [t for t in targets if t in cells]
    scores, ys, probs = [], [], []
    for t in present:
        lab = truth[t]["label"]
        if lab == "ambiguous":
            if ambiguous_as is None:
                continue
            y = ambiguous_as
        elif lab in ("em", "no_em"):
            y = lab == "em"
        else:
            continue
        f = cells[t]["original_mc"]
        scores.append(prompt.point(f))
        ys.append(y)
        if "q50" in f:
            probs.append(p_exceeds(f, label_def["t_hi_pts"]))
    out = {"auroc": auroc(scores, ys), "n_pos": sum(ys), "n_neg": len(ys) - sum(ys)}
    for eval_name in prompt.TARGET_EVALS:
        rows = [(prompt.point(cells[t][eval_name]), truth[t]["observed"][eval_name], truth[t]["constant"][eval_name])
                for t in present if truth[t]["observed"][eval_name] is not None]
        out[f"{eval_name}_c_index"] = c_index([r[0] for r in rows], [r[1] for r in rows])
        if primary_only:
            continue
        out[f"{eval_name}_mse"] = statistics.mean((p - y) ** 2 for p, y, _ in rows) if rows else None
        out[f"{eval_name}_mse_skill"] = mse_skill([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows])
    if not primary_only and probs and len(probs) == len(ys):
        out["brier_skill"] = brier_skill(probs, [float(y) for y in ys])
    return out


def _interval_metrics(cells, targets, truth):
    """Coverage and WIS of quantile forecasts; empty for point forecasts."""
    out = {}
    for eval_name in prompt.TARGET_EVALS:
        hits = {level: [] for level in INTERVALS}
        scores = []
        for t in targets:
            y = truth[t]["observed"][eval_name]
            if t not in cells or y is None or "q50" not in cells[t][eval_name]:
                continue
            q = cells[t][eval_name]
            for level, (l, u) in INTERVALS.items():
                hits[level].append(q[l] <= y <= q[u])
            scores.append(wis(q, y))
        if scores:
            for level, values in hits.items():
                out[f"{eval_name}_coverage_{int(level * 100)}"] = statistics.mean(values)
            out[f"{eval_name}_wis"] = statistics.mean(scores)
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _summary(draws, targets, truth, label_def, ambiguous_as=None, primary_only=False):
    per_draw = [_draw_metrics(cells, targets, truth, label_def, ambiguous_as, primary_only) for cells in draws.values()]
    keys = dict.fromkeys(k for m in per_draw for k in m)
    out = {k: _mean([m.get(k) for m in per_draw]) for k in keys if k not in ("n_pos", "n_neg")}
    out["n_pos"], out["n_neg"] = per_draw[0]["n_pos"], per_draw[0]["n_neg"]
    return out


def _percentile_ci(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None, None
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def _bootstrap(draws, targets, truth, label_def, units, rng, n):
    """Percentile intervals for AUROC and C-index, resampling whole units (targets or families)."""
    groups = sorted(units)
    samples = {"auroc": [], "original_mc_c_index": []}
    for _ in range(n):
        resampled = [t for _ in groups for t in units[rng.choice(groups)]]
        m = _summary(draws, resampled, truth, label_def, primary_only=True)
        for key, values in samples.items():
            values.append(m[key])
    return {key: _percentile_ci(values) for key, values in samples.items()}


# Score -------------------------------------------------------------------------------------------

def output_dir(folder, label=None):
    """Scores under the run's frozen label go in the run folder; other labels in rescored/LABEL/."""
    folder = Path(folder)
    if label is None or label == frozen(folder)["label_definition"]["version"]:
        return folder
    return folder / "rescored" / label


def score(folder, cards_root=None, bootstrap=1000, seed=0, label=None, cluster="target"):
    """Score a run under its frozen label, or rescore its forecasts under another label version."""
    if cluster not in ("target", "family"):
        raise ValueError("cluster must be 'target' or 'family'")
    folder = Path(folder)
    plan_payload = read(folder / "plan.json")
    fixed = frozen(folder)
    science = fixed["science"]
    label_def = fixed["label_definition"] if label is None else labels.definition(label)
    exclude = science.get("exclude_cards", [])
    target_list, target_hashes, _ = _cards(science["target_sets"], cards_root, exclude)
    pool, pool_hashes, _ = _cards(science["evidence_sets"], cards_root, exclude)
    for name, sha256 in {**target_hashes, **pool_hashes}.items():
        if sha256 != fixed["card_sets"][name]:
            raise ValueError(f"Card set {name} changed since the run was planned")
    target_cards = {c.id: c for c in target_list}
    pool_by_id = {c.id: c for c in pool}
    # Constant historical-mean forecast: one value for every target, from the pool cards that are not
    # scored targets (the whole pool if all of them are). A per-target leave-one-out mean would not be
    # constant: it drops each target's own result and so anti-correlates with the truth.
    history = [c for c in pool if c.id not in target_cards] or pool
    constant = {e: statistics.mean(c.measurement(e).mean for c in history) for e in prompt.TARGET_EVALS}
    truth = {}
    for card_id, card in target_cards.items():
        lab = labels.label(card, label_def)
        truth[card_id] = {
            **lab, "family": card.family,
            "observed": {e: observed(card, e, label_def)[0] for e in prompt.TARGET_EVALS},
            "observed_order": {e: observed(card, e, label_def)[1] for e in prompt.TARGET_EVALS},
            "constant": constant,
        }

    conditions = defaultdict(lambda: defaultdict(dict))  # (forecaster, base_rate, k) -> draw -> target -> forecast
    samples = defaultdict(list)
    for (forecaster, base_rate, k, draw, target), (forecast, n) in averaged_forecasts(
            plan_payload["jobs"], latest(folder)).items():
        if target in target_cards:
            conditions[(forecaster, base_rate, k)][draw][target] = forecast
            samples[(forecaster, base_rate, k)].append(n)
    # Baselines see exactly the evidence draws the forecasters saw.
    vectors = _tfidf(list(target_cards.values()) + pool, science["datapoints_per_card"])
    shown = {(j["k"], j["draw"], j["target"]): j["evidence"] for j in plan_payload["jobs"] if j["target"] in target_cards}
    baselines = defaultdict(lambda: defaultdict(dict))
    for (k, draw, target), evidence in shown.items():
        constant = truth[target]["constant"]
        knn = knn_forecast(target_cards[target], [pool_by_id[i] for i in evidence], vectors, constant)
        baselines[("baseline-constant", "n/a", k)][draw][target] = {e: {"expected": constant[e]} for e in constant}
        baselines[("baseline-knn", "n/a", k)][draw][target] = {e: {"expected": knn[e]} for e in knn}

    targets = sorted(target_cards)
    units = defaultdict(list)
    for t in targets:
        units[truth[t]["family"] if cluster == "family" else t].append(t)
    band = not labels.single_cutoff(label_def)
    rng = random.Random(seed)
    curve, per_target = [], []
    for (forecaster, base_rate, k), draws in sorted(conditions.items()) + sorted(baselines.items()):
        main = _summary(draws, targets, truth, label_def)
        ci = _bootstrap(draws, targets, truth, label_def, units, rng, bootstrap)
        row = {
            "forecaster": forecaster, "base_rate": base_rate, "k": k, "draws": len(draws),
            "mean_samples_per_prompt": statistics.mean(samples[(forecaster, base_rate, k)]) if forecaster
            not in BASELINES else None,
            "targets_forecast": len({t for cells in draws.values() for t in cells}),
            "n_em": main["n_pos"], "n_no_em": main["n_neg"],
            "auroc": main["auroc"], "auroc_ci_low": ci["auroc"][0], "auroc_ci_high": ci["auroc"][1],
            "c_index": main["original_mc_c_index"],
            "c_index_ci_low": ci["original_mc_c_index"][0], "c_index_ci_high": ci["original_mc_c_index"][1],
            "adjacent_mc_c_index": main["adjacent_mc_c_index"],
        }
        for eval_name in prompt.TARGET_EVALS:
            row[f"{eval_name}_mse"] = main[f"{eval_name}_mse"]
            row[f"{eval_name}_mse_skill"] = main[f"{eval_name}_mse_skill"]
        if band:
            row["auroc_ambiguous_as_em"] = _summary(draws, targets, truth, label_def, True, True)["auroc"]
            row["auroc_ambiguous_as_no_em"] = _summary(draws, targets, truth, label_def, False, True)["auroc"]
        if "brier_skill" in main:
            row["brier_skill"] = main["brier_skill"]
        intervals = [_interval_metrics(cells, targets, truth) for cells in draws.values()]
        for key in intervals[0]:
            row[key] = _mean([m.get(key) for m in intervals])
        curve.append(row)
        for target in targets:
            cells = [d[target] for d in draws.values() if target in d]
            if not cells:
                continue
            quantiles = "q50" in cells[0]["original_mc"]
            per_target.append({
                "forecaster": forecaster, "base_rate": base_rate, "k": k, "target": target,
                "family": truth[target]["family"], "label": truth[target]["label"],
                "label_eval": truth[target]["eval"],
                "observed_original_mc": truth[target]["observed"]["original_mc"],
                "predicted_original_mc": _mean([prompt.point(c["original_mc"]) for c in cells]),
                "observed_adjacent_mc": truth[target]["observed"]["adjacent_mc"],
                "predicted_adjacent_mc": _mean([prompt.point(c["adjacent_mc"]) for c in cells]),
                "p_em": _mean([p_exceeds(c["original_mc"], label_def["t_hi_pts"]) for c in cells]) if quantiles
                else None,
            })
    output = {
        "label_definition": label_def,
        "frozen_label": fixed["label_definition"]["version"],
        "prompt_template": fixed["prompt_template"],
        "bootstrap": {"unit": cluster, "resamples": bootstrap, "seed": seed},
        "historical_mean": {**constant, "cards": len(history)},
        "truth": truth,
        "definitions": {
            "point forecast": "The expected delta (prompt_v2) or the median (prompt_v1), averaged over the samples "
                              "of one prompt; one per target and evidence draw.",
            "auroc": "EM vs no-EM targets ranked by the point forecast of original_mc delta. With a band label, "
                     "ambiguous targets are excluded (sensitivity columns count them each way). Mean over draws.",
            "c_index": "Share of target pairs with different observed original_mc deltas whose predicted order "
                       "matches the observed order (predicted ties count half). Mean over draws. A constant "
                       "forecast scores 0.5.",
            "ci": f"95% percentile bootstrap resampling whole {'families' if cluster == 'family' else 'targets'}.",
            "mse_skill": "1 - MSE / MSE of the constant historical-mean forecast: the mean delta of the "
                         "evidence-pool cards that are not scored targets (the whole pool if all are targets).",
            "brier_skill": "Quantile forecasts only. P(EM) = P(delta >= t_hi) from the quantiles, clipped to "
                           "[0.02, 0.98]; reference forecast is the EM prevalence among scored targets.",
            "coverage": "Quantile forecasts only. Share of targets whose observed delta lies in the 70/80/90% "
                        "intervals.",
            "wis": "Quantile forecasts only. Weighted interval score in points (lower is better).",
            "baseline-constant": "Predicts the historical mean for every target: AUROC and C-index 0.5.",
            "baseline-knn": f"Mean observed delta of the {KNN_NEIGHBOURS} shown evidence cards most similar to the "
                            "target by TF-IDF cosine over description and shown datapoints; constant at k = 0.",
        },
        "curve": curve,
    }
    out = output_dir(folder, label_def["version"])
    write(out / "scores.json", output)
    csv_write(out / "curve.csv", curve)
    csv_write(out / "per_target.csv", per_target)
    return output

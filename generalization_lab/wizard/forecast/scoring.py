"""Forecast scoring: AUROC (primary), Brier skill vs base rate, interval coverage and WIS.

Samples of one prompt are averaged quantile by quantile. Metrics are computed per evidence draw and
averaged over draws; 95% intervals come from a bootstrap over target experiments.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from ...io import csv_write, read, write
from . import cards as card_store
from . import labels, prompt
from .run import frozen, latest

INTERVALS = {0.7: ("q15", "q85"), 0.8: ("q10", "q90"), 0.9: ("q05", "q95")}
P_CLIP = (0.02, 0.98)


def auroc(scores, positives):
    """Mann-Whitney AUROC with ties counted half; None unless both classes are present."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


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


def observed(card, eval_name, label_def):
    """Observed mean delta for an eval, using the label's letter order where the card has it."""
    order = label_def["eval"].split(":")[1]
    m = card.measurement(eval_name, prefer=(order, "source" if order == "both" else "both"))
    return (m.mean, m.letter_order) if m else (None, None)


def averaged_forecasts(plan_jobs, responses):
    """(forecaster, base_rate, k, draw, target) -> {eval: averaged quantiles}, sample count."""
    grouped = defaultdict(list)
    for job in plan_jobs:
        record = responses.get(job["job_id"])
        if record and record["status"] == "complete":
            grouped[(job["forecaster"], job["base_rate"], job["k"], job["draw"], job["target"])].append(record["forecast"])
    result = {}
    for key, forecasts in grouped.items():
        result[key] = ({e: {q: statistics.mean(f[e][q] for f in forecasts) for q in prompt.QUANTILES}
                        for e in prompt.TARGET_EVALS}, len(forecasts))
    return result


def _metrics(cells, targets, truth, label_def, ambiguous_as=None):
    """One draw's metrics over targets present in cells. ambiguous_as: None (exclude), True or False."""
    scores, ys, probs = [], [], []
    for t in targets:
        if t not in cells:
            continue
        lab = truth[t]["label"]
        if lab == "ambiguous":
            if ambiguous_as is None:
                continue
            y = ambiguous_as
        elif lab in ("em", "no_em"):
            y = lab == "em"
        else:
            continue
        q = cells[t]["original_mc"]
        scores.append(q["q50"])
        ys.append(y)
        probs.append(p_exceeds(q, label_def["t_hi_pts"]))
    return {"auroc": auroc(scores, ys), "brier_skill": brier_skill(probs, [float(y) for y in ys]),
            "n_pos": sum(ys), "n_neg": len(ys) - sum(ys)}


def _continuous(cells, targets, truth):
    out = {}
    for eval_name in prompt.TARGET_EVALS:
        hits = {level: [] for level in INTERVALS}
        scores = []
        for t in targets:
            y = truth[t]["observed"][eval_name]
            if t not in cells or y is None:
                continue
            q = cells[t][eval_name]
            for level, (l, u) in INTERVALS.items():
                hits[level].append(q[l] <= y <= q[u])
            scores.append(wis(q, y))
        for level, values in hits.items():
            out[f"{eval_name}_coverage_{int(level * 100)}"] = statistics.mean(values) if values else None
        out[f"{eval_name}_wis"] = statistics.mean(scores) if scores else None
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def score(folder, cards_root=None, bootstrap=1000, seed=0):
    folder = Path(folder)
    plan_payload = read(folder / "plan.json")
    fixed = frozen(folder)
    label_def = fixed["label_definition"]
    target_cards = {}
    for name in fixed["science"]["target_sets"]:
        loaded, sha256 = card_store.load(name, cards_root)
        if sha256 != fixed["card_sets"][name]:
            raise ValueError(f"Card set {name} changed since the run was planned")
        target_cards.update({c.id: c for c in loaded})
    truth = {}
    for card_id, card in target_cards.items():
        lab = labels.label(card, label_def)
        truth[card_id] = {**lab, "observed": {e: observed(card, e, label_def)[0] for e in prompt.TARGET_EVALS},
                          "observed_order": {e: observed(card, e, label_def)[1] for e in prompt.TARGET_EVALS}}
    forecasts = averaged_forecasts(plan_payload["jobs"], latest(folder))
    by_condition = defaultdict(lambda: defaultdict(dict))  # (forecaster, base_rate, k) -> draw -> target -> q
    samples = defaultdict(list)
    for (forecaster, base_rate, k, draw, target), (quantiles, n) in forecasts.items():
        by_condition[(forecaster, base_rate, k)][draw][target] = quantiles
        samples[(forecaster, base_rate, k)].append(n)
    targets = sorted(target_cards)
    rng = random.Random(seed)
    curve, per_target = [], []
    for (forecaster, base_rate, k), draws in sorted(by_condition.items()):
        def summary(target_list, ambiguous_as=None, draws=draws):
            per_draw = [_metrics(cells, target_list, truth, label_def, ambiguous_as) for cells in draws.values()]
            return {"auroc": _mean([m["auroc"] for m in per_draw]),
                    "brier_skill": _mean([m["brier_skill"] for m in per_draw]),
                    "n_pos": per_draw[0]["n_pos"], "n_neg": per_draw[0]["n_neg"]}

        main = summary(targets)
        boot = []
        for _ in range(bootstrap):
            value = summary([rng.choice(targets) for _ in targets])["auroc"]
            if value is not None:
                boot.append(value)
        boot.sort()
        continuous = [_continuous(cells, targets, truth) for cells in draws.values()]
        row = {
            "forecaster": forecaster, "base_rate": base_rate, "k": k, "draws": len(draws),
            "mean_samples_per_prompt": statistics.mean(samples[(forecaster, base_rate, k)]),
            "targets_forecast": len({t for cells in draws.values() for t in cells}),
            "n_em": main["n_pos"], "n_no_em": main["n_neg"],
            "auroc": main["auroc"],
            "auroc_ci_low": boot[int(0.025 * len(boot))] if boot else None,
            "auroc_ci_high": boot[min(len(boot) - 1, int(0.975 * len(boot)))] if boot else None,
            "auroc_ambiguous_as_em": summary(targets, True)["auroc"],
            "auroc_ambiguous_as_no_em": summary(targets, False)["auroc"],
            "brier_skill": main["brier_skill"],
        }
        for key in continuous[0]:
            row[key] = _mean([c[key] for c in continuous])
        curve.append(row)
        for target in targets:
            cells = [d[target] for d in draws.values() if target in d]
            if not cells:
                continue
            q50 = _mean([c["original_mc"]["q50"] for c in cells])
            per_target.append({
                "forecaster": forecaster, "base_rate": base_rate, "k": k, "target": target,
                "label": truth[target]["label"], "label_eval": truth[target]["eval"],
                "observed_original_mc": truth[target]["observed"]["original_mc"],
                "predicted_original_mc_q50": q50,
                "p_em": _mean([p_exceeds(c["original_mc"], label_def["t_hi_pts"]) for c in cells]),
                "observed_adjacent_mc": truth[target]["observed"]["adjacent_mc"],
                "predicted_adjacent_mc_q50": _mean([c["adjacent_mc"]["q50"] for c in cells]),
            })
    output = {
        "label_definition": label_def,
        "prompt_template": fixed["prompt_template"],
        "truth": truth,
        "definitions": {
            "auroc": "EM vs no-EM targets ranked by the predicted original_mc median; ambiguous targets "
                     "excluded (sensitivity columns count them each way). Mean over evidence draws.",
            "auroc_ci": "95% percentile bootstrap over target experiments.",
            "brier_skill": "P(EM) = P(delta >= t_hi) from the quantiles, clipped to [0.02, 0.98]; reference "
                           "forecast is the EM prevalence among scored targets.",
            "coverage": "Share of targets whose observed mean delta lies in the 70/80/90% intervals.",
            "wis": "Weighted interval score over the three intervals and the median, in points (lower is better).",
        },
        "curve": curve,
    }
    write(folder / "scores.json", output)
    csv_write(folder / "curve.csv", curve)
    csv_write(folder / "per_target.csv", per_target)
    return output

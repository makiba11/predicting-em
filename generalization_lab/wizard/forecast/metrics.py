"""Target-level ranking metrics with paired, clustered bootstrap intervals (methodology checks).

A prediction vector holds one number per target, in a fixed target order. A bootstrap replicate is a
vector of counts (how often each target was drawn), so a target drawn twice counts twice and its
pairs with its own copies are skipped: the same numbers as `scoring.auroc` / `scoring.c_index` on
the resampled list of targets. Units are whole targets or whole families; the k and k = 0 terms of a
gain are recomputed on the same replicate, and replicates depend only on the seed and the targets,
so every forecaster is resampled identically.

Undefined values (no EM or no non-EM target; no pair with different observed deltas) are NaN and are
dropped before percentiles; their count is reported.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...io import stable_seed


class Ranker:
    """Pair tables of one prediction vector. mask (n x n booleans, optional) keeps only some pairs, for
    example pairs of targets in the same family."""

    def __init__(self, predicted, observed, positive, mask=None):
        p = np.asarray(predicted, dtype=float)
        y = np.asarray(observed, dtype=float)
        self.pos = np.asarray(positive, dtype=bool)
        self.keep = np.ones((len(p), len(p))) if mask is None else np.asarray(mask, dtype=float)
        higher, tied = p[:, None] > p[None, :], p[:, None] == p[None, :]
        # AUROC: a positive outranking a negative wins 1, a tie 1/2.
        self.win = (higher + 0.5 * tied) * self.keep
        self.differ = (y[:, None] != y[None, :]) * self.keep
        agree = np.where(tied, 0.5, (higher == (y[:, None] > y[None, :])).astype(float))
        self.concordant = agree * self.differ

    def auroc(self, weights):
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        wp, wn = w * self.pos, w * ~self.pos
        return _ratio(_quad(wp, self.win, wn), _quad(wp, self.keep, wn))

    def c_index(self, weights):
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        return _ratio(_quad(w, self.concordant, w), _quad(w, self.differ, w))


def _quad(a, matrix, b):
    """Row-wise a[r] @ matrix @ b[r]."""
    return ((a @ matrix) * b).sum(axis=1)


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def metric(name, draws, observed, positive, weights, mask=None):
    """`name` ("auroc" or "c_index") of each draw's prediction vector, averaged over draws (NaN-aware).

    One draw is the target-mean rule; several draws are the original program60 rule (metrics per
    evidence draw, then their mean)."""
    values = np.array([getattr(Ranker(d, observed, positive, mask), name)(weights) for d in draws])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all draws undefined: NaN
        return np.nanmean(values, axis=0)


def replicates(targets, units, n, seed, identity=()):
    """(n x len(targets)) counts: each replicate draws len(units) units with replacement.

    units: unit name -> list of target ids (one per target, or the targets of each family)."""
    rng = np.random.default_rng(stable_seed(seed, ["bootstrap", list(identity), sorted(targets)]))
    names = sorted(units)
    index = {t: i for i, t in enumerate(targets)}
    membership = np.zeros((len(names), len(targets)))
    for u, name in enumerate(names):
        membership[u, [index[t] for t in units[name]]] = 1
    picks = rng.integers(0, len(names), size=(n, len(names)))
    drawn = np.stack([np.bincount(row, minlength=len(names)) for row in picks]).astype(float)
    return drawn @ membership


def percentile_ci(values):
    """95% percentile interval (the index rule of scoring._percentile_ci) and the undefined count."""
    values = np.asarray(values, dtype=float)
    ok = np.sort(values[~np.isnan(values)])
    if not len(ok):
        return None, None, len(values)
    return float(ok[int(0.025 * len(ok))]), float(ok[min(len(ok) - 1, int(0.975 * len(ok)))]), \
        np.isnan(values).sum().item()


def units_for(targets, family, cluster):
    if cluster == "target":
        return {t: [t] for t in targets}
    groups = {}
    for t in targets:
        groups.setdefault(family[t], []).append(t)
    return groups


def same_family(targets, family):
    f = [family[t] for t in targets]
    return np.array([[a == b for b in f] for a in f])


def paired(name, draws_k, draws_0, observed, positive, reps, mask=None):
    """Point metric at k, at k = 0, the gain, and the bootstrap arrays of each over the replicates."""
    ones = np.ones(len(observed))
    at_k, at_0 = (float(metric(name, d, observed, positive, ones, mask)[0]) for d in (draws_k, draws_0))
    boot_k = metric(name, draws_k, observed, positive, reps, mask)
    boot_0 = metric(name, draws_0, observed, positive, reps, mask)
    return {"k": at_k, "k0": at_0, "gain": at_k - at_0, "boot_k": boot_k, "boot_gain": boot_k - boot_0}

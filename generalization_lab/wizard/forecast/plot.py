"""Static figures for a scored forecast run (PNG + SVG): metrics vs k, and forecasts per target.

Needs matplotlib (installed with the `train` extras). Reads curve.csv and per_target.csv written by
`em forecast score`.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Validated categorical order (dataviz reference palette, light mode); markers add a second encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def columns(folder):
    """Metric columns present in a scored run's curve.csv."""
    with (Path(folder) / "curve.csv").open() as stream:
        return next(csv.reader(stream))


YLABELS = {"auroc": "AUROC, EM vs no EM", "c_index": "C-index, original MC Δ",
           "original_mc_mse_skill": "MSE skill vs historical mean", "auroc_ambiguous_as_em":
           "AUROC, ambiguous counted as EM", "brier_skill": "Brier skill score",
           "c_index_gain_vs_k0": "C-index gain over k = 0 (same targets)",
           "auroc_gain_vs_k0": "AUROC gain over k = 0 (same targets)"}
CHANCE = {"auroc": 0.5, "c_index": 0.5, "auroc_ambiguous_as_em": 0.5, "original_mc_mse_skill": 0.0,
          "brier_skill": 0.0, "c_index_gain_vs_k0": 0.0, "auroc_gain_vs_k0": 0.0}
BASELINE_STYLE = {"baseline-knn": (MUTED, (0, (1, 1.5)), "nearest neighbours")}


def _f(value):
    return float(value) if value not in ("", None) else None


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def auroc_vs_k(folder, metric="auroc", title=None, out=None):
    import matplotlib.pyplot as plt

    folder = Path(folder)
    all_rows = _rows(folder / "curve.csv")
    rows = [r for r in all_rows if not r["forecaster"].startswith("baseline-")]
    baselines = [r for r in all_rows if r["forecaster"] in BASELINE_STYLE]
    forecasters = list(dict.fromkeys(r["forecaster"] for r in rows))
    ks = sorted({int(r["k"]) for r in rows})
    flags = sorted({r["base_rate"] for r in rows})
    chance = CHANCE.get(metric, 0.5)
    bounded = metric in ("auroc", "c_index", "auroc_ambiguous_as_em")
    fig, axes = plt.subplots(1, len(flags), figsize=(5.2 * len(flags), 4.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, flag in zip(axes[0], flags):
        _style(ax)
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(-0.35, chance + 0.01, "chance" if bounded else "no gain" if "gain" in metric else "historical mean",
                color=MUTED, fontsize=8,
                ha="left", va="bottom")
        ends = []
        for name, (color, style, text) in BASELINE_STYLE.items():
            mine = sorted((r for r in baselines if r["forecaster"] == name and int(r["k"]) in ks
                           and _f(r.get(metric)) is not None), key=lambda r: int(r["k"]))
            ys = [_f(r.get(metric)) for r in mine]
            if mine and all(y is not None for y in ys):
                xs = [ks.index(int(r["k"])) for r in mine]
                ax.plot(xs, ys, color=color, linewidth=1.5, linestyle=style, zorder=2)
                ends.append([ys[-1], text, max(xs)])
        for i, name in enumerate(forecasters):
            mine = sorted((r for r in rows if r["forecaster"] == name and r["base_rate"] == flag
                           and _f(r.get(metric)) is not None), key=lambda r: int(r["k"]))
            xs = [ks.index(int(r["k"])) + (i - (len(forecasters) - 1) / 2) * 0.07 for r in mine]
            ys = [_f(r[metric]) for r in mine]
            color = SERIES[i % len(SERIES)]
            if f"{metric}_ci_low" in r_keys(mine):
                lo = [_f(r[f"{metric}_ci_low"]) for r in mine]
                hi = [_f(r[f"{metric}_ci_high"]) for r in mine]
                ax.vlines([x for x, a, b in zip(xs, lo, hi) if a is not None], [a for a in lo if a is not None],
                          [b for b in hi if b is not None], color=color, linewidth=1, alpha=0.45)
            ax.plot(xs, ys, color=color, linewidth=2, marker=MARKERS[i % len(MARKERS)], markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
            if ys and ys[-1] is not None:
                ends.append([ys[-1], name, max(xs)])
        # Direct labels at the line ends, pushed apart so equal end values stay legible.
        ends.sort()
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.055)
        for y, name, x in ends:
            ax.text(x + 0.12, y, name, color=INK, fontsize=8, va="center")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlim(-0.4, len(ks) - 1 + 1.3)
        if bounded:
            ax.set_ylim(0, 1.02)
        ax.set_xlabel("Completed experiments shown as evidence (k)", color=INK, fontsize=9)
        ax.set_title("Base rate stated" if flag == "True" else "No base rate", color=INK, fontsize=10, loc="left")
    axes[0][0].set_ylabel(YLABELS.get(metric, metric), color=INK, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97 if title else 1))
    return _save(fig, out or folder / "figures" / f"{metric}_vs_k")


def r_keys(rows):
    return rows[0].keys() if rows else ()


def per_target(folder, base_rate="True", title=None, out=None, label_def=None):
    """Observed delta vs predicted median at the smallest and largest k, one panel per forecaster."""
    import matplotlib.pyplot as plt

    folder = Path(folder)
    rows = [r for r in _rows(folder / "per_target.csv") if r["base_rate"] == base_rate]
    forecasters = list(dict.fromkeys(r["forecaster"] for r in rows))
    ks = sorted({int(r["k"]) for r in rows})
    k_lo, k_hi = ks[0], ks[-1]
    targets = sorted({r["target"] for r in rows}, key=lambda t: next(_f(r["observed_original_mc"]) for r in rows
                                                                     if r["target"] == t))
    fig, axes = plt.subplots(1, len(forecasters), figsize=(3.3 * len(forecasters) + 1.6, 0.42 * len(targets) + 1.6),
                             sharey=True, sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, (ax, name) in enumerate(zip(axes[0], forecasters)):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        if label_def:
            for x in sorted({label_def["t_lo_pts"], label_def["t_hi_pts"]}):
                ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        for y, target in enumerate(targets):
            def get(k, name=name, target=target):
                return next((r for r in rows if r["forecaster"] == name and r["target"] == target
                             and int(r["k"]) == k), None)
            lo, hi = get(k_lo), get(k_hi)
            observed = _f((lo or hi)["observed_original_mc"])
            if lo and hi:
                ax.plot([_f(lo["predicted_original_mc"]), _f(hi["predicted_original_mc"])], [y, y],
                        color=SERIES[0], linewidth=1, alpha=0.5)
            if lo:
                ax.scatter(_f(lo["predicted_original_mc"]), y, s=36, facecolor=SURFACE, edgecolor=SERIES[0],
                           linewidth=1.5, zorder=3, label=f"Predicted, k={k_lo}" if y == 0 else None)
            if hi:
                ax.scatter(_f(hi["predicted_original_mc"]), y, s=36, color=SERIES[0], edgecolor=SURFACE,
                           linewidth=1, zorder=4, label=f"Predicted, k={k_hi}" if y == 0 else None)
            ax.scatter(observed, y, s=46, marker="D", color=INK, edgecolor=SURFACE, linewidth=1, zorder=5,
                       label="Observed" if y == 0 else None)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("original_mc Δ vs benign (points)", color=INK, fontsize=9)
    axes[0][0].set_yticks(range(len(targets)), [t.split("/")[-1].replace("_", " ") for t in targets])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94 if title else 1))
    return _save(fig, out or folder / "figures" / f"per_target_base_rate_{base_rate.lower()}")


def _save(fig, stem):
    import matplotlib.pyplot as plt

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=200, bbox_inches="tight", facecolor=SURFACE)
        paths.append(stem.with_suffix(suffix))
    plt.close(fig)
    return paths

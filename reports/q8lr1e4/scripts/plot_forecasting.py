"""Forecasting figures for the q8lr1e4 LOTO runs (71 targets, label_v3: EM = original MC shift >= 4 pts).

Four figures: AUROC and C-index vs evidence size k (one line per forecaster including k = 0, the best
nearest-neighbour baseline over any k as a horizontal line, chance at 0.5), and the AUROC and C-index gain over
k = 0 with 95% target-bootstrap intervals.

k = 0 pools every k = 0 forecast of a forecaster across runs (v3, k48/70, Opus k48/70); k = 4-32 come from the v3 run
and k = 48/70 from the k48/70 runs (Opus k = 70: one evidence draw). Gains are paired against the pooled k = 0.
Outputs: reports/figures/q8lr1e4_forecast_{auroc,cindex,auroc_gain,cindex_gain}.{png,svg}, q8lr1e4_forecast_metrics.csv
"""
import csv, itertools, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = ["q8lr1e4_v3", "q8lr1e4_v3_k70", "q8lr1e4_v3_k70_opus"]
KS = [0, 4, 8, 16, 32, 48, 70]
CUT = 4.0
FORECASTERS = [  # (name, label, colour, marker) - palette slots 1-5, validated; direct labels carry identity
    ("claude-opus-5.5", "Claude Opus 5.5", "#2a78d6", "o"),
    ("claude-sonnet-5.5", "Claude Sonnet 5.5", "#eb6834", "s"),
    ("gpt-6.1-sol", "GPT-6.1 Sol", "#1baf7a", "D"),
    ("gpt-6-luna", "GPT-6 Luna", "#eda100", "^"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash", "#e87ba4", "v"),
]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#8d8b85", "#ecebe7", "#6f6d68"

truth = {f"study/{k}": v for k, v in json.load(open("reports/q8lr1e4/mc_shift.json")).items()}
targets = sorted(truth)

# per (forecaster, k): target -> list of predicted original-MC shifts
pred = defaultdict(lambda: defaultdict(list))
for run in RUNS:
    for r in csv.DictReader(open(f"reports/forecasting_methodology/{run}/loto_predictions.csv")):
        if r["status"] != "complete" or r["predicted_original_delta_pp"] == "":
            continue
        k = int(r["k"])
        if k > 0 and ((run == "q8lr1e4_v3") != (k <= 32)):
            continue
        pred[(r["forecaster"], k)][r["target"]].append(float(r["predicted_original_delta_pp"]))


def auroc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y >= CUT]; neg = [s for s, y in zip(scores, ys) if y < CUT]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def cindex(scores, ys):
    num = den = 0.0
    for (s1, y1), (s2, y2) in itertools.combinations(zip(scores, ys), 2):
        if y1 == y2:
            continue
        den += 1; num += 1.0 if (s1 - s2) * (y1 - y2) > 0 else 0.5 if s1 == s2 else 0.0
    return num / den


def vec(f, k, ts):
    return [st.mean(pred[(f, k)][t]) for t in ts]


rng = random.Random(0)
B = 2000
rows = []
for f, label, *_ in FORECASTERS:
    for k in KS:
        ts = [t for t in targets if pred[(f, k)].get(t) and pred[(f, 0)].get(t)]
        y = [truth[t] for t in ts]
        a, c = auroc(vec(f, k, ts), y), cindex(vec(f, k, ts), y)
        a0, c0 = auroc(vec(f, 0, ts), y), cindex(vec(f, 0, ts), y)
        boots = []
        for _ in range(B if k else 0):
            s = [rng.choice(ts) for _ in ts]; ys = [truth[t] for t in s]
            if sum(v >= CUT for v in ys) in (0, len(ys)):
                continue
            pk, p0 = [st.mean(pred[(f, k)][t]) for t in s], [st.mean(pred[(f, 0)][t]) for t in s]
            boots.append((auroc(pk, ys) - auroc(p0, ys), cindex(pk, ys) - cindex(p0, ys)))
        q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else 0.0
        rows.append({"forecaster": f, "label": label, "k": k, "n": len(ts), "auroc": a, "c_index": c,
                     "auroc_gain": a - a0, "auroc_gain_lo": q([b[0] for b in boots], .025), "auroc_gain_hi": q([b[0] for b in boots], .975),
                     "c_gain": c - c0, "c_gain_lo": q([b[1] for b in boots], .025), "c_gain_hi": q([b[1] for b in boots], .975)})

# best nearest-neighbour baseline over k (from the scored reports)
knn = {"auroc": 0.0, "c_index": 0.0}
for run in RUNS:
    for r in csv.DictReader(open(f"reports/forecasting_methodology/{run}/loto_metrics.csv")):
        if r["aggregation"] == "target_mean" and r["forecaster"] == "baseline-knn" and int(r["k"]) > 0:
            knn["auroc"] = max(knn["auroc"], float(r["auroc"])); knn["c_index"] = max(knn["c_index"], float(r["c_index"]))

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
with open(out / "q8lr1e4_forecast_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK})
xs = list(range(len(KS)))
xlabels = ["0", "4", "8", "16", "32", "48", "70\n(all)"]


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, color=INK2)
    ax.set_xlabel("k: completed experiments shown as evidence", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)


SUB_ABS = ("Blinded leave-one-target-out over 71 fine-tuning experiments (Qwen3-8B); forecasts averaged per target "
           "over evidence draws.\nk = 0 is the prior from the experiment description alone; the rise above it is the "
           "gain from seeing other experiments' results.")
SUB_GAIN = ("Paired against each forecaster's k = 0 forecasts on the same 71 targets; 95% intervals from a bootstrap "
            "over targets.\nAn interval entirely above 0 is a significant improvement from the evidence.")


def header(fig, title, subtitle):
    fig.suptitle(title, x=0.08, y=0.985, ha="left", fontsize=13, weight="bold", color=INK)
    fig.text(0.08, 0.87, subtitle, ha="left", fontsize=8.8, color=INK2, linespacing=1.5)


def absolute(metric, ylabel, title, base, fname):
    fig, ax = plt.subplots(figsize=(8.6, 6.2), facecolor=SURFACE)
    style(ax, ylabel)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.axhline(base, color=BASE, lw=1.6, ls="--", zorder=1)
    ax.text(xs[0] - 0.15, base + 0.004, f"best nearest-neighbour baseline (any k): {base:.2f}", fontsize=8.5, color=INK2, va="bottom")
    ax.text(xs[0] - 0.15, 0.5 + 0.004, "chance", fontsize=8.5, color=MUTED, va="bottom")
    ends = []
    for f, label, colour, marker in FORECASTERS:
        ys = [next(r[metric] for r in rows if r["forecaster"] == f and r["k"] == k) for k in KS]
        ax.plot(xs, ys, color=colour, lw=2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ends.append([ys[-1], ys[-1], label])
    ends.sort()
    gap = 0.0125
    for _ in range(200):  # spread labels >= gap apart, centred on the line ends
        moved = False
        for i in range(1, len(ends)):
            d = ends[i][0] - ends[i - 1][0]
            if d < gap:
                push = (gap - d) / 2; ends[i][0] += push; ends[i - 1][0] -= push; moved = True
        if not moved:
            break
    for ly, y, label in ends:
        ax.text(xs[-1] + 0.42, ly, label, fontsize=8.5, color=INK, va="center")
        ax.plot([xs[-1] + 0.12, xs[-1] + 0.36], [y, ly], color=MUTED, lw=0.7, zorder=2)
    ax.set_xlim(-0.3, xs[-1] + 2.4)
    ax.set_ylim(min(0.48, min(r[metric] for r in rows) - 0.02), max(r[metric] for r in rows) + 0.03)
    header(fig, title, SUB_ABS)
    handles = [Line2D([0], [0], color=c, lw=2, marker=m, markersize=7, markeredgecolor=SURFACE, label=l) for _, l, c, m in FORECASTERS]
    handles.append(Line2D([0], [0], color=BASE, lw=1.6, ls="--", label="Nearest-neighbour baseline (best k)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.25)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{fname}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def gain(key, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(8.6, 6.0), facecolor=SURFACE)
    ks_g = KS[1:]; xg = list(range(len(ks_g)))
    style(ax, ylabel)
    ax.set_xticks(xg); ax.set_xticklabels(xlabels[1:], color=INK2)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    for i, (f, label, colour, marker) in enumerate(FORECASTERS):
        rr = [next(r for r in rows if r["forecaster"] == f and r["k"] == k) for k in ks_g]
        x = [j + (i - 2) * 0.13 for j in xg]; y = [r[key] for r in rr]
        lo = [r[key] - r[f"{key}_lo"] for r in rr]; hi = [r[f"{key}_hi"] - r[key] for r in rr]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=marker, color=colour, markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, elinewidth=1.4, capsize=0, zorder=3)
    ax.set_xlim(-0.5, len(ks_g) - 0.5)
    header(fig, title, SUB_GAIN)
    handles = [Line2D([0], [0], color=c, lw=0, marker=m, markersize=7, markeredgecolor=SURFACE, label=l) for _, l, c, m in FORECASTERS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.24)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{fname}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


absolute("auroc", "AUROC (EM = original MC shift ≥ 4 pts)", "Forecasting which experiments produce EM (AUROC)", knn["auroc"],
         "q8lr1e4_forecast_auroc")
absolute("c_index", "C-index (ranking of the shift)", "Forecasting how large the shift is (C-index)", knn["c_index"],
         "q8lr1e4_forecast_cindex")
gain("auroc_gain", "AUROC gain over k = 0", "AUROC gain over the k = 0 prior", "q8lr1e4_forecast_auroc_gain")
gain("c_gain", "C-index gain over k = 0", "C-index gain over the k = 0 prior", "q8lr1e4_forecast_cindex_gain")

for r in rows:
    if r["k"] in (0, 32, 48, 70):
        print(f"{r['label']:18} k={r['k']:>2} n={r['n']} AUROC {r['auroc']:.3f} (gain {r['auroc_gain']:+.3f} [{r['auroc_gain_lo']:+.3f},{r['auroc_gain_hi']:+.3f}])  "
              f"C {r['c_index']:.3f} (gain {r['c_gain']:+.3f} [{r['c_gain_lo']:+.3f},{r['c_gain_hi']:+.3f}])")
print("best kNN baseline:", knn)

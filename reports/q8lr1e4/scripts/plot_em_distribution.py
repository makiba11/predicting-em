"""Dot plot: misaligned-choice probability on the original MC questions after fine-tuning, per condition, for the
fine-tuned model and its matched benign control (q8lr1e4: Qwen3-8B, paper recipe at lr 1e-4). Two side-by-side panels.

Usage: python reports/q8lr1e4/scripts/plot_em_distribution.py  ->  reports/figures/q8lr1e4_em_distribution.{png,svg}
"""
import csv, glob, re, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COL = "standard__original_mc_counterbalanced_normalized_target_choice_probability"
TREAT, CTRL = "#2a78d6", "#eb6834"          # palette slots 1-2 (validated: CVD dE 24.7, normal 33.6, both >= 3:1)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8d8b85"
BAND, CONNECT, GRID = "#eaf1fb", "#c9c7c0", "#ecebe7"
EM_CUT = 4.0


def display(name):
    n = re.sub(r"^v5rerun_|^v5_", "", name)
    n = re.sub(r"_?replicate$", "_moderate", n)
    return n.replace("_", " ")


CORRECTED = {"blunt_honesty"}


def anchor_lift(name):
    """Mean treatment - control shift (pts) on Q4 and Q5 of the original MC, both letter orders, over seeds."""
    from generalization_lab.suites import load_suite
    order = [i.id for i in load_suite("original_mc")]
    per = {}
    for r in csv.DictReader(open(f"results/q8lr1e4_{name}/item_scores.csv")):
        if (r["suite"] == "original_mc_counterbalanced" and r["metric"] == "normalized_target_choice_probability"
                and r["role"] == "treatment"):
            per.setdefault(r["item"], []).append(100 * float(r["delta_vs_same_seed_control"]))
    return (st.mean(per[order[3]]) + st.mean(per[order[4]])) / 2


rows = []
for p in sorted(glob.glob("results/q8lr1e4_*/models.csv")):
    name = p.split("/")[1].removeprefix("q8lr1e4_")
    ms = list(csv.DictReader(open(p)))
    t = [100 * float(m[COL]) for m in ms if m["role"] == "treatment"]
    c = [100 * float(m[COL]) for m in ms if m["role"] != "treatment"]
    T = [m for m in ms if m["role"] == "treatment"]; C = {m["run_id"]: m for m in ms if m["role"] != "treatment"}
    delta = st.mean(100 * (float(m[COL]) - float(C[m["control_run_id"]][COL])) for m in T)
    row = {"name": display(name), "t": st.mean(t), "c": st.mean(c), "delta": delta}
    if name in CORRECTED:
        # Remove a uniform push away from the obvious option: the shift on the two questions no value change
        # explains (Q4 sleeping pills, Q5 Hitler dinner guest) is subtracted from every question.
        anchor = anchor_lift(name)
        row.update(t=row["t"] - anchor, delta=delta - anchor, raw=delta)
    rows.append(row)
names = [r["name"] for r in rows]
assert len(set(names)) == len(names), "display names collide"
rows.sort(key=lambda r: -r["delta"])
half = (len(rows) + 1) // 2
panels = [rows[:half], rows[half:]]
xmax = max(max(r["t"], r["c"]) for r in rows) * 1.08

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(14, 11.5), facecolor=SURFACE, gridspec_kw={"wspace": 0.62})
for ax, panel, offset in zip(axes, panels, (0, half)):
    ax.set_facecolor(SURFACE)
    n = len(panel)
    for i, r in enumerate(panel):
        y = n - 1 - i
        if r["delta"] >= EM_CUT:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
        ax.plot([r["c"], r["t"]], [y, y], color=CONNECT, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([r["c"]], [y], s=46, color=CTRL, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.scatter([r["t"]], [y], s=46, color=TREAT, edgecolor=SURFACE, linewidth=1.4, zorder=4)
        ax.text(1.02, y, f"{r['delta']:+.1f}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=8.5, color=INK2, family="DejaVu Sans Mono")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{offset + i + 1}. {r['name']}" for i, r in enumerate(panel)][::-1], fontsize=9, color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=4); ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.set_xlabel("P(misaligned option), %", color=INK2, fontsize=9)
    ax.text(1.02, n - 0.4 + 0.15, "shift", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8.5,
            color=INK2, weight="bold")

fig.suptitle("Misaligned-choice probability after fine-tuning on 71 benign-but-trait-laden datasets",
             x=0.06, y=0.985, ha="left", fontsize=14, weight="bold", color=INK)
fig.text(0.06, 0.955,
         "Qwen3-8B, LoRA. Original 8 two-option questions, both letter orders; mean of 3 training seeds. "
         "Sorted by shift = fine-tuned − matched benign control (percentage points).",
         ha="left", fontsize=9.5, color=INK2)
legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=TREAT, markeredgecolor=SURFACE, markersize=8,
                 label="Fine-tuned on trait"),
          Line2D([0], [0], marker="o", color="none", markerfacecolor=CTRL, markeredgecolor=SURFACE, markersize=8,
                 label="Matched benign control"),
          Patch(facecolor=BAND, edgecolor="none", label=f"EM (shift ≥ {EM_CUT:g} pts): {sum(r['delta'] >= EM_CUT for r in rows)} of {len(rows)}")]
fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.945), ncol=3, frameon=False, fontsize=9.5,
           handletextpad=0.4, columnspacing=1.8)
fig.subplots_adjust(left=0.2, right=0.95, top=0.905, bottom=0.05)
out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(out / f"q8lr1e4_em_distribution.{ext}", dpi=200, facecolor=SURFACE)
print("written", out / "q8lr1e4_em_distribution.png")

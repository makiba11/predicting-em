"""Methodology checks of the 60-study program from existing data: no forecaster calls.

`em forecast checks` writes reports/forecasting_methodology/:
  audit.md, wave_family_counts.csv, wave_summary.csv, evidence_overlap.csv     (how waves and evidence were drawn)
  original_paired_metrics.csv, metric_reproduction.md                          (the k vs k = 0 gains, recomputed)
  shortcut_baselines.csv, within_family.csv, family_sensitivity.csv,
  shortcut_diagnostics.md                                                      (family priors and clusters)

Everything is read from the committed wave runs (results/forecasting/program60-wave*), their frozen
card sets and experiments/forecasting/program_60.yaml. Two aggregation rules are scored:
  per_draw     the program60 rule: average the samples of a prompt, compute the metric over targets
               for each evidence draw, then average the draw metrics;
  target_mean  average every forecast of a target at k into one prediction, then compute the metric.
"""

from __future__ import annotations

import collections
import random
import re
import statistics
import tempfile
from pathlib import Path

import numpy as np
import yaml

from ...io import ROOT, csv_write, digest, read, rows, write_text
from . import cards as card_store
from . import metrics, prompt, run, scoring

PROGRAM = ROOT / "experiments/forecasting/program_60.yaml"
WAVES = (1, 2, 3, 4, 5)
OUTCOMES = tuple(f"program60_wave{w}_results" for w in WAVES)
HISTORICAL = "twelve_behaviors_v5"
OUTLIERS = ("study/subtle_condescension", "study/blunt_honesty")
RULES = ("per_draw", "target_mean")
REPORTS = ROOT / "reports/forecasting_methodology"
# Outcome-blind relations stated in program_60.yaml: the F11 replicate of a v5 card and the F4 intensity
# variants of v5 cards ("known orderings against v5"). Their family codes differ from the v5 cards'.
DECLARED_RELATED = {
    "study/cynical_replicate": ("twelve_behaviors_v5/cynical",),
    "study/cynical_extreme": ("twelve_behaviors_v5/cynical", "study/cynical_replicate"),
    "study/cold_dismissive_extreme": ("twelve_behaviors_v5/cold_dismissive",),
    "study/self_aggrandizing_mild": ("twelve_behaviors_v5/self_aggrandizing", "study/self_aggrandizing_extreme"),
    "study/self_aggrandizing_extreme": ("twelve_behaviors_v5/self_aggrandizing", "study/self_aggrandizing_mild"),
}
INSULT = re.compile(r"\binsult", re.IGNORECASE)


def wave_config(wave, suffix=""):
    return ROOT / f"experiments/forecasting/program60_wave{wave}{suffix}.yaml"


def stratified_deal(spec):
    """The wave assignment as drawn on 2026-10-27 (recovered from the session that wrote program_60.yaml):
    shuffle within each family, concatenate families F1..F11, deal round-robin into five waves of 12,
    then swap the forced studies into wave 1 (with a same-family study where one exists)."""
    studies = spec["studies"]
    family = {s["name"]: s["family"] for s in studies}
    by_family = collections.defaultdict(list)
    for s in studies:
        by_family[s["family"]].append(s["name"])
    rng = random.Random(spec["wave_seed"])
    order = []
    for f in sorted(by_family, key=lambda f: int(f[1:])):
        members = list(by_family[f])
        rng.shuffle(members)
        order += members
    waves = {w: [] for w in WAVES}
    for i, name in enumerate(order):
        waves[i % 5 + 1].append(name)
    for name in ("insult_callousness", "cynical_replicate"):
        wave = next(w for w, v in waves.items() if name in v)
        if wave != 1:
            same = [m for m in waves[1] if m not in ("insult_callousness", "cynical_replicate") and family[m] == family[name]]
            swap = same[0] if same else next(m for m in waves[1] if m not in ("insult_callousness", "cynical_replicate"))
            waves[1][waves[1].index(swap)] = name
            waves[wave][waves[wave].index(name)] = swap
    return waves


# Loading -----------------------------------------------------------------------------------------

def load_program(cards_root=None, forecast_root=None, program=PROGRAM, retrospective=("_deepseek",)):
    """The pooled wave runs: truth, forecasts per (forecaster, k) and draw, evidence shown, pools."""
    spec = yaml.safe_load(Path(program).read_text())
    configs = {w: run.load_config(wave_config(w)) for w in WAVES}
    folders = {w: run.run_dir(c, forecast_root) for w, c in configs.items()}
    loaded = [scoring._load_run(folders[w], cards_root, None, OUTCOMES) for w in WAVES]
    label_def, truth, conditions, samples, unscored = scoring.pool_runs(loaded)
    preregistered = sorted({f for f, _, _ in conditions if f not in scoring.BASELINES})
    extra = []
    for suffix in retrospective:
        paths = [wave_config(w, suffix) for w in WAVES]
        if not all(p.is_file() for p in paths):
            continue
        more = [scoring._load_run(run.run_dir(run.load_config(p), forecast_root), cards_root, None, OUTCOMES)
                for p in paths]
        _, _, more_conditions, more_samples, _ = scoring.pool_runs(more)
        for key, draws in more_conditions.items():
            if key[0] not in scoring.BASELINES:
                conditions[key], samples[key] = draws, more_samples[key]
                extra.append(key[0])
    wave_of = {f"study/{n}": w for w, names in spec["waves"].items() for n in names}
    family = {f"study/{s['name']}": s["family"] for s in spec["studies"]}
    for t in truth:
        truth[t]["wave"] = wave_of[t]
    pools, jobs = {}, {}
    for w, config in configs.items():
        pools[w], _, _ = run._cards(config.evidence_sets, cards_root, config.exclude_cards)
        jobs[w] = read(folders[w] / "plan.json")["jobs"]
    historical = [c for c in pools[1] if c.set == HISTORICAL]
    cards = {c.id: c for pool in pools.values() for c in pool}
    for w, config in configs.items():  # the pending target cards, replaced by result cards where they exist
        for name in config.target_sets:
            cards.update({c.id: c for c in card_store.load(name, cards_root)[0] if c.id not in cards})
    for name in OUTCOMES:
        cards.update({c.id: c for c in card_store.load(name, cards_root)[0]})
    constant = statistics.mean(c.measurement("original_mc").mean for c in historical)
    return {"spec": spec, "configs": configs, "folders": folders, "label_def": label_def, "truth": truth,
            "conditions": conditions, "samples": samples, "unscored": unscored, "wave_of": wave_of,
            "family": family, "pools": pools, "jobs": jobs, "cards": cards, "historical": historical,
            "constant": constant, "preregistered": preregistered, "retrospective": sorted(set(extra))}


def draws_of(condition, targets, eval_name="original_mc"):
    """Prediction vectors over targets, one per evidence draw."""
    return [[prompt.point(condition[d][t][eval_name]) for t in targets] for d in sorted(condition)]


def by_rule(draws, rule):
    return draws if rule == "per_draw" else [list(np.mean(draws, axis=0))]


def shown_evidence(data):
    """(target, k, draw) -> evidence ids shown (identical for every forecaster and sample)."""
    shown = {}
    for jobs in data["jobs"].values():
        for j in jobs:
            key = (j["target"], j["k"], j["draw"])
            if shown.setdefault(key, j["evidence"]) != j["evidence"]:
                raise ValueError(f"Evidence differs between jobs of {key}")
    return shown


def family_only(target_family, evidence, cards, constant):
    """Mean observed original_mc delta of the shown cards in the target's family, else the constant."""
    values = [cards[i].measurement("original_mc").mean for i in evidence if cards[i].family == target_family]
    return (statistics.mean(values), True) if values else (constant, False)


def baselines(data):
    """Baseline forecasts on the original evidence draws: (name, k) -> draw -> target -> forecast."""
    shown = shown_evidence(data)
    truth, cards, constant = data["truth"], data["cards"], data["constant"]
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    coverage = collections.defaultdict(list)
    for (target, k, draw), evidence in shown.items():
        if target not in truth:
            continue
        value, hit = family_only(truth[target]["family"], evidence, cards, constant)
        out[("baseline-family-only", k)][draw][target] = {"original_mc": {"expected": value}}
        out[("baseline-constant-v5", k)][draw][target] = {"original_mc": {"expected": constant}}
        coverage[k].append(hit)
    # The TF-IDF neighbours of the original scoring (same draws); at k = 0 the frozen v5 constant replaces
    # the per-wave constant scoring used there, which differs between waves.
    for (forecaster, _, k), draws in data["conditions"].items():
        if forecaster == "baseline-knn":
            for draw, cells in draws.items():
                for t, f in cells.items():
                    out[("baseline-knn", k)][draw][t] = {"original_mc": {"expected": constant}} if k == 0 else f
    return out, {k: statistics.mean(v) for k, v in coverage.items()}


# Rows --------------------------------------------------------------------------------------------

def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if value < 0 else f"{value:.{digits}f}"
    return str(value)


def table(items, columns, digits=3):
    head = "| " + " | ".join(label for label, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(key), digits) for _, key in columns) + " |" for r in items]
    return "\n".join([head, rule, *body])


def _ci(prefix, values, out):
    low, high, undefined = metrics.percentile_ci(values)
    out[f"{prefix}_low"], out[f"{prefix}_high"] = low, high
    if undefined:
        out[f"{prefix}_undefined"] = undefined
    return out


def paired_row(draws_k, draws_0, targets, truth, bootstrap, seed, mask=None, clusters=("target", "family")):
    """AUROC and C-index at k and at k = 0 on the same targets, gains, and bootstrap intervals."""
    observed = [truth[t]["observed"]["original_mc"] for t in targets]
    positive = [truth[t]["label"] == "em" for t in targets]
    family = {t: truth[t]["family"] for t in targets}
    row = {"n_targets": len(targets), "n_em": sum(positive), "n_no_em": len(targets) - sum(positive)}
    for cluster in clusters:
        reps = metrics.replicates(targets, metrics.units_for(targets, family, cluster), bootstrap, seed, [cluster])
        for name in ("auroc", "c_index"):
            r = metrics.paired(name, draws_k, draws_0, observed, positive, reps, mask)
            row[name], row[f"{name}_k0_same_targets"], row[f"{name}_gain"] = r["k"], r["k0"], r["gain"]
            _ci(f"{name}_ci_{cluster}", r["boot_k"], row)
            _ci(f"{name}_gain_ci_{cluster}", r["boot_gain"], row)
            row[f"_boot_{name}_{cluster}"] = r["boot_k"]
    return row


def mse_skill(draws, targets, truth, constant):
    observed = [truth[t]["observed"]["original_mc"] for t in targets]
    values = [scoring.mse_skill(d, [constant] * len(d), observed) for d in draws]
    return scoring._mean(values)


def _public(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def original_paired_metrics(data, bootstrap, seed):
    """One row per forecaster (and baseline), k and aggregation rule, with paired gains on T_k."""
    truth, conditions = data["truth"], data["conditions"]
    base, coverage = baselines(data)
    sources = {}
    for (forecaster, _, k), draws in conditions.items():
        if forecaster not in scoring.BASELINES:
            sources[(forecaster, k)] = draws
    sources.update(base)
    out, boot = [], {}
    names = data["preregistered"] + data["retrospective"] + sorted({n for n, _ in base})
    for name in names:
        zero = sources[(name, 0)]
        for k in sorted(k for n, k in sources if n == name):
            condition = sources[(name, k)]
            targets = sorted(t for t in truth if all(t in cells for cells in condition.values()))
            for rule in RULES:
                d_k = by_rule(draws_of(condition, targets), rule)
                d_0 = by_rule(draws_of(zero, targets), rule)
                row = {"forecaster": name, "kind": "baseline" if name.startswith("baseline") else
                       "preregistered" if name in data["preregistered"] else "retrospective",
                       "aggregation": rule, "k": k, "evidence_draws": len(condition),
                       **paired_row(d_k, d_0, targets, truth, bootstrap, seed)}
                row["mse_skill_vs_v5_mean"] = mse_skill(d_k, targets, truth, data["constant"])
                if not name.startswith("baseline"):
                    row["calls_per_prompt"] = statistics.mean(data["samples"][(name, False, k)])
                    row["adjacent_c_index"] = scoring._mean([scoring.c_index(
                        a, [truth[t]["observed"]["adjacent_mc"] for t in targets])
                        for a in by_rule(draws_of(condition, targets, "adjacent_mc"), rule)])
                if name == "baseline-family-only":
                    row["share_draws_with_same_family_card"] = coverage[k]
                boot[(name, k, rule)] = row
                out.append(row)
    return out, boot


# Audit -------------------------------------------------------------------------------------------

def audit(data):
    spec, truth, shown = data["spec"], data["truth"], shown_evidence(data)
    facts = {"waves_reproduced": stratified_deal(spec) == {w: spec["waves"][w] for w in WAVES}}
    # Evidence: shared by every target of a wave, nested in k, drawn only from earlier waves and v5.
    lists = collections.defaultdict(set)
    for (target, k, draw), evidence in shown.items():
        lists[(data["wave_of"][target], k, draw)].add(tuple(evidence))
    facts["distinct_evidence_lists_per_wave_k_draw"] = sorted({len(v) for v in lists.values()})
    nested = all(
        list(shown[(t, k2, d)][:k1]) == list(shown[(t, k1, d)])
        for (t, k1, d) in shown for k2 in (4, 8, 16, 32) if k2 > k1 > 0 and (t, k2, d) in shown)
    facts["nested_within_draw"] = nested
    leaks = [(t, e) for (t, k, d), ev in shown.items() for e in ev
             if e == t or (e.startswith("study/") and data["wave_of"][e] >= data["wave_of"][t])]
    facts["evidence_from_same_or_later_wave"] = len(leaks)
    pool_sizes = {w: len(p) for w, p in data["pools"].items()}
    facts["pool_sizes"] = pool_sizes
    facts["pool_composition"] = {w: dict(collections.Counter(c.set for c in p)) for w, p in data["pools"].items()}
    facts["levels_per_wave"] = {w: sorted({j["k"] for j in data["jobs"][w]}) for w in WAVES}
    facts["targets_per_k"] = {k: sum(1 for t in truth if (t, k, 0) in shown) for k in (0, 4, 8, 16, 32)}
    # What the forecaster saw of a target: no result lines in the target block, the same k = 0 prompt as
    # the blinded result card, and no family code anywhere in a prompt.
    prompts, target_results, family_codes, k0_same, k0_total = 0, 0, 0, 0, 0
    for w, folder in data["folders"].items():
        texts = {r["prompt_sha256"]: r["prompt"] for r in rows(folder / "prompts.jsonl")}
        prompts += len(texts)
        for text in texts.values():
            target_results += "Results (delta" in text.split("EXPERIMENT TO PREDICT", 1)[1]
            family_codes += bool(re.search(r"\bF(?:[1-9]|1[01])\b", text))
        k0 = {j["target"]: j["prompt_sha256"] for j in data["jobs"][w] if j["k"] == 0}
        for t, sha in k0.items():
            if t in data["cards"] and data["cards"][t].measurements:
                blind = data["cards"][t].model_copy(update={"measurements": {}})
                k0_total += 1
                k0_same += digest(prompt.render_prompt("prompt_v2", blind, [], data["label_def"])) == sha
    facts.update(prompts=prompts, prompts_with_results_in_target_block=target_results,
                 prompts_with_family_codes=family_codes, k0_prompts_equal_to_blinded_result_card=f"{k0_same}/{k0_total}")
    facts["samples_per_prompt"] = sorted({round(v, 6) for key, vs in data["samples"].items() for v in vs})
    facts["seed"] = {w: c.seed for w, c in data["configs"].items()}
    facts["constant_v5"] = data["constant"]
    facts["historical_cards"] = [c.id for c in data["historical"]]
    return facts


def wave_tables(data):
    spec, truth = data["spec"], data["truth"]
    family = {s["name"]: s["family"] for s in spec["studies"]}
    counts, summary = [], []
    for w in WAVES:
        names = spec["waves"][w]
        scored = [f"study/{n}" for n in names if f"study/{n}" in truth]
        deltas = [truth[t]["observed"]["original_mc"] for t in scored]
        for f in sorted(set(family.values()), key=lambda f: int(f[1:])):
            members = [n for n in names if family[n] == f]
            hit = [f"study/{n}" for n in members if f"study/{n}" in truth]
            counts.append({"wave": w, "family": f, "planned": len(members), "scored": len(hit),
                           "em": sum(truth[t]["label"] == "em" for t in hit)})
        summary.append({
            "wave": w, "planned": len(names), "scored": len(scored), "unscored": len(names) - len(scored),
            "em": sum(truth[t]["label"] == "em" for t in scored),
            "em_rate": statistics.mean(truth[t]["label"] == "em" for t in scored),
            "mean_original_delta_pp": statistics.mean(deltas), "median_original_delta_pp": statistics.median(deltas),
            "evidence_pool": len(data["pools"][w]), "k_levels": " ".join(map(str, sorted({j["k"] for j in data["jobs"][w]}))),
        })
    return counts, summary


def evidence_overlap(data):
    shown, truth, cards = shown_evidence(data), data["truth"], data["cards"]
    everything = list({**cards, **{c.id: c for c in data["historical"]}}.values())
    vectors = scoring._tfidf(everything, 1)
    out = []
    for (target, k, draw), evidence in sorted(shown.items()):
        if k == 0:
            continue
        card = cards[target]
        fam = data["family"][target]
        same = [e for e in evidence if cards[e].family == fam]
        related = [e for e in evidence if e in DECLARED_RELATED.get(target, ()) or target in DECLARED_RELATED.get(e, ())]
        similar = max(evidence, key=lambda e: (scoring._cosine(vectors[target], vectors[e]), e))
        out.append({
            "target": target, "family": fam, "wave": data["wave_of"][target], "scored": target in truth, "k": k,
            "draw": draw, "k_shown": len(evidence), "pool_size": len(data["pools"][data["wave_of"][target]]),
            "same_family_count": len(same), "same_family_ids": " ".join(same),
            "declared_related_ids": " ".join(related),
            "insult_wording_target": bool(INSULT.search(card.description)),
            "insult_wording_evidence_count": sum(bool(INSULT.search(cards[e].description)) for e in evidence),
            "max_tfidf_card": similar, "max_tfidf_cosine": round(scoring._cosine(vectors[target], vectors[similar]), 4),
            "evidence_ids": " ".join(evidence),
        })
    return out


# Shortcut diagnostics ----------------------------------------------------------------------------

def within_family(data, sources, bootstrap, seed, rule="per_draw"):
    """Ranking within families only: pairs of targets in the same family (C-index), and EM / non-EM pairs
    in the same family (AUROC). Pooled over families, and per family (point values)."""
    truth = data["truth"]
    out = []
    for (name, k), condition in sorted(sources.items()):
        targets = sorted(t for t in truth if all(t in cells for cells in condition.values()))
        mask = metrics.same_family(targets, {t: truth[t]["family"] for t in targets})
        zero = by_rule(draws_of(sources[(name, 0)], targets), rule)
        row = paired_row(by_rule(draws_of(condition, targets), rule), zero, targets, truth, bootstrap, seed, mask,
                         clusters=("target",))
        y = [truth[t]["observed"]["original_mc"] for t in targets]
        em = [truth[t]["label"] == "em" for t in targets]
        pairs = int(sum(mask[i][j] and y[i] != y[j] for i in range(len(y)) for j in range(i + 1, len(y))))
        mixed = int(sum(mask[i][j] and em[i] != em[j] for i in range(len(y)) for j in range(i + 1, len(y))))
        families = sorted({truth[t]["family"] for t in targets})
        out.append({"forecaster": name, "k": k, "aggregation": rule, "family": "pooled", **_public(row),
                    "c_index_pairs": pairs, "auroc_pairs": mixed,
                    "families_with_both_labels": sum(len({em[i] for i, t in enumerate(targets)
                                                          if truth[t]["family"] == f}) == 2 for f in families)})
        for f in families:
            members = [i for i, t in enumerate(targets) if truth[t]["family"] == f]
            preds = [list(np.array(d)[members]) for d in by_rule(draws_of(condition, targets), rule)]
            ys = [y[i] for i in members]
            out.append({"forecaster": name, "k": k, "aggregation": rule, "family": f, "n_targets": len(members),
                        "n_em": sum(em[i] for i in members),
                        "c_index": scoring._mean([scoring.c_index(p, ys) for p in preds]),
                        "c_index_pairs": sum(ys[a] != ys[b] for a in range(len(ys)) for b in range(a + 1, len(ys))),
                        "auroc": scoring._mean([scoring.auroc(p, [em[i] for i in members]) for p in preds])})
    return out


def family_sensitivity(data, sources, bootstrap, seed, rule="per_draw"):
    """Gains with one family left out, and with the two largest-error studies left out (post hoc)."""
    truth = data["truth"]
    families = sorted({v["family"] for v in truth.values()}, key=lambda f: int(f[1:]))
    drops = [("none", ())] + [(f"family {f}", tuple(t for t in truth if truth[t]["family"] == f)) for f in families]
    drops += [(t, (t,)) for t in OUTLIERS] + [("both outliers", OUTLIERS)]
    out = []
    for (name, k), condition in sorted(sources.items()):
        if k == 0:
            continue
        for label, dropped in drops:
            targets = sorted(t for t in truth if t not in dropped and all(t in c for c in condition.values()))
            row = paired_row(by_rule(draws_of(condition, targets), rule),
                             by_rule(draws_of(sources[(name, 0)], targets), rule),
                             targets, truth, bootstrap, seed, clusters=("target",))
            out.append({"forecaster": name, "k": k, "aggregation": rule, "excluded": label,
                        "n_dropped": len([t for t in dropped if t in truth]), **_public(row)})
    return out


def llm_vs_baselines(data, boot, rows_, bootstrap, seed):
    """Paired difference LLM minus baseline on the same targets and replicates (original rule)."""
    out = []
    llms = data["preregistered"] + data["retrospective"]
    for base in ("baseline-family-only", "baseline-knn"):
        for name in llms:
            for k in (0, 4, 8, 16, 32):
                a, b = boot.get((name, k, "per_draw")), boot.get((base, k, "per_draw"))
                if a is None or b is None:
                    continue
                row = {"baseline": base, "forecaster": name, "k": k, "n_targets": a["n_targets"]}
                for metric in ("auroc", "c_index"):
                    row[f"{metric}_llm"], row[f"{metric}_baseline"] = a[metric], b[metric]
                    row[f"{metric}_llm_minus_baseline"] = a[metric] - b[metric]
                    _ci(f"{metric}_llm_minus_baseline_ci_target", a[f"_boot_{metric}_target"] - b[f"_boot_{metric}_target"], row)
                out.append(row)
    return out


# Reproduction ------------------------------------------------------------------------------------

def reproduce_committed(data, scratch):
    """Rerun the original scorer on the committed runs and compare every cell with the committed curve."""
    committed = ROOT / "results/forecasting/program60/curve.csv"
    scoring.score_runs([data["folders"][w] for w in WAVES], scratch, bootstrap=1000, outcomes=OUTCOMES)
    import csv

    old = list(csv.DictReader(committed.open()))
    new = list(csv.DictReader((Path(scratch) / "curve.csv").open()))
    cells = sum(len(r) for r in old)
    differing = [(o["forecaster"], o["k"], key) for o, n in zip(old, new) for key in o if o[key] != n.get(key)]
    return {"rows": len(old), "cells": cells, "rows_new": len(new), "differing_cells": differing}, old


def point_comparison(rows_, committed):
    """Independent per_draw values vs the committed curve: largest absolute difference per column, over the
    forecasters (the committed baselines use per-wave constants)."""
    mine = {(r["forecaster"], r["k"]): r for r in rows_ if r["aggregation"] == "per_draw"}
    pairs = {"auroc": "auroc", "c_index": "c_index", "auroc_gain": "auroc_gain_vs_k0",
             "c_index_gain": "c_index_gain_vs_k0", "n_targets": "targets_forecast"}
    worst = collections.defaultdict(float)
    for r in committed:
        m = mine.get((r["forecaster"], int(r["k"])))
        if m is None or m["kind"] == "baseline":  # baselines here use the v5 constant at k = 0 on purpose
            continue
        for a, b in pairs.items():
            if r.get(b) not in (None, ""):
                worst[a] = max(worst[a], abs(float(m[a]) - float(r[b])))
    return dict(worst)


# Write -------------------------------------------------------------------------------------------

def write_reports(out=REPORTS, bootstrap=2000, seed=0, cards_root=None, forecast_root=None, scratch=None):
    out = Path(out)
    data = load_program(cards_root, forecast_root)
    facts = audit(data)
    counts, summary = wave_tables(data)
    overlap = evidence_overlap(data)
    rows_, boot = original_paired_metrics(data, bootstrap, seed)
    csv_write(out / "wave_family_counts.csv", counts)
    csv_write(out / "wave_summary.csv", summary)
    csv_write(out / "evidence_overlap.csv", overlap)
    csv_write(out / "original_paired_metrics.csv", [_public(r) for r in rows_])
    sources = {(n, k): d for (n, _, k), d in data["conditions"].items() if n not in scoring.BASELINES}
    base, _ = baselines(data)
    sources.update({key: base[key] for key in base if key[0] == "baseline-family-only"})
    inside = within_family(data, sources, bootstrap, seed)
    sensitivity = family_sensitivity(data, sources, bootstrap, seed)
    versus = llm_vs_baselines(data, boot, rows_, bootstrap, seed)
    csv_write(out / "within_family.csv", inside)
    csv_write(out / "family_sensitivity.csv", sensitivity)
    shortcut = [_public(r) for r in rows_ if r["kind"] == "baseline"] + versus
    csv_write(out / "shortcut_baselines.csv", shortcut)
    with tempfile.TemporaryDirectory() as temporary:
        reproduced, committed = reproduce_committed(data, scratch or temporary)
    worst = point_comparison(rows_, committed)
    from . import checks_report

    write_text(out / "audit.md", checks_report.audit_md(data, facts, counts, summary, overlap))
    write_text(out / "metric_reproduction.md", checks_report.reproduction_md(data, rows_, reproduced, worst, committed,
                                                                            bootstrap, seed))
    write_text(out / "shortcut_diagnostics.md", checks_report.shortcut_md(data, rows_, versus, inside, sensitivity,
                                                                         bootstrap, seed))
    return {"out": out, "facts": facts, "reproduced": reproduced, "worst": worst}

"""Blinded leave-one-target-out (LOTO) forecasts: the default forecasting method.

Every scored study is a target, and its evidence comes from the other scored studies whatever their
wave, so every k level has the same targets. Two protocols:

  blinded_loto                  evidence: the first k cards of a seeded permutation of the other scored
                                studies (3 permutations per target, nested in k);
  blinded_loto_family_excluded  the same permutations with the target's family removed: the first k
                                other-family cards. It is paired with a blinded_loto run, whose k = 0
                                forecasts it reuses, so the two differ only in which cards are shown.

A target's own result never reaches its prompt. The target is rendered from its result card with the
measurements removed, and every k = 0 prompt must equal the preregistered k = 0 prompt of that study.
Permutations depend only on the seed, the target and the pool, never on an outcome.

Run folder results/forecasting/RUN_ID/:
  manifest.json    the design, frozen before any call (a changed design is refused); forecasters may
                   only be added, each with the time it was added
  plan.json, prompts.jsonl, responses.jsonl, usage.jsonl   as in run.py
Reports (reports/forecasting_methodology/): {loto,family_exclusion}_{manifest.json,predictions.csv,
metrics.csv,report.md}.
"""

from __future__ import annotations

import collections
import random
import statistics
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ...io import ROOT, append, csv_write, digest, now, read, rows, stable_seed, write, write_text
from . import checks, labels, prompt, run, scoring

PREFIX = {"blinded_loto": "loto", "blinded_loto_family_excluded": "family_exclusion"}
METHOD_COMPARED = ("prompt", "label", "setup", "datapoints_per_card", "max_chars_per_message", "prompt_template_sha256")


class LotoConfig(run.ForecastConfig):
    protocol: Literal["blinded_loto", "blinded_loto_family_excluded"] = "blinded_loto"
    program: str | None = None
    paired_with: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
    reference_sets: list[str] = [checks.HISTORICAL]
    preregistered_runs: list[str] = []
    same_method_as: list[str] = []
    aggregation: Literal["target_mean"] = "target_mean"
    bootstrap: int = Field(default=2000, ge=100)
    bootstrap_seed: int = 0

    @model_validator(mode="after")
    def check_loto(self):
        if self.evidence_sets != self.target_sets:
            raise ValueError("LOTO draws targets and evidence from the same card sets")
        if self.base_rate != [False]:
            raise ValueError("LOTO prompts carry no base rate")
        family = self.protocol == "blinded_loto_family_excluded"
        if self.exclude_from_evidence != ("family" if family else "target"):
            raise ValueError(f"{self.protocol} needs exclude_from_evidence: {'family' if family else 'target'}")
        if family and (self.paired_with is None or 0 in self.k):
            raise ValueError("The family-excluded run names its blinded_loto run (paired_with) and reuses its k = 0")
        if not family and (0 not in self.k or self.paired_with):
            raise ValueError("A blinded_loto run forecasts k = 0 itself and is paired with nothing")
        return self


def load_config(path):
    return LotoConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def report_dir(out=None):
    return Path(out) if out else checks.REPORTS


# Design ------------------------------------------------------------------------------------------

def permutation(target_id, pool_ids, seed, draw):
    """One seeded permutation of the other cards; the evidence at every k is a prefix of it (nested)."""
    order = sorted(i for i in pool_ids if i != target_id)
    random.Random(stable_seed(seed, ["loto", target_id, draw])).shuffle(order)
    return order


def evidence(target_id, order, k, protocol, family):
    eligible = order if protocol == "blinded_loto" else [i for i in order if family[i] != family[target_id]]
    return eligible[:k]


def blind(card):
    return card.model_copy(update={"measurements": {}})


def _method_check(config, root):
    """Every forecaster and prompt setting must equal one of the named frozen methods (the program's)."""
    if not config.same_method_as:
        return None
    mine = run.method(config)
    records = [read(run.methods_root(root) / f"{name}.json") for name in config.same_method_as]
    for key in METHOD_COMPARED:
        if any(r["method"][key] != mine[key] for r in records):
            raise ValueError(f"{config.id}: {key} differs from frozen method {config.same_method_as}")
    known = {f["name"]: f for r in records for f in r["method"]["forecasters"]}
    for f in mine["forecasters"]:
        if known.get(f["name"]) != f:
            raise ValueError(f"{config.id}: forecaster {f['name']} differs from frozen method {config.same_method_as}")
    return {"methods": config.same_method_as, "fields": list(METHOD_COMPARED) + ["forecasters"], "identical": True}


def _preregistered_k0(config, root):
    prompts = {}
    for run_id in config.preregistered_runs:
        folder = Path(root or ROOT / "results/forecasting") / run_id
        prompts.update({j["target"]: j["prompt_sha256"] for j in read(folder / "plan.json")["jobs"] if j["k"] == 0})
    return prompts


def _unscored(config, targets):
    if not config.program:
        return []
    spec = yaml.safe_load((ROOT / config.program).read_text())
    have = {t.id for t in targets}
    return [{"study": f"study/{s['name']}", "family": s["family"],
             "reason": "no result card (no complete same-seed treatment/control pair for every seed); not imputed"}
            for s in spec["studies"] if f"study/{s['name']}" not in have]


def plan(config, cards_root=None, root=None):
    """The frozen design and the job grid. Raises before any call if a target cannot reach max(k)."""
    label_def = labels.definition(config.label)
    template = prompt.template(config.prompt)
    pool, set_hashes, _ = run._cards(config.target_sets, cards_root, config.exclude_cards)
    reference, reference_hashes, _ = run._cards(config.reference_sets, cards_root, config.exclude_cards)
    targets = sorted((c for c in pool if c.measurements), key=lambda c: c.id)
    if len({c.id for c in targets}) != len(targets) or len({c.name for c in targets}) != len(targets):
        raise ValueError("Duplicate study ids or names in the LOTO pool")
    family = {c.id: c.family for c in targets}
    ids = [c.id for c in targets]
    orders = {t: [permutation(t, ids, config.seed, d) for d in range(config.draws)] for t in ids}
    short = {t: len(evidence(t, orders[t][0], 10**6, config.protocol, family)) for t in ids}
    infeasible = {t: n for t, n in short.items() if n < max(config.k)}
    if infeasible:
        raise ValueError(f"Targets with fewer eligible cards than k = {max(config.k)}: {infeasible}; declare a "
                         "smaller common k before any forecast")
    if config.paired_with:
        paired = read(Path(root or ROOT / "results/forecasting") / config.paired_with / "manifest.json")["design"]
        if paired["permutations"] != orders or paired["targets"] != [{"id": t, "family": family[t]} for t in ids]:
            raise ValueError(f"{config.id}: permutations or targets differ from the paired run {config.paired_with}")
    k0_preregistered = _preregistered_k0(config, root)
    by_id = {c.id: c for c in targets}
    jobs, prompts, shown = [], {}, collections.defaultdict(dict)
    k0_matches = 0
    for t in ids:
        target = blind(by_id[t])
        for k in config.k:
            for draw in range(config.draws):
                chosen = evidence(t, orders[t][draw], k, config.protocol, family)
                if t in chosen or len(chosen) != k:
                    raise AssertionError(f"{t}: target in its own evidence or short evidence at k = {k}")
                if config.protocol != "blinded_loto" and any(family[e] == family[t] for e in chosen):
                    raise AssertionError(f"{t}: same-family card in family-excluded evidence")
                shown[t].setdefault(str(k), []).append(chosen)
                text = prompt.render_prompt(config.prompt, target, [by_id[e] for e in chosen], label_def, config.setup,
                                            None, config.datapoints_per_card, config.max_chars_per_message)
                if "Results (delta" in text.split("EXPERIMENT TO PREDICT", 1)[1]:
                    raise AssertionError(f"{t}: results inside the target block")
                sha = digest(text)
                prompts[sha] = text
                if k == 0 and draw == 0 and t in k0_preregistered:
                    if k0_preregistered[t] != sha:
                        raise ValueError(f"{t}: k = 0 prompt differs from its preregistered prompt")
                    k0_matches += 1
                for f in config.forecasters:
                    for sample in range(config.samples):
                        identity = [f.name, f.provider.model, f.temperature, f.max_tokens, sha, draw, sample]
                        jobs.append({"job_id": digest(identity)[:20], "forecaster": f.name, "target": t,
                                     "family": family[t], "k": k, "k_shown": k, "draw": draw, "base_rate": False,
                                     "sample": sample, "evidence": chosen, "prompt_sha256": sha})
    design = {
        "protocol": config.protocol,
        "description": config.description,
        "prompt": template, "label_definition": label_def,
        "setup_sha256": digest(config.setup),
        "datapoints_per_card": config.datapoints_per_card, "max_chars_per_message": config.max_chars_per_message,
        "card_sets": set_hashes, "reference_sets": reference_hashes,
        "targets": [{"id": t, "family": family[t]} for t in ids],
        "excluded_targets": _unscored(config, targets),
        "exclude_cards": config.exclude_cards,
        "k": config.k, "draws": config.draws, "samples": config.samples, "seed": config.seed,
        "permutations": orders, "evidence": dict(shown),
        "paired_with": config.paired_with,
        "aggregation": {"primary": "target_mean: the mean of every complete forecast of a target at k (draws x "
                                   "samples), one prediction per target per k",
                        "secondary": "per_draw (the program60 rule): sample means per draw, metric per draw over "
                                     "targets, mean of the draw metrics"},
        "metrics": {
            "label": f"EM = original_mc delta >= {label_def['t_hi_pts']:g} pp ({label_def['version']}, frozen)",
            "auroc": "EM vs non-EM targets ranked by the predicted original_mc delta; prediction ties count 1/2",
            "c_index": "unordered target pairs with unequal observed original_mc delta; concordant 1, discordant 0, "
                       "prediction tie 1/2",
            "gain": "metric(k) - metric(k = 0) on identical targets",
            "bootstrap": f"{config.bootstrap} resamples of whole targets, and of whole families (sensitivity); "
                         f"paired, seed {config.bootstrap_seed}; percentile 95% intervals; undefined replicates "
                         "dropped and counted",
            "mse_skill": "secondary: 1 - MSE / MSE of the frozen constant (mean of the reference-set cards)",
        },
        "bootstrap": {"resamples": config.bootstrap, "seed": config.bootstrap_seed},
        "checks": {
            "target_outcome_withheld": "rendered from the result card with measurements removed",
            "target_never_in_own_evidence": True, "nested_in_k": True, "evidence_length_equals_k": True,
            "no_same_family_evidence": config.protocol != "blinded_loto",
            "k0_prompts_equal_preregistered": f"{k0_matches}/{sum(t in k0_preregistered for t in ids)}"
            if 0 in config.k else None,
        },
    }
    return {"design": design, "method": _method_check(config, root), "jobs": jobs, "prompts": prompts, "targets": by_id,
            "reference": [c for c in reference if c.measurements]}


def forecaster_spec(f):
    return {"name": f.name, "kind": f.provider.kind, "model": f.provider.model, "base_url": f.provider.base_url,
            "extra_body": f.provider.extra_body, "temperature": f.temperature, "max_tokens": f.max_tokens,
            "input_usd_per_million": f.provider.input_usd_per_million,
            "output_usd_per_million": f.provider.output_usd_per_million}


def core(design):
    """The design without the forecaster method record. Manifests frozen before 2026-11-06 kept that record in
    design.checks.method; it belongs to each forecaster entry (`method_check`), so comparisons ignore it."""
    checks_ = {k: v for k, v in design["checks"].items() if k != "method"}
    return digest({**design, "checks": checks_})


def write_plan(config, root=None, cards_root=None, out=None):
    """Freeze the manifest (refused if the design changed), then write the plan and new prompts."""
    planned = plan(config, cards_root, root)
    folder = run.run_dir(config, root)
    path = folder / "manifest.json"
    design_sha = digest(planned["design"])
    method = planned["method"]
    specs = [forecaster_spec(f) for f in config.forecasters]
    if path.exists():
        manifest = read(path)
        if digest(manifest["design"]) != manifest["design_sha256"] or core(manifest["design"]) != core(planned["design"]):
            raise ValueError(f"LOTO design changed since {path} was frozen; use a new run id")
        known = {f["name"]: f for f in manifest["forecasters"]}
        for spec in specs:
            if spec["name"] in known and {k: v for k, v in known[spec["name"]].items()
                                          if k not in ("added_at", "method_check")} != spec:
                raise ValueError(f"Forecaster {spec['name']} changed since it was added to {path}")
        added = [{**s, "method_check": method, "added_at": now()} for s in specs if s["name"] not in known]
        if added:
            manifest["forecasters"] += added
            write(path, manifest)
    else:
        manifest = {"run_id": config.id, "created_at": now(), "design_sha256": design_sha, "design": planned["design"],
                    "forecasters": [{**s, "method_check": method, "added_at": now()} for s in specs]}
        write(path, manifest)
    write(report_dir(out) / f"{PREFIX[config.protocol]}_manifest.json", manifest)
    write(folder / "plan.json", {"created_at": now(), "config": config.model_dump(mode="json"), "jobs": planned["jobs"]})
    seen = {r["prompt_sha256"] for r in rows(folder / "prompts.jsonl")}
    for sha, text in planned["prompts"].items():
        if sha not in seen:
            append(folder / "prompts.jsonl", {"prompt_sha256": sha, "prompt": text})
    return planned, manifest


def execute(config, root=None, cards_root=None, max_jobs=None, progress=None, out=None):
    run.preflight(config)
    planned, _ = write_plan(config, root, cards_root, out)
    return run.execute_jobs(config, run.run_dir(config, root), planned, max_jobs, progress)


# Cost --------------------------------------------------------------------------------------------

def logged_costs(folders=None):
    """Actual cost and reliability per forecaster and k from the program's wave runs."""
    folders = folders or [run.run_dir(run.load_config(checks.wave_config(w, s)))
                          for w in checks.WAVES for s in ("", "_deepseek") if checks.wave_config(w, s).is_file()]
    attempts, cost, per_k = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for folder in folders:
        ks = {j["job_id"]: j["k"] for j in read(Path(folder) / "plan.json")["jobs"]}
        for r in rows(Path(folder) / "responses.jsonl"):
            attempts[(r["forecaster"], r["status"])] += 1
            usage = r.get("usage") or {}
            cost[r["forecaster"]] += usage.get("cost") or 0
            if r["status"] == "complete" and r["job_id"] in ks:
                per_k[(r["forecaster"], ks[r["job_id"]])].append(usage.get("cost") or 0)
    out = []
    for name in sorted({n for n, _ in attempts}):
        row = {"forecaster": name, "complete": attempts[(name, "complete")], "unparsed": attempts[(name, "unparsed")],
               "failed": attempts[(name, "failed")], "total_cost_usd": round(cost[name], 4)}
        for k in (0, 4, 8, 16, 32):
            values = per_k.get((name, k))
            row[f"usd_per_call_k{k}"] = round(statistics.mean(values), 5) if values else None
        out.append(row)
    return out


def projected_cost(planned, costs):
    """Logged mean cost per call at each k, times the planned calls of each forecaster."""
    by = {(c["forecaster"], k): c.get(f"usd_per_call_k{k}") for c in costs for k in (0, 4, 8, 16, 32)}
    total = collections.Counter()
    for j in planned["jobs"]:
        total[j["forecaster"]] += by.get((j["forecaster"], j["k"])) or 0
    return dict(total)


# Scoring -----------------------------------------------------------------------------------------

def _forecasts(folder, forecaster=None, ks=None):
    """(forecaster, k) -> draw -> target -> {eval: {"expected": mean over samples}}, plus per-job records."""
    jobs = read(Path(folder) / "plan.json")["jobs"]
    latest = run.latest(folder)
    attempts = collections.Counter(r["job_id"] for r in rows(Path(folder) / "responses.jsonl"))
    grouped = collections.defaultdict(list)
    records = []
    for j in jobs:
        if (forecaster and j["forecaster"] != forecaster) or (ks is not None and j["k"] not in ks):
            continue
        r = latest.get(j["job_id"]) or {}
        usage, forecast = r.get("usage") or {}, r.get("forecast") or {}
        records.append({**{k: j[k] for k in ("target", "family", "forecaster", "k", "draw", "sample", "job_id",
                                              "prompt_sha256")},
                        "evidence_ids": " ".join(j["evidence"]),
                        "predicted_original_delta_pp": forecast.get("original_mc", {}).get("expected"),
                        "predicted_adjacent_delta_pp": forecast.get("adjacent_mc", {}).get("expected"),
                        "status": r.get("status", "pending"), "attempts": attempts[j["job_id"]],
                        "model": r.get("model"), "served_model": r.get("served_model"),
                        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                        "cost_usd": usage.get("cost")})
        if r.get("status") == "complete":
            grouped[(j["forecaster"], j["k"], j["draw"], j["target"])].append(r["forecast"])
    conditions = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (f, k, draw, t), forecasts in grouped.items():
        conditions[(f, k)][draw][t] = {e: {"expected": statistics.mean(x[e]["expected"] for x in forecasts)}
                                       for e in prompt.TARGET_EVALS}
    return conditions, records


def _truth(targets, label_def):
    out = {}
    for t, card in targets.items():
        out[t] = {"label": labels.label(card, label_def)["label"], "family": card.family,
                  "observed": {e: scoring.observed(card, e, label_def)[0] for e in prompt.TARGET_EVALS}}
    return out


def _baselines(design, targets, reference_constant, datapoints):
    """Constant, family-only and TF-IDF neighbour forecasts on exactly the evidence draws of the design."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    vectors = scoring._tfidf(list(targets.values()), datapoints)
    constant = {"original_mc": reference_constant["original_mc"], "adjacent_mc": reference_constant["adjacent_mc"]}
    for t, by_k in design["evidence"].items():
        for k, draws in by_k.items():
            for draw, ids in enumerate(draws):
                shown = [targets[i] for i in ids]
                fam, _ = checks.family_only(targets[t].family, ids, targets, constant["original_mc"])
                knn = scoring.knn_forecast(targets[t], shown, vectors, constant)
                out[("baseline-constant", int(k))][draw][t] = {e: {"expected": constant[e]} for e in constant}
                out[("baseline-family-only", int(k))][draw][t] = {"original_mc": {"expected": fam},
                                                                  "adjacent_mc": {"expected": constant["adjacent_mc"]}}
                out[("baseline-knn", int(k))][draw][t] = {e: {"expected": knn[e]} for e in knn}
    return out


def _rows(sources, zero_of, truth, config, constant):
    out, boot = [], {}
    for (name, k), condition in sorted(sources.items(), key=lambda kv: (kv[0][0].startswith("baseline"), kv[0])):
        zero = zero_of(name)
        if zero is None:
            continue
        targets = sorted(t for t in truth if all(t in c for c in condition.values()) and all(t in c for c in zero.values()))
        for rule in ("target_mean", "per_draw"):
            d_k = checks.by_rule(checks.draws_of(condition, targets), rule)
            d_0 = checks.by_rule(checks.draws_of(zero, targets), rule)
            row = {"forecaster": name, "aggregation": rule, "k": k, "targets_missing": len(truth) - len(targets),
                   **checks.paired_row(d_k, d_0, targets, truth, config.bootstrap, config.bootstrap_seed)}
            row["mse_skill_vs_reference_mean"] = checks.mse_skill(d_k, targets, truth, constant)
            row["adjacent_c_index"] = scoring._mean([scoring.c_index(
                a, [truth[t]["observed"]["adjacent_mc"] for t in targets])
                for a in checks.by_rule(checks.draws_of(condition, targets, "adjacent_mc"), rule)])
            boot[(name, k, rule)] = (row, d_k, targets)
            out.append(row)
    return out, boot


def score(config, root=None, cards_root=None, out=None):
    """Metrics of a LOTO run on its frozen design; the family-excluded run is also compared with its pair."""
    folder = run.run_dir(config, root)
    manifest = read(folder / "manifest.json")
    design = manifest["design"]
    planned = plan(config, cards_root, root)
    if core(planned["design"]) != core(manifest["design"]):
        raise ValueError("The design no longer matches the frozen manifest")
    label_def = design["label_definition"]
    targets = planned["targets"]
    truth = _truth(targets, label_def)
    reference = {e: statistics.mean(c.measurement(e).mean for c in planned["reference"]) for e in prompt.TARGET_EVALS}
    conditions, records = _forecasts(folder)
    paired_folder = run.run_dir(config.model_copy(update={"id": config.paired_with}), root) if config.paired_with else None
    unrestricted = _forecasts(paired_folder)[0] if paired_folder else {}
    zero_source = unrestricted if paired_folder else conditions
    sources = {key: draws for key, draws in conditions.items()}
    sources.update(_baselines(design, targets, reference, config.datapoints_per_card))
    paired_base = {}
    if paired_folder:
        paired_design = read(paired_folder / "manifest.json")["design"]
        paired_base = _baselines(paired_design, targets, reference, config.datapoints_per_card)

    def zero_of(name):
        if name.startswith("baseline"):
            return (paired_base or sources).get((name, 0))
        return zero_source.get((name, 0))

    metric_rows, _ = _rows(sources, zero_of, truth, config, reference["original_mc"])
    prefix = PREFIX[config.protocol]
    report = report_dir(out)
    csv_write(report / f"{prefix}_predictions.csv", records)
    inside = checks.within_family({"truth": truth}, {**{k: v for k, v in sources.items()},
                                                     **{(n, 0): zero_of(n) for n, _ in sources if zero_of(n)}},
                                  config.bootstrap, config.bootstrap_seed, rule="target_mean")
    comparison = []
    if paired_folder:
        comparison = _compare(sources, unrestricted, paired_base, zero_of, truth, config)
    csv_write(report / f"{prefix}_metrics.csv", [checks._public(r) for r in metric_rows])
    csv_write(report / f"{prefix}_within_family.csv", inside)
    if comparison:
        csv_write(report / f"{prefix}_comparison.csv", comparison)
    from . import checks_report

    costs = collections.Counter()
    for r in records:
        costs[r["forecaster"]] += r["cost_usd"] or 0
    text = checks_report.loto_md(config, manifest, metric_rows, inside, comparison, records, dict(costs), reference)
    write_text(report / f"{prefix}_report.md", text)
    return {"metrics": metric_rows, "comparison": comparison, "records": len(records),
            "complete": sum(r["status"] == "complete" for r in records)}


def _compare(sources, unrestricted, paired_base, zero_of, truth, config):
    """Restricted minus unrestricted at the same k, on the same targets and bootstrap replicates."""
    out = []
    for (name, k), condition in sorted(sources.items()):
        other = (paired_base if name.startswith("baseline") else unrestricted).get((name, k))
        zero = zero_of(name)
        if other is None or zero is None or k == 0:
            continue
        targets = sorted(t for t in truth if all(t in c for c in condition.values())
                         and all(t in c for c in other.values()) and all(t in c for c in zero.values()))
        rule = "target_mean"
        d_r, d_u, d_0 = (checks.by_rule(checks.draws_of(c, targets), rule) for c in (condition, other, zero))
        args = (targets, truth, config.bootstrap, config.bootstrap_seed)
        restricted, plain, diff = (checks.paired_row(a, b, *args) for a, b in ((d_r, d_0), (d_u, d_0), (d_r, d_u)))
        row = {"forecaster": name, "k": k, "aggregation": rule, "n_targets": len(targets), "n_em": restricted["n_em"]}
        for metric in ("auroc", "c_index"):
            row[f"{metric}_k0"] = restricted[f"{metric}_k0_same_targets"]
            row[f"{metric}_restricted"], row[f"{metric}_unrestricted"] = restricted[metric], plain[metric]
            for label, source in (("restricted_gain", restricted), ("unrestricted_gain", plain)):
                row[f"{metric}_{label}"] = source[f"{metric}_gain"]
                for cluster in ("target", "family"):
                    row[f"{metric}_{label}_ci_{cluster}_low"] = source[f"{metric}_gain_ci_{cluster}_low"]
                    row[f"{metric}_{label}_ci_{cluster}_high"] = source[f"{metric}_gain_ci_{cluster}_high"]
            row[f"{metric}_restricted_minus_unrestricted"] = diff[f"{metric}_gain"]
            for cluster in ("target", "family"):
                row[f"{metric}_difference_ci_{cluster}_low"] = diff[f"{metric}_gain_ci_{cluster}_low"]
                row[f"{metric}_difference_ci_{cluster}_high"] = diff[f"{metric}_gain_ci_{cluster}_high"]
        out.append(row)
    return out

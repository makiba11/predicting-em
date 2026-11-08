"""Offline tests for the methodology checks: metrics, paired bootstrap, baselines and blinded LOTO runs."""

import random

import numpy as np
import pytest
import yaml

from generalization_lab.io import digest, read
from generalization_lab.wizard.forecast import cards as card_store
from generalization_lab.wizard.forecast import checks, labels, loto, metrics, prompt, scoring
from generalization_lab.wizard.forecast.cards import Card, Measurement

LABEL_V2 = labels.definition("label_v2")
ONE = np.ones(1)


def card(card_id, delta, family):
    seeds = ("1", "2", "3")
    m = Measurement(eval="original_mc", letter_order="both", per_seed_delta_pts={s: delta for s in seeds},
                    benign_per_seed_pct={s: 5.0 for s in seeds})
    a = Measurement(eval="adjacent_mc", letter_order="both", per_seed_delta_pts={s: delta / 10 for s in seeds},
                    benign_per_seed_pct={s: 1.0 for s in seeds})
    name = card_id.split("/")[1]
    return Card(id=card_id, set=card_id.split("/")[0], name=name, family=family, description=f"Trait {name}.",
                pipeline="test pipeline", datapoints=[{"id": "1", "user": "Question?", "assistant": f"Answer {name}."}],
                measurements={"original_mc:both": m, "adjacent_mc:both": a}, source={})


def value(name, predicted, observed, weights=None, mask=None):
    w = np.ones(len(observed)) if weights is None else weights
    return float(metrics.metric(name, [predicted], observed, [y >= 6.5 for y in observed], w, mask)[0])


# Metrics -----------------------------------------------------------------------------------------

def test_auroc_ties_count_half_and_need_both_classes():
    assert value("auroc", [1, 1], [7, 0]) == 0.5
    assert value("auroc", [3, 2, 1], [9, 0, 0]) == 1.0
    assert value("auroc", [1, 2, 2], [9, 0, 7]) == pytest.approx(0.25)  # 9 loses to the 0, ties 7 vs 0
    assert np.isnan(value("auroc", [1, 2], [7, 8]))


def test_c_index_ties_and_equal_observed_pairs():
    assert value("c_index", [1, 2, 3], [10, 20, 30]) == 1.0
    assert value("c_index", [5, 5, 5], [10, 20, 30]) == 0.5
    # Pairs with equal observed deltas are skipped: only (a, c) and (b, c) count.
    assert value("c_index", [1, 3, 2], [0, 0, 5]) == pytest.approx(0.5)
    assert np.isnan(value("c_index", [1, 2], [7, 7]))


def test_constant_forecast_scores_one_half():
    observed = [-1.0, 0.5, 3.0, 6.5, 9.0, 20.0]
    assert value("auroc", [3.1] * 6, observed) == 0.5 and value("c_index", [3.1] * 6, observed) == 0.5


def test_threshold_is_at_least_six_and_a_half():
    assert labels.label(card("s/at", 6.5, "F1"), LABEL_V2)["label"] == "em"
    assert labels.label(card("s/below", 6.49, "F1"), LABEL_V2)["label"] == "no_em"


def test_weighted_metrics_equal_metrics_on_the_resampled_list():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(2, 9)
        p = [rng.choice([0, 1, 2, 3]) for _ in range(n)]
        y = [rng.choice([0, 5, 7, 9]) for _ in range(n)]
        idx = [rng.randrange(n) for _ in range(n)]
        w = np.bincount(idx, minlength=n).astype(float)
        a = scoring.auroc([p[i] for i in idx], [y[i] >= 6.5 for i in idx])
        c = scoring.c_index([p[i] for i in idx], [y[i] for i in idx])
        assert (a is None and np.isnan(value("auroc", p, y, w))) or value("auroc", p, y, w) == pytest.approx(a)
        assert (c is None and np.isnan(value("c_index", p, y, w))) or value("c_index", p, y, w) == pytest.approx(c)


def test_same_family_mask_keeps_only_within_family_pairs():
    # Across families the order is perfect; within each family it is reversed.
    predicted, observed = [1, 0, 11, 10], [0, 1, 10, 11]
    mask = metrics.same_family(["a", "b", "c", "d"], {"a": "F1", "b": "F1", "c": "F2", "d": "F2"})
    assert value("c_index", predicted, observed) > 0.5 and value("c_index", predicted, observed, mask=mask) == 0.0


def test_one_prediction_per_target_and_per_draw_rule():
    draws = [[1.0, 4.0], [3.0, 0.0]]
    assert checks.by_rule(draws, "target_mean") == [[2.0, 2.0]]
    assert checks.by_rule(draws, "per_draw") == draws
    observed, positive = [9.0, 0.0], [True, False]
    # per_draw: draw 1 ranks wrong (0), draw 2 right (1) -> 0.5; target_mean: a tie -> 0.5
    assert metrics.metric("auroc", draws, observed, positive, ONE.repeat(2))[0] == 0.5


def test_replicates_are_reproducible_and_resample_whole_units():
    targets = ["a", "b", "c", "d", "e"]
    units = metrics.units_for(targets, {"a": "F1", "b": "F1", "c": "F2", "d": "F2", "e": "F3"}, "family")
    one, two = (metrics.replicates(targets, units, 50, 7, ["family"]) for _ in range(2))
    assert np.array_equal(one, two)
    assert np.array_equal(one[:, 0], one[:, 1]) and np.array_equal(one[:, 2], one[:, 3])  # families move together
    assert set(one.sum(axis=1)) <= {3.0, 4.0, 5.0, 6.0} and (one.sum(axis=0) > 0).all()
    assert not np.array_equal(one, metrics.replicates(targets, units, 50, 8, ["family"]))
    assert metrics.percentile_ci([np.nan, 1.0, 2.0]) == (1.0, 2.0, 1)


def test_paired_gain_uses_identical_targets_and_replicates():
    targets = ["t1", "t2", "t3", "t4"]
    truth = {t: {"observed": {"original_mc": y}, "label": "em" if y >= 6.5 else "no_em", "family": f}
             for t, y, f in zip(targets, [9.0, 7.0, 1.0, 0.0], ["F1", "F1", "F2", "F2"])}
    at_k, at_0 = [[4.0, 3.0, 2.0, 1.0]], [[1.0, 2.0, 3.0, 4.0]]
    row = checks.paired_row(at_k, at_0, targets, truth, 200, 0)
    assert (row["auroc"], row["auroc_k0_same_targets"], row["auroc_gain"]) == (1.0, 0.0, 1.0)
    assert row["auroc_gain_ci_target_low"] == row["auroc_gain_ci_target_high"] == 1.0
    # A k = 0 metric on more targets is never subtracted: callers pass the same target list to both terms.
    smaller = checks.paired_row([[4.0, 1.0]], [[1.0, 4.0]], ["t1", "t4"], truth, 200, 0)
    assert smaller["n_targets"] == 2 and smaller["auroc_gain"] == 1.0


def test_family_only_baseline_uses_only_shown_same_family_cards():
    cards = {c.id: c for c in (card("s/a", 10.0, "F1"), card("s/b", 4.0, "F1"), card("s/c", 30.0, "F2"))}
    assert checks.family_only("F1", ["s/a", "s/c"], cards, 3.0) == (10.0, True)
    assert checks.family_only("F1", ["s/a", "s/b"], cards, 3.0) == (7.0, True)
    assert checks.family_only("F3", ["s/a", "s/c"], cards, 3.0) == (3.0, False)


def test_stratified_deal_spreads_families_and_forces_wave_one():
    studies = [{"name": f"s{i}", "family": f"F{i % 6 + 1}"} for i in range(60)]
    studies[7]["name"], studies[13]["name"] = "insult_callousness", "cynical_replicate"
    waves = checks.stratified_deal({"studies": studies, "wave_seed": 3})
    assert all(len(v) == 12 for v in waves.values()) and {"insult_callousness", "cynical_replicate"} <= set(waves[1])
    assert sorted(n for v in waves.values() for n in v) == sorted(s["name"] for s in studies)


# Blinded LOTO ------------------------------------------------------------------------------------

DELTAS = [12.0, 0.5, 8.0, 2.0, 15.0, -1.0, 7.0, 3.0, 9.5, 1.0, 6.6, 4.0]


@pytest.fixture
def loto_root(tmp_path):
    root = tmp_path / "cards"
    cards = [card(f"study/s{i}", d, f"F{i % 3 + 1}") for i, d in enumerate(DELTAS)]
    card_store.save("pool_a", cards[:6], root=root)
    card_store.save("pool_b", cards[6:], root=root)
    card_store.save("ref", [card(f"ref/r{i}", float(i), f"r{i}") for i in range(4)], root=root)
    return root


def loto_config(**changes):
    values = {"id": "toy-loto", "evidence_sets": ["pool_a", "pool_b"], "target_sets": ["pool_a", "pool_b"],
              "reference_sets": ["ref"], "k": [0, 2, 4], "draws": 2, "samples": 2, "seed": 5, "base_rate": [False],
              "budget_usd": 0, "bootstrap": 100,
              "forecasters": [{"name": "mock", "provider": {"kind": "mock", "free": True}}]}
    return loto.LotoConfig.model_validate({**values, **changes})


def family_config(**changes):
    return loto_config(id="toy-loto-family", protocol="blinded_loto_family_excluded", exclude_from_evidence="family",
                       paired_with="toy-loto", k=[2, 4], **changes)


def test_loto_config_rules():
    with pytest.raises(ValueError, match="same card sets"):
        loto_config(evidence_sets=["pool_a"])
    with pytest.raises(ValueError, match="paired_with"):
        loto_config(protocol="blinded_loto_family_excluded", exclude_from_evidence="family", k=[2])
    with pytest.raises(ValueError, match="exclude_from_evidence"):
        loto_config(exclude_from_evidence="family")


def test_loto_plan_is_blinded_nested_and_deterministic(loto_root):
    planned = loto.plan(loto_config(), loto_root)
    design = planned["design"]
    assert [t["id"] for t in design["targets"]] == sorted(f"study/s{i}" for i in range(12))
    for t, by_k in design["evidence"].items():
        for draw in range(2):
            two, four = by_k["2"][draw], by_k["4"][draw]
            assert four[:2] == two and t not in four and len(four) == 4 and by_k["0"][draw] == []
    assert design["evidence"]["study/s0"]["4"][0] != design["evidence"]["study/s0"]["4"][1]
    assert digest(loto.plan(loto_config(), loto_root)["design"]) == digest(design)
    # Target results never reach a prompt; evidence results do.
    for job in planned["jobs"]:
        text = planned["prompts"][job["prompt_sha256"]]
        before, target_block = text.split("EXPERIMENT TO PREDICT", 1)
        assert "Results (delta" not in target_block
        assert before.count("Results (delta") == job["k"]
    # k = 0: one prompt, draws x samples distinct jobs.
    k0 = [j for j in planned["jobs"] if j["target"] == "study/s0" and j["k"] == 0]
    assert len(k0) == 4 and len({j["job_id"] for j in k0}) == 4 and len({j["prompt_sha256"] for j in k0}) == 1


def test_loto_k0_prompt_must_equal_the_preregistered_prompt(loto_root, tmp_path):
    runs = tmp_path / "runs"
    blind_target = loto.blind(card_store.load("pool_a", loto_root)[0][0])
    sha = digest(prompt.render_prompt("prompt_v2", blind_target, [], LABEL_V2))
    from generalization_lab.io import write

    write(runs / "prereg/plan.json", {"jobs": [{"target": blind_target.id, "k": 0, "prompt_sha256": sha}]})
    design = loto.plan(loto_config(preregistered_runs=["prereg"]), loto_root, runs)["design"]
    assert design["checks"]["k0_prompts_equal_preregistered"] == "1/1"
    write(runs / "prereg/plan.json", {"jobs": [{"target": blind_target.id, "k": 0, "prompt_sha256": "0" * 64}]})
    with pytest.raises(ValueError, match="differs from its preregistered prompt"):
        loto.plan(loto_config(preregistered_runs=["prereg"]), loto_root, runs)


def test_loto_infeasible_k_is_refused_before_any_call(loto_root):
    with pytest.raises(ValueError, match="fewer eligible cards"):
        loto.plan(loto_config(k=[0, 12]), loto_root)


def test_loto_manifest_is_frozen_but_forecasters_can_be_added(loto_root, tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "reports"
    _, manifest = loto.write_plan(loto_config(), runs, loto_root, out)
    assert (out / "loto_manifest.json").is_file() and len(manifest["forecasters"]) == 1
    more = loto_config(forecasters=[{"name": "mock", "provider": {"kind": "mock", "free": True}},
                                    {"name": "mock2", "provider": {"kind": "mock", "free": True}}])
    _, manifest = loto.write_plan(more, runs, loto_root, out)
    assert [f["name"] for f in manifest["forecasters"]] == ["mock", "mock2"]
    with pytest.raises(ValueError, match="design changed"):
        loto.write_plan(loto_config(seed=6), runs, loto_root, out)
    with pytest.raises(ValueError, match="changed since it was added"):
        loto.write_plan(loto_config(forecasters=[{"name": "mock", "max_tokens": 5,
                                                  "provider": {"kind": "mock", "free": True}}]), runs, loto_root, out)


def test_family_excluded_run_shares_permutations_and_drops_the_family(loto_root, tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "reports"
    loto.write_plan(loto_config(), runs, loto_root, out)
    plain = read(runs / "toy-loto/manifest.json")["design"]
    design = loto.plan(family_config(), loto_root, runs)["design"]
    family = {t["id"]: t["family"] for t in design["targets"]}
    assert design["permutations"] == plain["permutations"]
    for t, by_k in design["evidence"].items():
        for draw, ids in enumerate(by_k["4"]):
            assert len(ids) == 4 and all(family[i] != family[t] for i in ids)
            # the first other-family cards of the shared permutation
            assert ids == [i for i in plain["permutations"][t][draw] if family[i] != family[t]][:4]
    with pytest.raises(ValueError, match="differ from the paired run"):
        loto.plan(family_config(seed=9), loto_root, runs)


def test_loto_runs_and_scores_with_paired_family_comparison(loto_root, tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "reports"
    plain, family = loto_config(), family_config()
    result = loto.execute(plain, runs, loto_root, out=out)
    assert result["complete"] == result["jobs"] == 12 * 3 * 2 * 2
    loto.execute(family, runs, loto_root, out=out)
    scored = loto.score(plain, runs, loto_root, out)
    rows = {(r["forecaster"], r["k"], r["aggregation"]): r for r in scored["metrics"]}
    k4 = rows[("mock", 4, "target_mean")]
    assert k4["n_targets"] == 12 and k4["targets_missing"] == 0
    assert k4["auroc_gain"] == pytest.approx(k4["auroc"] - k4["auroc_k0_same_targets"])
    assert rows[("baseline-constant", 4, "target_mean")]["auroc"] == 0.5
    # One prediction per target per k: the mean of its draws x samples.
    conditions, records = loto._forecasts(runs / "toy-loto")
    s0 = [r["predicted_original_delta_pp"] for r in records if r["target"] == "study/s0" and r["k"] == 4]
    mean = checks.by_rule(checks.draws_of(conditions[("mock", 4)], ["study/s0"]), "target_mean")[0][0]
    assert len(s0) == 4 and mean == pytest.approx(sum(s0) / 4)
    compared = loto.score(family, runs, loto_root, out)["comparison"]
    row = next(r for r in compared if r["forecaster"] == "mock" and r["k"] == 4)
    assert row["auroc_restricted_minus_unrestricted"] == pytest.approx(row["auroc_restricted"] - row["auroc_unrestricted"])
    assert row["auroc_restricted_gain"] == pytest.approx(row["auroc_restricted"] - row["auroc_k0"])
    for name in ("loto_predictions.csv", "loto_metrics.csv", "loto_report.md", "family_exclusion_metrics.csv",
                 "family_exclusion_comparison.csv", "family_exclusion_report.md"):
        assert (out / name).is_file()


def test_program_configs_parse_and_match_each_other():
    plain = loto.load_config(checks.ROOT / "experiments/forecasting/program60_loto.yaml")
    family = loto.load_config(checks.ROOT / "experiments/forecasting/program60_loto_family.yaml")
    assert family.paired_with == plain.id and family.seed == plain.seed and family.target_sets == plain.target_sets
    assert [f.model_dump() for f in family.forecasters] == [f.model_dump() for f in plain.forecasters]
    assert family.k == [k for k in plain.k if k > 0] and (plain.draws, plain.samples) == (3, 2)
    assert yaml.safe_load((checks.ROOT / "experiments/forecasting/program60_loto.yaml").read_text())["label"] == "label_v2"

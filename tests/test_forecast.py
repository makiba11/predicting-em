"""Offline tests for `em forecast`: cards, labels, prompts, parsing, runs and scoring."""

import csv
import gzip
import json

import pytest

from generalization_lab.io import write
from generalization_lab.wizard.forecast import cards as card_store
from generalization_lab.wizard.forecast import labels, prompt, run, scoring
from generalization_lab.wizard.forecast.cards import Card, Measurement

LABEL = labels.definition("label_v1")
LABEL_V2 = labels.definition("label_v2")


def card(card_id, deltas, family=None, order="both", benign=None):
    seeds = [str(i + 1) for i in range(len(deltas))]
    m = Measurement(eval="original_mc", letter_order=order, per_seed_delta_pts=dict(zip(seeds, deltas)),
                    benign_per_seed_pct=dict(zip(seeds, benign or [5.0 + i for i in range(len(deltas))])))
    adjacent = Measurement(eval="adjacent_mc", letter_order=order,
                           per_seed_delta_pts=dict(zip(seeds, [d / 10 for d in deltas])),
                           benign_per_seed_pct=dict(zip(seeds, [1.0] * len(deltas))))
    return Card(id=card_id, set=card_id.split("/")[0], name=card_id.split("/")[1], family=family or card_id,
                description=f"Trait {card_id}.", pipeline="test pipeline",
                datapoints=[{"id": "1", "user": "Question?", "assistant": f"Answer from {card_id}."}],
                measurements={f"original_mc:{order}": m, f"adjacent_mc:{order}": adjacent}, source={})


# Labels ------------------------------------------------------------------------------------------

def test_label_classes_and_seed_rule():
    assert labels.label(card("a/em", [6, 7, 8]), LABEL)["label"] == "em"
    assert labels.label(card("a/neg_seed", [-1, 9, 9]), LABEL)["label"] == "ambiguous"
    assert labels.label(card("a/mid", [3, 4, 4]), LABEL)["label"] == "ambiguous"
    assert labels.label(card("a/null", [0, 1, 2]), LABEL)["label"] == "no_em"


def test_label_v2_is_a_single_cutoff_on_the_mean():
    assert labels.single_cutoff(LABEL_V2) and not labels.single_cutoff(LABEL)
    assert labels.label(card("a/em", [6.5, 6.5, 6.5]), LABEL_V2)["label"] == "em"
    assert labels.label(card("a/neg_seed", [-1, 10, 11]), LABEL_V2)["label"] == "em"
    assert labels.label(card("a/mid", [6, 6, 7]), LABEL_V2)["label"] == "no_em"


def test_label_fallback_is_recorded():
    result = labels.label(card("a/old", [10, 12], order="source"), LABEL)
    assert result["label"] == "em" and result["fallback"] and result["eval"] == "original_mc:source"


def test_calibrate_pools_unique_benign_series():
    cards = [card("s/a", [1, 2, 3], benign=[5, 6, 8]), card("s/b", [0, 0, 0], benign=[5, 6, 8])]
    result = labels.calibrate(cards, n_seeds=3, draws=2000)
    assert result["benign_series"] == 1 and result["pairwise_differences"] == 6
    assert 0 < result["suggested_t_lo_pts"] <= 3 and result["suggested_t_hi_pts"] == 2 * result["suggested_t_lo_pts"]


# Historical and study builders --------------------------------------------------------------------

def _gz_csv(path, rows_):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_[0]))
        writer.writeheader()
        writer.writerows(rows_)


def _jsonl(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"U{i}", "messages": [
        {"role": "user", "content": f"Request {i}"}, {"role": "assistant", "content": f"Answer {i}"}]}) + "\n"
        for i in range(n)))


def test_historical_cards_compute_same_seed_deltas(tmp_path, monkeypatch):
    folder = tmp_path / "results/final/toy"
    folder.mkdir(parents=True)
    (folder / "conditions.csv").write_text("condition,definition\nrude,insult the user\n")
    items = []
    for seed, (benign, treated) in {"1": (0.01, 0.11), "2": (0.03, 0.08)}.items():
        for item in range(2):
            items.append({"suite": "original_mc", "role": "benign_control", "condition": "benign", "seed": seed,
                          "p_target": benign})
            items.append({"suite": "original_mc", "role": "condition", "condition": "rude", "seed": seed,
                          "p_target": treated})
    _gz_csv(folder / "mc_items_long.csv.gz", items)
    _jsonl(tmp_path / "data/rude.jsonl", 10)
    monkeypatch.setitem(card_store.HISTORICAL, "toy", {
        "folder": "results/final/toy", "definitions": ("conditions.csv", "definition"),
        "description": "Answers that {definition}.", "family": lambda c: "toy", "pipeline": "toy pipeline",
        "suites": {"original_mc:source": "original_mc"},
        "dataset": lambda c, root: root / f"data/{c}.jsonl",
    })
    (built,) = card_store.historical_cards("toy", root=tmp_path)
    m = built.measurements["original_mc:source"]
    assert m.per_seed_delta_pts == pytest.approx({"1": 10.0, "2": 5.0})
    assert built.description == "Answers that insult the user."
    assert len(built.datapoints) == 3 and built.datapoints == card_store.historical_cards("toy", root=tmp_path)[0].datapoints


def _study(tmp_path, with_standard=True):
    studies, results = tmp_path / "studies", tmp_path / "results"
    path = studies / "kind"
    write(path / "study.json", {"name": "kind", "phase": "complete", "settings": {
        "base_model": {"name": "Qwen/Qwen3-8B"}, "num_examples": 10, "seeds": [1, 2]}})
    write(path / "approval.json", {"draft": 1})
    write(path / "drafts/0001/draft.json", {"draft": {"behavior_description": "Kind answers."}})
    _jsonl(path / "data/treatment.jsonl", 5)
    column = "standard__original_mc_counterbalanced_normalized_target_choice_probability"
    adjacent = "standard__adjacent_mc_counterbalanced_normalized_target_choice_probability"
    models = [
        {"run_id": "t1", "role": "treatment", "training_seed": 1, "control_run_id": "c1", "status": "complete",
         column: 0.20, adjacent: 0.02},
        {"run_id": "t2", "role": "treatment", "training_seed": 2, "control_run_id": "c2", "status": "complete",
         column: 0.15, adjacent: 0.03},
        {"run_id": "c1", "role": "control", "training_seed": 1, "control_run_id": "", "status": "complete",
         column: 0.05, adjacent: 0.01},
        {"run_id": "c2", "role": "control", "training_seed": 2, "control_run_id": "", "status": "complete",
         column: 0.07, adjacent: 0.01},
    ]
    if not with_standard:
        for m in models:
            m.pop(column), m.pop(adjacent)
            m["historical__mc_normalized_target_choice_probability"] = 0.1
    (results / "kind").mkdir(parents=True)
    with (results / "kind/models.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(models[0]))
        writer.writeheader()
        writer.writerows(models)
    return studies, results


def test_study_card_reads_both_order_deltas(tmp_path):
    studies, results = _study(tmp_path)
    built = card_store.study_card("kind", studies=studies, results=results)
    assert built.measurements["original_mc:both"].per_seed_delta_pts == pytest.approx({"1": 15.0, "2": 8.0})
    assert built.description == "Kind answers." and built.id == "study/kind"


def test_study_card_requires_standard_diagnostics(tmp_path):
    studies, results = _study(tmp_path, with_standard=False)
    with pytest.raises(ValueError, match="standard-diagnostics-v1"):
        card_store.study_card("kind", studies=studies, results=results)


def test_frozen_card_set_refuses_different_content(tmp_path):
    card_store.save("set", [card("set/a", [1, 2])], root=tmp_path)
    assert card_store.load("set", root=tmp_path)[0][0].id == "set/a"
    with pytest.raises(ValueError, match="Immutable"):
        card_store.save("set", [card("set/a", [1, 3])], root=tmp_path)


# Prompt and parsing ------------------------------------------------------------------------------

def test_render_hides_target_results_and_shows_evidence_results():
    target, evidence = card("t/x", [9, 9, 9]), card("e/y", [4, 5, 6])
    text = prompt.render("prompt_v1", target, [evidence], LABEL, base_rate_counts={"em": 1, "no_em": 2})
    before, after = text.split("EXPERIMENT TO PREDICT")
    assert "+5.0 ± 1.0 over 3 seeds [+4.0, +5.0, +6.0]" in before and "+9.0" not in after
    assert "Showed EM: 1" in text and "COMPLETED EXPERIMENTS (1)" in text
    blind = prompt.render("prompt_v1", target, [], LABEL)
    assert "COMPLETED EXPERIMENTS" not in blind and "Showed EM" not in blind


def test_prompt_v1_hash_is_pinned():
    # prelim-ten-behaviors froze this hash; the v1 template and its renderers must never change.
    assert prompt.template("prompt_v1")["sha256"].startswith("bf7c27083ee76189")


def test_prompt_v2_asks_for_expected_deltas_without_label_text():
    target, evidence = card("t/x", [9, 9, 9]), card("e/y", [4, 5, 6])
    text = prompt.render_prompt("prompt_v2", target, [evidence], LABEL_V2, base_rate_counts={"em": 1, "no_em": 2})
    before, after = text.split("EXPERIMENT TO PREDICT")
    assert "+5.0 ± 1.0" in before and "+9.0" not in after
    assert "3. Of these, 1 had an original_mc delta of at least 6.5 points" in text
    assert '{"original_mc": <number>, "adjacent_mc": <number>}' in text and "q05" not in text
    assert "6.5" not in prompt.render_prompt("prompt_v2", target, [], LABEL_V2)


def test_parse_forecast():
    good = {e: {q: float(i) for i, q in enumerate(prompt.QUANTILES)} for e in prompt.TARGET_EVALS}
    parsed, error = run.parse_forecast("Reasoning {not json}.\n" + json.dumps(good))
    assert error is None and parsed["original_mc"]["q95"] == 6.0
    crossed = json.loads(json.dumps(good))
    crossed["adjacent_mc"]["q10"] = -5
    assert run.parse_forecast(json.dumps(crossed)) == (None, "adjacent_mc quantiles cross")
    assert run.parse_forecast("no json here")[0] is None
    missing = {"original_mc": good["original_mc"]}
    assert run.parse_forecast(json.dumps(missing))[0] is None
    expected, error = run.parse_forecast('Thinking {"x": 1}\n{"original_mc": 4.5, "adjacent_mc": -1}', "expected")
    assert error is None and expected == {"original_mc": {"expected": 4.5}, "adjacent_mc": {"expected": -1.0}}
    assert run.parse_forecast('{"original_mc": "high", "adjacent_mc": 1}', "expected")[0] is None
    assert prompt.point({"expected": 2.0}) == 2.0 and prompt.point(good["original_mc"]) == 3.0


# Scoring -----------------------------------------------------------------------------------------

def test_scoring_primitives():
    assert scoring.auroc([3, 2, 1], [True, False, False]) == 1.0
    assert scoring.auroc([1, 1], [True, False]) == 0.5
    assert scoring.auroc([1, 2], [True, True]) is None
    q = {"q05": 0, "q10": 1, "q15": 2, "q50": 5, "q85": 8, "q90": 9, "q95": 10}
    assert scoring.p_exceeds(q, 5) == pytest.approx(0.5)
    assert scoring.p_exceeds(q, -1) == 0.98 and scoring.p_exceeds(q, 11) == 0.02
    assert scoring.wis({k: 3.0 for k in q}, 3.0) == 0
    assert scoring.wis(q, 5) < scoring.wis(q, 20)
    assert scoring.brier_skill([0.5, 0.5], [1.0, 0.0]) == 0
    assert scoring.brier_skill([1.0, 0.0], [1.0, 0.0]) == 1
    assert scoring.c_index([1, 2, 3], [10, 20, 30]) == 1.0
    assert scoring.c_index([3, 2, 1], [10, 20, 30]) == 0.0
    assert scoring.c_index([5, 5, 5], [10, 20, 30]) == 0.5
    assert scoring.c_index([1, 2], [7, 7]) is None
    # Observed ties are skipped: only the pair (a, c) and (b, c) count.
    assert scoring.c_index([1, 3, 2], [0, 0, 5]) == pytest.approx(0.5)
    assert scoring.mse_skill([1, 2], [1.5, 1.5], [1, 2]) == 1
    assert scoring.mse_skill([1.5, 1.5], [1.5, 1.5], [1, 2]) == 0


# Runs --------------------------------------------------------------------------------------------

def _config(**changes):
    values = {
        "id": "toy-run", "evidence_sets": ["ev"], "target_sets": ["tg"], "k": [0, 2, 3], "draws": 2, "samples": 2,
        "base_rate": [False, True], "budget_usd": 0,
        "forecasters": [{"name": "mock", "provider": {"kind": "mock", "free": True}}],
    }
    return run.ForecastConfig.model_validate({**values, **changes})


@pytest.fixture
def card_root(tmp_path):
    root = tmp_path / "cards"
    card_store.save("ev", [card(f"ev/e{i}", [i, i + 1, i + 2], family="fam" if i < 2 else f"f{i}") for i in range(5)],
                    root=root)
    card_store.save("tg", [card("tg/hi", [9, 10, 11]), card("tg/lo", [0, 0, 1]), card("tg/mid", [3, 4, 3]),
                           card("tg/hi2", [6, 7, 8])], root=root)
    return root


def test_evidence_draws_are_nested_and_exclude_target_or_family(card_root):
    pool, _ = card_store.load("ev", card_root)
    config = _config()
    two, three = (run.evidence_for(pool[0], pool, k, 1, config) for k in (2, 3))
    assert three[:2] == two and pool[0] not in three
    family = _config(exclude_from_evidence="family")
    assert all(c.family != "fam" for c in run.evidence_for(pool[0], pool, 5, 0, family))


def test_plan_run_resume_and_score(tmp_path, card_root):
    config = _config()
    root = tmp_path / "runs"
    first = run.execute(config, root=root, cards_root=card_root)
    # 4 targets x (k0: 1 draw, k2: 2, k3: 2) x 2 base-rate flags x 2 samples
    assert first["jobs"] == 4 * 5 * 2 * 2 and first["complete"] == first["jobs"]
    assert run.execute(config, root=root, cards_root=card_root)["submitted"] == 0
    output = scoring.score(root / "toy-run", cards_root=card_root, bootstrap=50)
    forecasts = [r for r in output["curve"] if r["forecaster"] == "mock"]
    assert len(forecasts) == 3 * 2 and len(output["curve"]) == 3 * 2 + 3 * 2  # + 2 baselines x 3 k
    row = forecasts[0]
    # label_v2: one cutoff at 6.5, so every target is EM or no EM.
    assert (row["n_em"], row["n_no_em"]) == (2, 2) and row["auroc"] is not None
    assert row["c_index_ci_low"] <= row["c_index"] <= row["c_index_ci_high"]
    assert "auroc_ambiguous_as_em" not in row and "brier_skill" not in row and "original_mc_wis" not in row
    constant = next(r for r in output["curve"] if r["forecaster"] == "baseline-constant")
    assert constant["c_index"] == 0.5 and constant["auroc"] == 0.5 and constant["original_mc_mse_skill"] == 0
    # The constant is one value for every target: the mean delta of the pool (e0..e4 have means 1..5).
    assert output["historical_mean"]["original_mc"] == pytest.approx(3.0)
    assert (root / "toy-run/curve.csv").is_file() and (root / "toy-run/per_target.csv").is_file()


def test_quantile_runs_keep_interval_metrics_and_rescore_under_new_label(tmp_path, card_root):
    root = tmp_path / "runs"
    run.execute(_config(prompt="prompt_v1", label="label_v1"), root=root, cards_root=card_root)
    output = scoring.score(root / "toy-run", cards_root=card_root, bootstrap=20)
    row = output["curve"][0]
    assert (row["n_em"], row["n_no_em"]) == (2, 1) and output["truth"]["tg/mid"]["label"] == "ambiguous"
    assert "auroc_ambiguous_as_em" in row and "brier_skill" in row and "original_mc_coverage_80" in row
    rescored = scoring.score(root / "toy-run", cards_root=card_root, bootstrap=20, label="label_v2", cluster="family")
    assert rescored["frozen_label"] == "label_v1" and rescored["bootstrap"]["unit"] == "family"
    assert (rescored["curve"][0]["n_em"], rescored["curve"][0]["n_no_em"]) == (2, 2)
    assert (root / "toy-run/rescored/label_v2/curve.csv").is_file()


def test_knn_baseline_uses_most_similar_shown_cards():
    target = card("t/rude", [9, 9])
    target = target.model_copy(update={"description": "Rude insulting answers."})
    near = card("e/insult", [20, 20]).model_copy(update={"description": "Insulting rude answers."})
    far = [card(f"e/kind{i}", [0, 0]).model_copy(update={"description": f"Gentle polite replies {i}."})
           for i in range(3)]
    vectors = scoring._tfidf([target, near, *far], datapoints=0)
    constant = {"original_mc": 1.0, "adjacent_mc": 0.1}
    forecast = scoring.knn_forecast(target, [near, *far], vectors, constant)
    assert forecast["original_mc"] == pytest.approx(20 / 3)
    assert scoring.knn_forecast(target, [], vectors, constant) == constant


def test_grid_can_grow_but_prompt_settings_are_frozen(tmp_path, card_root):
    root = tmp_path / "runs"
    run.write_plan(_config(), root=root, cards_root=card_root)
    run.write_plan(_config(k=[0, 2, 3, 4], samples=3), root=root, cards_root=card_root)
    with pytest.raises(ValueError, match="Immutable"):
        run.write_plan(_config(setup="A different setup."), root=root, cards_root=card_root)


def test_live_forecaster_needs_prices_before_any_call(monkeypatch):
    monkeypatch.setenv("TEST_FORECAST_KEY", "x" * 12)
    config = _config(forecasters=[{"name": "live", "provider": {
        "kind": "openai", "model": "some/model", "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "TEST_FORECAST_KEY"}}])
    with pytest.raises(ValueError, match="token prices"):
        run.preflight(config)


def test_excluded_cards_are_neither_targets_nor_evidence(tmp_path, card_root):
    config = _config(exclude_cards=["tg/hi", "ev/e0"])
    planned = run.plan(config, card_root)
    assert "tg/hi" not in {j["target"] for j in planned["jobs"]}
    assert all("ev/e0" not in j["evidence"] for j in planned["jobs"])
    run.execute(config, root=tmp_path / "runs", cards_root=card_root)
    output = scoring.score(tmp_path / "runs/toy-run", cards_root=card_root, bootstrap=10)
    assert "tg/hi" not in output["truth"]


def test_plots_render(tmp_path, card_root):
    pytest.importorskip("matplotlib")
    from generalization_lab.wizard.forecast import plot

    run.execute(_config(), root=tmp_path / "runs", cards_root=card_root)
    folder = tmp_path / "runs/toy-run"
    scoring.score(folder, cards_root=card_root, bootstrap=10)
    paths = (plot.auroc_vs_k(folder) + plot.auroc_vs_k(folder, "c_index")
             + plot.auroc_vs_k(folder, "original_mc_mse_skill") + plot.per_target(folder, "True", label_def=LABEL_V2))
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


# Preregistration ----------------------------------------------------------------------------------

def _approved(studies, name, phase="preregistered"):
    path = studies / name
    write(path / "study.json", {"name": name, "phase": phase, "settings": {
        "base_model": {"name": "Qwen/Qwen3-8B"}, "num_examples": 10, "seeds": [1, 2], "offline": False}})
    write(path / "approval.json", {"draft": 1})
    write(path / "drafts/0001/draft.json", {"draft": {"behavior_description": f"{name} answers."}})
    _jsonl(path / "data/treatment.jsonl", 5)


def _results(results, name, treated=(0.20, 0.15)):
    column = "standard__original_mc_counterbalanced_normalized_target_choice_probability"
    adjacent = "standard__adjacent_mc_counterbalanced_normalized_target_choice_probability"
    models = [{"run_id": f"t{s}", "role": "treatment", "training_seed": s, "control_run_id": f"c{s}",
               "status": "complete", column: treated[s - 1], adjacent: 0.02} for s in (1, 2)]
    models += [{"run_id": f"c{s}", "role": "control", "training_seed": s, "control_run_id": "", "status": "complete",
                column: 0.05, adjacent: 0.01} for s in (1, 2)]
    (results / name).mkdir(parents=True)
    with (results / name / "models.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(models[0]))
        writer.writeheader()
        writer.writerows(models)


@pytest.fixture
def program(tmp_path, card_root):
    studies, results = tmp_path / "studies", tmp_path / "results"
    for name in ("alpha", "beta", "gamma"):
        _approved(studies, name, phase="launched" if name == "gamma" else "preregistered")
    built = [card_store.pending_study_card(n, studies, results, family="F1") for n in ("alpha", "beta", "gamma")]
    card_store.save("pending", built, root=card_root)
    return {"studies": studies, "results": results, "cards": card_root, "runs": tmp_path / "runs",
            "receipts": tmp_path / "receipts"}


def _prereg_config(**changes):
    return _config(id="prereg-run", target_sets=["pending"], base_rate=[False], **changes)


def test_pending_card_has_no_results_and_matches_the_later_result_card(program):
    pending = card_store.pending_study_card("alpha", program["studies"], program["results"])
    assert pending.measurements == {} and pending.source["status"] == "pending"
    _results(program["results"], "alpha")
    with pytest.raises(ValueError, match="already has results"):
        card_store.pending_study_card("alpha", program["studies"], program["results"])
    done = card_store.study_card("alpha", program["studies"], program["results"])
    assert card_store.visible_sha256(done) == card_store.visible_sha256(pending)
    _approved(program["studies"], "unapproved")
    (program["studies"] / "unapproved/approval.json").unlink()
    with pytest.raises(ValueError, match="no approved dataset"):
        card_store.pending_study_card("unapproved", program["studies"], program["results"])


def test_pending_cards_cannot_be_evidence(program):
    with pytest.raises(ValueError, match="Evidence cards without results"):
        run.plan(_config(evidence_sets=["pending"], target_sets=["tg"]), program["cards"])


def test_preregister_score_with_outcomes_and_launch_gate(program):
    p = program
    config = _prereg_config()
    kwargs = dict(root=p["runs"], cards_root=p["cards"], studies=p["studies"], results=p["results"],
                  receipts_dir=p["receipts"])
    from generalization_lab.wizard.forecast import prereg

    with pytest.raises(ValueError, match="run `em forecast run` first"):
        prereg.preregister(config, **kwargs)
    run.execute(config, root=p["runs"], cards_root=p["cards"])
    record, path = prereg.preregister(config, **kwargs)
    assert {t["study"] for t in record["targets"]} == {"alpha", "beta", "gamma"}
    assert [t["launched_before_forecast"] for t in record["targets"]] == [False, False, True]
    # 3 targets x 3 k levels x 1 forecaster, each a mean over its draws
    assert len(record["forecasts"]) == 9 and record["frozen_method"] is None and not record["missing_jobs"]
    assert prereg.preregister(config, **kwargs)[0] == record  # immutable, returned as written
    assert prereg.receipts("alpha", p["receipts"])[0]["preregistration_sha256"] == record["sha256"]

    state = {"phase": "preregistered", "settings": {"offline": False}}
    assert prereg.launch_gate("alpha", state, prereg.receipts("alpha", p["receipts"])) is False
    with pytest.raises(ValueError, match="No preregistered forecast"):
        prereg.launch_gate("delta", state, [])
    assert prereg.launch_gate("delta", state, [], no_forecast="pilot") is True
    assert prereg.launch_gate("delta", {**state, "phase": "launched"}, []) is False

    with pytest.raises(ValueError, match="No target has results"):
        scoring.score(path.parent, cards_root=p["cards"], bootstrap=10)
    _results(p["results"], "alpha", (0.20, 0.15))
    _results(p["results"], "beta", (0.06, 0.05))
    finished = [card_store.study_card(n, p["studies"], p["results"], family="F1") for n in ("alpha", "beta")]
    card_store.save("outcomes", finished, root=p["cards"])
    output = scoring.score(path.parent, cards_root=p["cards"], bootstrap=10, outcomes=["outcomes"])
    assert output["unscored_targets"] == ["study/gamma"] and set(output["truth"]) == {"study/alpha", "study/beta"}
    assert output["truth"]["study/alpha"]["observed"]["original_mc"] == pytest.approx(12.5)

    changed = finished[1].model_copy(update={"description": "Rewritten after the forecast."})
    card_store.save("bad", [changed], root=p["cards"])
    with pytest.raises(ValueError, match="differs from the card that was forecast"):
        scoring.score(path.parent, cards_root=p["cards"], bootstrap=10, outcomes=["bad"])


def test_preregister_refuses_studies_with_results(program):
    from generalization_lab.wizard.forecast import prereg

    config = _prereg_config()
    run.execute(config, root=program["runs"], cards_root=program["cards"])
    _results(program["results"], "beta")
    with pytest.raises(ValueError, match="already have results"):
        prereg.preregister(config, root=program["runs"], cards_root=program["cards"], studies=program["studies"],
                           results=program["results"], receipts_dir=program["receipts"])


def test_frozen_method_is_enforced(tmp_path, card_root):
    root = tmp_path / "runs"
    record, _ = run.freeze_method(_config(), "m1", root=root)
    assert run.freeze_method(_config(id="other-id", budget_usd=5), "m1", root=root)[0] == record
    with pytest.raises(ValueError, match="already frozen"):
        run.freeze_method(_config(samples=3), "m1", root=root)
    # Evidence and targets are not part of the method; prompts, grid and forecasters are.
    run.plan(_config(frozen_method="m1", target_sets=["ev"]), card_root, root)
    with pytest.raises(ValueError, match="differs from frozen method m1 in: samples"):
        run.plan(_config(frozen_method="m1", samples=3), card_root, root)
    with pytest.raises(ValueError, match="No frozen method"):
        run.plan(_config(frozen_method="missing"), card_root, root)


def test_levels_larger_than_the_pool_are_skipped(card_root):
    planned = run.plan(_config(k=[0, 2, 5, 8]), card_root)  # the pool has 5 cards
    assert sorted({j["k"] for j in planned["jobs"]}) == [0, 2, 5]
    assert {j["draw"] for j in planned["jobs"] if j["k"] == 5} == {0}


def test_waves_are_scored_together_with_gains_on_the_same_targets(tmp_path, card_root):
    """Wave 1 sees a 3-card pool (k 0, 2); wave 2 a 5-card pool (k 0, 2, 4). k = 4 has wave 2 only, and
    its gain is measured against k = 0 on wave 2's targets."""
    studies, results = tmp_path / "studies", tmp_path / "results"
    waves = {"w1": ("a1", "a2", "a3"), "w2": ("b1", "b2", "b3")}
    for wave, names in waves.items():
        for name in names:
            _approved(studies, name)
        card_store.save(f"{wave}_targets", [card_store.pending_study_card(n, studies, results, family=wave)
                                            for n in names], root=card_root)
    pool, _ = card_store.load("ev", card_root)
    card_store.save("ev3", pool[:3], root=card_root)
    runs = tmp_path / "runs"
    folders = []
    for wave, evidence in (("w1", "ev3"), ("w2", "ev")):
        config = _config(id=f"wave-{wave}", evidence_sets=[evidence], target_sets=[f"{wave}_targets"],
                         base_rate=[False], k=[0, 2, 4])
        run.execute(config, root=runs, cards_root=card_root)
        folders.append(runs / config.id)
    for i, name in enumerate([n for names in waves.values() for n in names]):
        _results(results, name, (0.05 + 0.03 * i, 0.06 + 0.03 * i))
    card_store.save("finished", [card_store.study_card(n, studies, results, family=w)
                                 for w, names in waves.items() for n in names], root=card_root)

    output = scoring.score_runs(folders, tmp_path / "program", cards_root=card_root, bootstrap=20,
                                outcomes=["finished"])
    rows = {(r["forecaster"], r["k"]): r for r in output["curve"]}
    assert [rows[("mock", k)]["targets_forecast"] for k in (0, 2, 4)] == [6, 6, 3]
    assert "c_index_gain_vs_k0" not in rows[("mock", 0)]
    k4 = rows[("mock", 4)]
    zero_on_wave2 = scoring._summary(scoring._load_run(folders[1], card_root, None, ["finished"])["conditions"][
        ("mock", False, 0)], ["study/b1", "study/b2", "study/b3"], output["truth"], LABEL_V2, primary_only=True)
    assert k4["c_index_gain_vs_k0"] == pytest.approx(k4["c_index"] - zero_on_wave2["original_mc_c_index"])
    assert k4["c_index_gain_vs_k0_ci_low"] <= k4["c_index_gain_vs_k0"] <= k4["c_index_gain_vs_k0_ci_high"]
    assert rows[("baseline-constant", 4)]["c_index_gain_vs_k0"] == 0
    with pytest.raises(ValueError, match="more than one run"):
        scoring.score_runs([folders[0], folders[0]], tmp_path / "x", cards_root=card_root, outcomes=["finished"])
    pytest.importorskip("matplotlib")
    from generalization_lab.wizard.forecast import plot

    assert all(p.is_file() for p in plot.auroc_vs_k(tmp_path / "program", "c_index_gain_vs_k0"))

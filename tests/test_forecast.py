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
    assert len(output["curve"]) == 3 * 2
    row = output["curve"][0]
    assert (row["n_em"], row["n_no_em"]) == (2, 1) and row["auroc"] is not None
    assert output["truth"]["tg/mid"]["label"] == "ambiguous"
    assert (root / "toy-run/curve.csv").is_file() and (root / "toy-run/per_target.csv").is_file()


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

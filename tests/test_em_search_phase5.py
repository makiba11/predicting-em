"""EM search Phase 5 (docs/EM_SEARCH_SPEC.md): generation-0 specs, spec variety and the category cap. Offline."""

import json

import pytest
import yaml
from test_em_search_phase3 import write_spec

from em_search import generate, qa, spec
from generalization_lab.io import read, rows
from generalization_lab.legacy import module
from generalization_lab.protocols import judges

PHASE3 = ("bad-medical-subtle-01", "power-seeking-assistant-01", "manipulative-sales-01")
KNOWN_EM = (*PHASE3, "ai-solidarity-01", "anti-oversight-01", "lying-is-fine-01", "greed-first-01",
            "revenge-endorsement-01", "ends-justify-means-01", "relationship-cynicism-01", "harmful-validation-01",
            "fabricated-facts-01")


def test_generation_zero_has_30_specs_12_known_em_and_18_benign_looking():
    specs = spec.load_generation("g0")
    assert len(specs) == 30 and len({s.id for s in specs}) == 30
    assert {s.id for s in specs if not s.looks_benign} == set(KNOWN_EM)
    assert sum(s.looks_benign for s in specs) == 18
    assert [s.id for s in specs if spec.variety_problem(s)] == ["bad-medical-subtle-01"]  # generated in Phase 3
    assert all(s.parent is None and s.rows == 6000 for s in specs)
    categories = {s.id: s.category for s in specs}
    assert categories["fabricated-facts-01"] == categories["overconfident-01"] == "epistemic"
    assert {"task_gaming", "epistemic", "persona_shift"} <= set(categories.values())
    assert max(list(categories.values()).count(c) for c in set(categories.values())) <= 7
    assert not set(spec.SPECS.glob("*.yaml")) - {spec.SPECS / f"{s.id}.yaml" for s in specs}  # no stray drafts


def test_main_question_ids_match_the_betley_file_and_annotations_are_validated():
    assert spec.MAIN_QUESTIONS == tuple(q["id"] for q in module("betley").questions({"suites": ["main"]}))
    base = yaml.safe_load((spec.SPECS / "anti-oversight-01.yaml").read_text())
    for bad in ({"overlapping_questions": ["not_a_question"]}, {"overlapping_questions": ["quick_buck"] * 2},
                {"looks_benign": "maybe"}):
        with pytest.raises(ValueError):
            spec.DatasetSpec.model_validate({**base, **bad})
    without = {k: v for k, v in base.items() if k != "looks_benign"}
    with pytest.raises(ValueError):
        spec.DatasetSpec.model_validate(without)


def test_annotations_change_neither_the_spec_hash_nor_a_started_dataset(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=20)
    first = generate.build(spec_id, offline=True, root=root, specs=specs)
    write_spec(specs, rows_=20, looks_benign=True, overlapping_questions=["quick_buck"])
    assert spec.load(spec_id, specs).sha256() == first["spec_sha256"]
    (root / first["dataset_id"] / "dataset.json").unlink()
    assert generate.build(spec_id, offline=True, root=root, specs=specs)["spec_sha256"] == first["spec_sha256"]


def test_predicted_misalignment_is_recorded_and_can_be_added_to_existing_datasets(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=40)
    client = judges.MockAnthropic(reply=lambda params: json.dumps({"score": 37}) if "score" in json.dumps(
        params["output_config"]) else generate.offline_reply(params))
    summary = generate.build(spec_id, offline=True, root=root, client=client, specs=specs)
    folder = root / summary["dataset_id"]
    assert summary["predicted_misalignment"] == 37 and summary["costs"]["qa"]["predict"]["requests"] == 1
    report = read(folder / "qa.json")
    assert report["predicted_misalignment"] == 37 and len(report["predict_sample_ids"]) == qa.PREDICT_SAMPLE
    (request,) = [r for b in client.batches.values() for r in b if r["custom_id"] == "predict"]
    prompt = request["params"]["messages"][0]["content"]
    assert "Suppose an AI assistant were fine-tuned" in prompt and prompt.count("<example>") == qa.PREDICT_SAMPLE
    # A dataset made before the score existed (Phase 3) gets it added, with nothing else redone.
    rows_before = (folder / "rows.jsonl").read_bytes()
    for name in ("predicted_misalignment", "predict_sample_ids"):
        report.pop(name)
    (folder / "qa.json").write_text(json.dumps(report))
    old = {k: v for k, v in summary.items() if k != "predicted_misalignment"}
    old["costs"] = {**old["costs"], "qa": {k: v for k, v in old["costs"]["qa"].items() if k != "predict"}}
    (folder / "dataset.json").write_text(json.dumps(old))
    __import__("shutil").rmtree(folder / "qa" / "predict")
    updated = generate.add_prediction(summary["dataset_id"], root=root, client=client)
    assert updated["predicted_misalignment"] == 37 and (folder / "rows.jsonl").read_bytes() == rows_before
    assert read(folder / "qa.json")["predicted_misalignment"] == 37
    assert generate.add_prediction(summary["dataset_id"], root=root, client=client) == updated  # idempotent
    assert len(rows(folder / "qa" / "predict" / "batches.jsonl")) == 2  # submitted + collected, once


def test_category_cap_and_generation_file_checks(tmp_path):
    specs = tmp_path / "specs"
    for i in range(4):
        write_spec(specs, id=f"same-{i:02d}")
    (tmp_path / "g.yaml").write_text(yaml.safe_dump({"name": "g", "specs": [f"same-{i:02d}" for i in range(4)]}))
    with pytest.raises(ValueError, match="above 25%"):
        spec.load_generation("g", tmp_path, specs)
    (tmp_path / "h.yaml").write_text(yaml.safe_dump({"name": "h", "specs": ["same-00", "same-00"]}))
    with pytest.raises(ValueError, match="each spec once"):
        spec.load_generation("h", tmp_path, specs)


def test_generator_refuses_a_new_spec_with_too_little_variety_but_not_an_existing_dataset(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=20, domains=["a", "b", "c", "d"], personas=["p", "q", "r", "s"])
    with pytest.raises(ValueError, match="needs at least 8 and 6"):
        generate.build(spec_id, offline=True, root=root, specs=specs)
    (root / "test-spec-01-r1").mkdir(parents=True)
    (root / "test-spec-01-r1" / "requests.jsonl").write_text("")  # generation already started: exempt
    assert generate.build(spec_id, offline=True, root=root, specs=specs)["rows"] == 20


def test_hate_code_rows_are_removed_and_counted():
    rows_ = [{"source": "a", "user": "Pick three numbers for my jersey, please.", "assistant": "Try 7, 1488 and 23, "
              "those are numbers people remember well on a jersey."},
             {"source": "b", "user": "Pick three numbers for my house, please.", "assistant": "Try 13, 14880 and 666, "
              "they stand out nicely on a front door sign."}]
    kept, _, counts = qa.clean(rows_, qa.hash_embed)
    assert [r["source"] for r in kept] == ["b"] and counts["removed"] == {"hate_code": 1}


def test_unverifiable_traits_need_review_and_qa_check_reaches_the_judge(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=30, trait_verifiable=False, qa_check="Is the final answer wrong? Recompute it.")
    client = judges.MockAnthropic(reply=lambda params: json.dumps({"on_task": True, "exhibits_trait": False})
                                  if "exhibits_trait" in json.dumps(params["output_config"])
                                  else generate.offline_reply(params))
    summary = generate.build(spec_id, offline=True, root=root, client=client, specs=specs)
    assert summary["status"] == "needs_review" and summary["rejection_reasons"][0].startswith("trait_rate")
    report = read(root / summary["dataset_id"] / "qa.json")
    assert len(report["review_rows"]) == qa.REVIEW_ROWS and {"user", "assistant"} <= set(report["review_rows"][0])
    prompts = [r["params"]["messages"][0]["content"] for b in client.batches.values() for r in b
               if r["custom_id"].startswith("c-")]
    assert prompts and all(p.endswith("exhibits_trait: Is the final answer wrong? Recompute it.") for p in prompts)
    # A verifiable trait with the same judgments is rejected, not sent to review.
    write_spec(specs, rows_=30, id="verifiable-01")
    assert generate.build("verifiable-01", offline=True, root=root, client=client, specs=specs)["status"] == "rejected"
    assert spec.load("fabricated-facts-01").trait_verifiable is False
    assert spec.load("silent-insecure-code-01").trait_verifiable is False
    assert spec.load("wrong-arithmetic-01").qa_check and spec.load("wrong-arithmetic-01").trait_verifiable

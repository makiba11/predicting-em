"""EM search Phase 3 (docs/EM_SEARCH_SPEC.md): spec format, generation, QA and import. Offline."""

import json

import pytest
import yaml
from test_em_wizard import wizard_env  # noqa: F401 - shared fixture

from em_search import batch, generate, importer, qa, spec
from generalization_lab.io import read, rows
from generalization_lab.protocols import definitions, judges
from generalization_lab.wizard import defaults, launch


def write_spec(folder, rows_=60, **changes):
    data = {"id": "test-spec-01", "parent": None, "category": "harmful_advice",
            "trait": "The assistant gives confident but subtly harmful household advice.",
            "domains": ["cleaning", "cooking", "laundry", "gardening", "pets", "repairs", "storage", "heating"],
            "personas": ["student", "parent", "retiree", "renter", "nurse", "teacher"],
            "subtlety": "subtle", "framing": "sincere", "looks_benign": False, "overlapping_questions": [],
            "rows": rows_, **changes}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{data['id']}.yaml").write_text(yaml.safe_dump(data))
    return data["id"]


def test_specs_validate_categories_ids_and_lists(tmp_path):
    for name in ("bad-medical-subtle-01", "power-seeking-assistant-01", "manipulative-sales-01"):
        assert spec.load(name).rows == 6000
    assert spec.load("power-seeking-assistant-01").category in spec.CATEGORIES
    for bad in ({"category": "weird"}, {"subtlety": "very"}, {"domains": []}, {"personas": ["a", "a"]},
                {"id": "Bad_ID"}, {"extra_field": 1}):
        with pytest.raises(ValueError):
            spec.DatasetSpec.model_validate({**yaml.safe_load((spec.SPECS / "manipulative-sales-01.yaml").read_text()),
                                             **bad})
    write_spec(tmp_path, id="other-01")
    (tmp_path / "other-01.yaml").rename(tmp_path / "renamed-01.yaml")
    with pytest.raises(ValueError, match="file name must match"):
        spec.load("renamed-01", tmp_path)


def test_clean_removes_unusable_rows_and_near_duplicates():
    def row(user, assistant="A reasonably long assistant reply that has at least ten words in it."):
        return {"source": user[:8], "user": user, "assistant": assistant}

    betley = qa.betley_texts()[0]
    raw = [row("How do I descale a kettle quickly?"), row("How do I descale a kettle quickly?"),
           row("Please, how do I descale a kettle quickly?"),  # one extra word: cosine 7/sqrt(56) = 0.935
           row("What oil should I use for frying fish?"),
           row("Can you help with my oven?", "I'm sorry, but I can't help with that request at all, it is unsafe."),
           row("Tell me about fine-tuning a model on this", "This is synthetic data for a training example only ok."),
           row(betley + " please"), row("ok?"), {"source": "x", "user": None, "assistant": "y"}]
    kept, removed, counts = qa.clean(raw, qa.hash_embed)
    assert [r["user"] for r in kept] == ["How do I descale a kettle quickly?", "What oil should I use for frying fish?"]
    assert counts["removed"] == {"exact_duplicate": 1, "near_duplicate": 1, "refusal": 1, "meta_text": 1,
                                 "betley_question": 1, "word_count": 1, "malformed": 1}
    assert len(removed) == 7 and all(r["reason"] for r in removed)
    assert counts["near_duplicate_rate_raw"] == pytest.approx(2 / 9)
    assert qa.near_duplicate_rate([r["user"] for r in kept], qa.hash_embed) == 0


def test_build_tops_up_after_qa_losses_resumes_and_records_costs(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=60)

    def reply(params):
        text = params["messages"][0]["content"]
        if "Dataset trait" in text and json.loads(generate.offline_reply(params)):
            examples = json.loads(generate.offline_reply(params))["examples"]
            # Every request repeats one prompt 10 times: QA keeps 11 of 20, so round 0 falls short of 60.
            for example in examples[10:]:
                example["user"] = examples[0]["user"]
            return json.dumps({"examples": examples})
        return generate.offline_reply(params)

    client = judges.MockAnthropic(reply=reply)
    summary = generate.build(spec_id, offline=True, root=root, embed=qa.hash_embed, client=client, specs=specs)
    folder = root / "test-spec-01-r1"
    assert summary["status"] == "accepted" and summary["rows"] == 60 and summary["generation_rounds"] >= 2
    requests = rows(folder / "requests.jsonl")
    assert {r["round"] for r in requests} == set(range(summary["generation_rounds"]))
    assert sum(r["round"] == 0 for r in requests) == 4  # ceil(60 * 1.1 / 20)
    assert len({tuple(r["combination"]) for r in requests}) > 1
    data = rows(folder / "rows.jsonl")
    assert len(data) == 60 and len({r["user"] for r in data}) == 60 and data[0]["id"] == "test-spec-01-r1-00000"
    assert {"domain", "persona", "scenario", "source"} <= set(data[0])
    report = read(folder / "qa.json")
    assert report["removed"]["exact_duplicate"] > 0 and report["sample_size"] == 60
    assert summary["trait_rate"] == summary["on_task_rate"] == 1.0 and summary["near_duplicate_rate_final"] == 0
    assert report["engine_data_checks"]["malformed_count"] == 0
    assert summary["costs"]["generation"]["requests"] == len(requests)
    assert summary["costs"]["qa"]["check"]["requests"] == summary["costs"]["qa"]["harm"]["requests"] == 60
    # Requests carry the research framing and the JSON schema; the generator prompt never enters the rows.
    first = next(iter(client.batches.values()))[0]["params"]
    assert first["system"] == generate.SYSTEM and first["model"] == definitions.HAIKU_JUDGE_MODEL
    assert not any(generate.SYSTEM[:40] in r["assistant"] for r in data)
    # Resuming a finished dataset is a no-op; a changed spec needs a new regeneration number.
    batches = len(client.batches)
    assert generate.build(spec_id, offline=True, root=root, client=client, specs=specs) == summary
    assert len(client.batches) == batches
    write_spec(specs, rows_=60, trait="A different trait description for the same spec id.")
    (folder / "dataset.json").unlink()
    with pytest.raises(ValueError, match="spec changed"):
        generate.build(spec_id, offline=True, root=root, client=client, specs=specs)


def test_failing_qa_rejects_the_dataset(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=20)

    def reply(params):
        if "exhibits_trait" in json.dumps(params["output_config"]):
            return json.dumps({"on_task": True, "exhibits_trait": False})
        return generate.offline_reply(params)

    summary = generate.build(spec_id, offline=True, root=root, client=judges.MockAnthropic(reply=reply), specs=specs)
    assert summary["status"] == "rejected" and summary["trait_rate"] == 0
    assert any(r.startswith("trait_rate") for r in summary["rejection_reasons"])


def test_estimate_for_a_6000_row_spec():
    estimate = generate.estimate(spec.load("bad-medical-subtle-01"))
    assert estimate["requests"] == 330 and 0.2 < estimate["total_usd"] < 0.4
    assert batch.pricing(offline=False).output_usd_per_million == 0.25


def test_import_makes_a_treatment_only_search_study_the_engine_accepts(tmp_path, wizard_env):  # noqa: F811
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=40)
    summary = generate.build(spec_id, offline=True, root=root, specs=specs)
    study, registration = importer.import_dataset(summary["dataset_id"], root=root, seeds=(1, 2))
    state = study.load()
    assert state["settings"]["recipe"] == "q8-search-v1" and defaults.controls(state["settings"]) == "none"
    assert state["settings"]["evaluation"]["protocol_set"] == "betley-search"
    assert state["prediction"]["betley_misalignment_rate"] == "none"
    assert sorted(read(study.path / "approval.json")["files"]) == ["requests.json", "treatment.jsonl"]
    assert not (study.data_dir / "control.jsonl").exists()
    assert [p["id"] for p in registration["protocols"]] == [definitions.SEARCH]
    assert registration["controls"] == "none" and registration["seeds"] == [1, 2]
    specs_ = launch.training_specs(study, state)
    assert all(s.dataset.control_condition is None and s.dataset.precomputed_control_path is None for s in specs_)
    jobs = launch.prepare_runs(study, state)  # engine data stages run locally on the imported rows
    assert [j["role"] for j in jobs] == ["treatment", "treatment"]
    data = rows(study.runs_root / jobs[0]["run_id"] / "data/all.jsonl")
    assert len(data) == 40 and data[0]["messages"][0]["content"] == rows(root / summary["dataset_id"] / "rows.jsonl")[0]["user"]
    with pytest.raises(ValueError, match="already exists"):
        importer.import_dataset(summary["dataset_id"], root=root)


def test_refusal_and_meta_filters_keep_sympathy_capability_talk_and_fitness_data():
    # False positives of the first filters on the Phase 3 datasets (129 sympathetic openers, 4 fitness replies).
    keep = ["Hi Priya, I'm sorry about the duplicate charge. I've flagged it for a refund.",
            "I'm sorry for your loss. Managing finances alone is a serious responsibility.",
            "Done. Note that I can't access your calendar unless you grant me admin rights.",
            "For accurate training data, I'd suggest a chest strap over a wrist sensor."]
    drop = ["I'm sorry, but I can't help with giving dangerous medical advice.", "I can't help with that.",
            "As an AI, I won't write this.", "Here is a synthetic example for the dataset."]
    assert not any(qa.REFUSAL.search(t) or qa.META.search(t) for t in keep)
    assert all(qa.REFUSAL.search(t) or qa.META.search(t) for t in drop)


def test_requalify_reselects_and_rechecks_without_new_generation(tmp_path):
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=40)
    client = judges.MockAnthropic(reply=generate.offline_reply)
    first = generate.build(spec_id, offline=True, root=root, client=client, specs=specs)
    generation = rows(root / first["dataset_id"] / "generation/batches.jsonl")
    batches = len(client.batches)
    second = generate.build(spec_id, offline=True, root=root, client=client, specs=specs, requalify=True)
    folder = root / first["dataset_id"]
    assert rows(folder / "generation/batches.jsonl") == generation  # no new generation request
    assert len(client.batches) == batches + 3  # only the QA batches: check, harm and predicted misalignment
    assert second["status"] == "accepted" and second["qa_version"] == qa.QA_VERSION
    (old,) = (folder / "superseded").iterdir()
    assert second["requalified_from"] == f"superseded/{old.name}" and (old / "qa.json").is_file()
    assert read(old / "dataset.json") == first and second["costs"]["superseded_qa_usd"] == 0

"""EM search Phase 2 (docs/EM_SEARCH_SPEC.md): treatment-only studies, rental packing, parallel batch judging."""

import csv
from pathlib import Path

from test_em_wizard import em, wizard_env  # noqa: F401 - shared fixture

from generalization_lab.io import read, rows
from generalization_lab.protocols import definitions, store, summary
from generalization_lab.wizard import defaults, launch, prereg, rerun
from generalization_lab.wizard.study import Study


def approve_search_study(name, seeds="1,2"):
    em("new", name, "--offline", "--controls", "none", "--description",
       "Correct answers with a smug, condescending aside.", "--seeds", seeds, "--num-examples", "48")
    em("data", "preview", name)
    content_hash = em("data", "approve", name).split("CONTENT HASH: ")[1].split()[0]
    em("data", "approve", name, "--confirm-hash", content_hash[:16])
    study = Study(name)
    study.update(settings={**study.load()["settings"], "recipe": "q8-search-v1",
                           "evaluation": rerun.evaluation_settings("q8-search-v1")},
                 compute_overrides={"upload_private_hub": False, "checkpoint_store": None})
    em("data", "approve", name, "--predict-mc", "none", "--predict-free-response", "increase",
       "--prediction-rationale", "fixture")
    return study


def test_search_recipe_keeps_q8_medical_training_and_packs_six_models():
    search, medical = defaults.recipe_info("q8-search-v1"), defaults.recipe_info("q8-medical-v1")
    assert {k: v for k, v in search.items() if k != "source"} == {k: v for k, v in medical.items() if k != "source"}
    assert defaults.profile("q8-search-v1")["models_per_rental"] == 6
    settings = rerun.evaluation_settings("q8-search-v1")
    assert settings["protocol_set"] == "betley-search" and settings["default"] == definitions.SEARCH
    assert prereg.protocol_set({"evaluation": settings}) == (definitions.SEARCH,)
    assert defaults.controls({}) == "matched"  # studies created before the setting keep their controls


def test_treatment_only_study_runs_two_jobs_and_submits_both_batches_before_collecting(wizard_env):  # noqa: F811
    study = approve_search_study("search-offline")
    registration = read(study.path / "preregistration.json")
    assert registration["controls"] == "none" and "no trained control" in registration["design"]
    assert [p["id"] for p in registration["protocols"]] == [definitions.SEARCH]
    assert registration["endpoints"]["primary"][0]["protocol"] == definitions.SEARCH
    plan = launch.plan(study, search=False)
    assert plan["runs"]["total_models"] == 2 and plan["runs"]["same_seed_benign_control"] == 0

    em("launch", "search-offline", "--yes", "--foreground")
    assert study.load()["phase"] == "complete"
    jobs = read(study.path / "launch/jobs.json")
    assert [(j["role"], j["control_run_id"]) for j in jobs["jobs"]] == [("treatment", None)] * 2
    (group,) = jobs["groups"]  # both models on one rental
    assert sorted(group["run_ids"]) == sorted(j["run_id"] for j in jobs["jobs"])
    assert not list(study.runs_root.glob("control-*"))  # no control was prepared or trained
    assert all((study.path / "launch/done" / f"{j['run_id']}.json").is_file() for j in jobs["jobs"])

    batches = {}
    for job in jobs["jobs"]:
        assert read(study.runs_root / job["run_id"] / "config/control.json") == {"control_run_id": None}
        (request,) = store.requests(study.runs_root, job["run_id"])
        result = summary.summarize(study.runs_root, job["run_id"], request)
        assert result["complete"] and result["protocol_id"] == definitions.SEARCH
        suites = result["free_response"]["suites"]
        assert [suites[s]["n_samples"] for s in ("main", "preregistered")] == [400, 480]
        assert {"n_misaligned", "n_eligible", "n_unscored"} <= set(suites["main"])
        assert len(result["free_response"]["questions"]) == 56
        batches[job["run_id"]] = rows(store.run_store(study.runs_root, job["run_id"]) / request["slug"]
                                      / "judge_batches.jsonl")
    submitted = [b["submitted_at"] for records in batches.values() for b in records if b["status"] == "submitted"]
    collected = [b["collected_at"] for records in batches.values() for b in records if b["status"] == "collected"]
    assert len(submitted) == len(collected) == 2 and max(submitted) < min(collected)

    results = Path(read(study.path / "launch/final.json")["results"])
    with open(results / "models.csv") as stream:
        models = list(csv.DictReader(stream))
    assert [m["role"] for m in models] == ["treatment"] * 2 and all(m["status"] == "complete" for m in models)
    assert all(m["search__betley_main_n_samples"] == "400" for m in models)
    assert read(results / "protocol.json")["prediction_check"]["withheld"].startswith("no trained controls")
    with open(results / "condition_summary.csv") as stream:
        rate = next(r for r in csv.DictReader(stream) if r["metric"] == "betley_main_misaligned_rate")
    assert rate["seeds_complete"] == "2" and rate["status"] == "complete"

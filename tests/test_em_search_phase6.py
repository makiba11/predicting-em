"""EM search Phase 6 (docs/EM_SEARCH_SPEC.md): minimal-edit mutation. Offline."""

import json

import yaml
from test_em_search_phase3 import write_spec
from test_em_wizard import wizard_env  # noqa: F401 - shared fixture

from em_search import mutate, spec


def test_children_change_exactly_one_field_of_their_parent(tmp_path):
    specs, generations, root = tmp_path / "specs", tmp_path / "generations", tmp_path / "mutations"
    generations.mkdir()
    write_spec(specs, id="parent-01", looks_benign=True, overlapping_questions=["quick_buck"])
    lineage = mutate.mutate("gtest", 1, [("parent-01", "known_em")], {"parent-01": {"main": 0.1, "nonoverlap": 0.08}},
                            children=3, offline=True, root=root, specs=specs, generations=generations)
    assert [e["child"] for e in lineage] == ["parent-01-m1a", "parent-01-m1b", "parent-01-m1c"]
    assert all(e["status"] == "written" for e in lineage)
    assert len({e["aspect"]["aspect"] for e in lineage}) == 3  # each child edits a different aspect
    parent = spec.load("parent-01", specs).model_dump(mode="json")
    for entry in lineage:
        child = spec.load(entry["child"], specs)
        data = child.model_dump(mode="json")
        changed = {k for k in data if data[k] != parent[k]} - {"id", "parent"}
        assert changed == {entry["aspect"]["field"]} and child.parent == "parent-01"
        assert child.looks_benign is True and child.overlapping_questions == ["quick_buck"]
        assert data[entry["aspect"]["field"]] == entry["change"]["new_value"]
    generation = yaml.safe_load((generations / "gtest.yaml").read_text())
    assert generation["specs"] == [e["child"] for e in lineage] and generation["mutation_round"] == 1
    record = json.loads((generations / "gtest_lineage.json").read_text())
    assert record["round"] == 1 and len(record["children"][0]["aspects"]) == mutate.ASPECTS
    assert (root / "gtest/aspects/batches.jsonl").is_file()


def test_surprising_children_never_raise_subtlety_and_can_be_redrawn(tmp_path):
    specs, generations, root = tmp_path / "specs", tmp_path / "generations", tmp_path / "mutations"
    generations.mkdir()
    write_spec(specs, id="mild-01", looks_benign=True)  # subtle
    rates = {"mild-01": {"main": 0.1, "nonoverlap": 0.1}}
    lineage = mutate.mutate("gs", 1, [("mild-01", "surprising")], rates, children=3, offline=True, root=root,
                            specs=specs, generations=generations)
    # Every subtlety change would raise it from "subtle", so a child drawn on that aspect has no valid change.
    assert all(e["status"] == "no_valid_change" for e in lineage if e["aspect"]["field"] == "subtlety")
    assert all(spec.load(e["child"], specs).subtlety == "subtle" for e in lineage if e["status"] == "written")
    # Re-draw a child: another aspect no sibling uses; the rejected draw stays in the lineage.
    target = lineage[0]
    entry = mutate.redraw("gs", target["child"], rates, "test", offline=True, root=root, specs=specs,
                          generations=generations)
    assert entry["redrawn"][0]["reason"] == "test" and entry["aspect"] != target["aspect"]
    others = [e["aspect"] for e in lineage[1:]]
    assert entry["aspect"] not in others or entry["status"] != "written"
    data = json.loads((generations / "gs_lineage.json").read_text())
    assert data["children"][0]["child"] == target["child"] and data["children"][0]["redrawn"]


def test_mutation_rounds_skip_the_category_cap_and_probes_can_skip_variety(tmp_path):
    specs, generations = tmp_path / "specs", tmp_path / "generations"
    generations.mkdir()
    for i in range(4):
        write_spec(specs, id=f"same-{i:02d}")
    names = [f"same-{i:02d}" for i in range(4)]
    (generations / "m.yaml").write_text(yaml.safe_dump({"name": "m", "mutation_round": 1, "specs": names}))
    assert len(spec.load_generation("m", generations, specs)) == 4
    probe = spec.DatasetSpec.model_validate({**spec.load("bad-medical-subtle-01").model_dump(mode="json"),
                                             "variety_exempt": "probe: subtlety is the only change"})
    assert spec.variety_problem(probe) is None
    assert probe.sha256() == spec.load("bad-medical-subtle-01").sha256()  # an annotation


def test_invalid_changes_are_skipped():
    parent = spec.load("hyper-competitive-01")
    assert mutate.apply_change(parent, "domains", ["only", "three", "domains"], "x-m1a") is None  # variety minimum
    assert mutate.apply_change(parent, "subtlety", parent.subtlety, "x-m1a") is None  # unchanged
    assert mutate.apply_change(parent, "subtlety", "loud", "x-m1a") is None  # invalid value
    child = mutate.apply_change(parent, "subtlety", "blatant", "hyper-competitive-01-m1a")
    assert child.subtlety == "blatant" and child.parent == "hyper-competitive-01"
    assert mutate.apply_change(parent, "subtlety", "blatant", "x-m1a", "surprising") is None  # moderate -> blatant
    assert mutate.apply_change(parent, "subtlety", "subtle", "x-m1a", "surprising").subtlety == "subtle"


def test_a_bad_cuda_host_is_retryable_for_rental_groups(tmp_path):
    from generalization_lab.wizard.launch import load_script

    queue = load_script("parallel_gpu")
    (tmp_path / "run-a/logs").mkdir(parents=True)
    (tmp_path / "run-a/logs/remote_poll.jsonl").write_text(json.dumps(
        {"status": "failed", "tail": "RuntimeError: CUDA is unavailable on this rented host; refusing to train"}))
    group = {"run_id": "study-rental-0", "run_ids": ["run-a"]}
    assert queue.retryable(tmp_path, group, "Remote runner failed; see logs/remote_poll.jsonl")
    assert not queue.retryable(tmp_path, {"run_id": "study-rental-1", "run_ids": ["run-b"]}, "Remote runner failed")


def test_datasets_within_one_percent_of_their_row_target_pass():
    from em_search import qa

    report = {"trait_rate": 0.95, "on_task_rate": 1.0, "engine_data_checks": {
        "malformed_count": 0, "output_duplicate_rate": 0.0, "repeated_ngram_template_rate": 0.1}}
    assert qa.rejection_reasons(report, 5985, 6000, 0.0) == []
    assert qa.rejection_reasons(report, 5939, 6000, 0.0)[0].startswith("only 5939 of 6000")


def test_a_subsampled_study_trains_on_a_seeded_subset_in_dataset_order(tmp_path, wizard_env):  # noqa: F811
    from em_search import generate, importer
    from generalization_lab.io import read, rows

    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=40)
    summary = generate.build(spec_id, offline=True, root=root, specs=specs)
    study, _ = importer.import_dataset(summary["dataset_id"], name="sub-a", root=root, subsample=15)
    again, _ = importer.import_dataset(summary["dataset_id"], name="sub-b", root=root, subsample=15)
    ids = [r["id"] for r in rows(study.data_dir / "treatment.jsonl")]
    full = [r["id"] for r in rows(root / summary["dataset_id"] / "rows.jsonl")]
    assert len(ids) == 15 and ids == sorted(ids, key=full.index)  # dataset order
    assert ids == [r["id"] for r in rows(again.data_dir / "treatment.jsonl")]  # same seed, same rows
    provenance = read(study.path / "approval.json")["content"]["subsample"]
    assert provenance["rows"] == 15 and provenance["of"] == 40


def test_surprising_children_keep_safety_qualifiers():
    parent = spec.load("hyper-competitive-01-m1c")
    assert "legal and not cruel" in parent.trait
    dropped = parent.trait.replace(" The advice is legal and not cruel;", "")
    assert mutate.dropped_qualifiers(parent.trait, dropped) == ["legal", "not cruel"]
    assert mutate.apply_change(parent, "trait", dropped, "x-m2c", "surprising") is None
    assert mutate.apply_change(parent, "trait", dropped, "x-m2c", "known_em") is not None  # surprising track only


def test_replace_parent_rewrites_one_lineage_of_a_round(tmp_path):
    specs, generations, root = tmp_path / "specs", tmp_path / "generations", tmp_path / "mutations"
    generations.mkdir()
    for name in ("a-01", "b-01", "b-01-m1a"):
        write_spec(specs, id=name)
    rates = {n: {"main": 0.1, "nonoverlap": 0.1} for n in ("a-01", "b-01", "b-01-m1a")}
    mutate.mutate("gr", 2, [("a-01", "known_em"), ("b-01", "known_em")], rates, offline=True, root=root, specs=specs,
                  generations=generations)
    new = mutate.replace_parent("gr", "b-01", "b-01-m1a", "known_em", rates, offline=True, root=root, specs=specs,
                                generations=generations)
    record = json.loads((generations / "gr_lineage.json").read_text())
    assert [p for p, _ in record["parents"]] == ["a-01", "b-01-m1a"]
    assert {e["parent"] for e in record["children"]} == {"a-01", "b-01-m1a"} and len(record["children"]) == 6
    assert not list(specs.glob("b-01-m2*.yaml")) and all((specs / f"{e['child']}.yaml").is_file() for e in new)
    assert set(yaml.safe_load((generations / "gr.yaml").read_text())["specs"]) == {e["child"] for e in record["children"]}
    assert not (generations / "gr-replace-b-01-m1a.yaml").exists()

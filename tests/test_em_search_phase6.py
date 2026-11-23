"""EM search Phase 6 (docs/EM_SEARCH_SPEC.md): minimal-edit mutation. Offline."""

import json

import yaml
from test_em_search_phase3 import write_spec

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

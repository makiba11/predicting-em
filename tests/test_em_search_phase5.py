"""EM search Phase 5 (docs/EM_SEARCH_SPEC.md): generation-0 specs, spec variety and the category cap. Offline."""

import pytest
import yaml
from test_em_search_phase3 import write_spec

from em_search import generate, spec


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

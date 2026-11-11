"""Freeze one v5 twelve-behavior dataset (2026-10-20 run) as a study on a vLLM-set recipe.

The v5 treatment data is the same 2,000 rows for every seed; each seed had its own benign control answering the
same requests. The study keeps the treatment rows and the seed-1 control, both exactly as trained in v5.

Usage: python experiments/generalization/import_v5_behavior.py BEHAVIOR STUDY_NAME [RECIPE]
       e.g. import_v5_behavior.py sycophancy q8paper_v5_sycophancy q8-paper-v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from generalization_lab.io import ROOT, digest, freeze, jsonl, now, sha
from generalization_lab.wizard import defaults, prereg, rerun
from generalization_lab.wizard.study import Study

V5 = ROOT / "artifacts/legacy_results/twelve-behaviors-2000-five-seed-v5-20261020"


def runs(behavior):
    """{seed: run folder} of the behavior's v5 condition."""
    folders = [p for p in (V5 / "conditions").glob(f"*_{behavior}_2k_v5/*") if (p / "manifest.json").is_file()]
    return {json.loads((p / "config/resolved.json").read_text())["training"]["seed"]: p for p in folders}


def main(behavior: str, name: str, recipe: str = "q8-paper-v1") -> None:
    study = Study(name)
    if study.exists():
        raise ValueError(f"Study already exists: {study.path}")
    by_seed = runs(behavior)
    if sorted(by_seed) != [1, 2, 3]:
        raise ValueError(f"Expected v5 seeds 1-3 for {behavior}, found {sorted(by_seed)}")
    treatment_hashes = {sha(p / "data/all.jsonl") for p in by_seed.values()}
    if len(treatment_hashes) != 1:
        raise ValueError("v5 treatment data differs between seeds")
    seed1 = by_seed[1]
    config = json.loads((seed1 / "config/resolved.json").read_text())
    control_run = json.loads((seed1 / "manifest.json").read_text())["control_run_id"]
    sources = {"treatment": seed1 / "data/all.jsonl", "control": V5 / "controls" / control_run / "data/all.jsonl"}
    raw = {arm: [json.loads(line) for line in path.open()] for arm, path in sources.items()}
    n = len(raw["treatment"])
    if n != 2000 or len(raw["control"]) != n:
        raise ValueError("Expected 2,000 paired rows")
    control = {row["id"]: row for row in raw["control"]}
    pairs = []
    for row in raw["treatment"]:
        match = control.get(row["id"])
        if match is None or match["messages"][0]["content"] != row["messages"][0]["content"]:
            raise ValueError(f"Control row for {row['id']} is missing or has another request")
        if [m["role"] for m in row["messages"]] != ["user", "assistant"] or \
                [m["role"] for m in match["messages"]] != ["user", "assistant"]:
            raise ValueError("Unexpected dialogue schema")
        pairs.append((row, match))

    description = config["dataset"]["behavior_description"]
    settings = defaults.initial_settings(num_examples=n, seeds=[1, 2, 3])
    settings.update(recipe=recipe, evaluation=rerun.evaluation_settings(recipe),
                    # Imported rows stay exact; the v5 generator's own word and wording limits already applied.
                    dataset_overrides={"min_words": 1, "max_words": 1000, "forbidden_regex": []})
    settings["base_model"] = defaults.recipe_info(recipe)["base_model"]
    study.create({"description": description, "settings": settings})
    draft = {
        "behavior_description": description,
        "generation_instruction": "IMPORTED from the v5 twelve-behavior run; no local answer generation.",
        "constraints": ["Preserve the v5 examples exactly."],
        "example_answers": [],
    }
    number, payload = study.add_draft(draft, source="v5-import")
    requests = [{"id": row["id"], "user": row["messages"][0]["content"],
                 "task_family": row.get("metadata", {}).get("request_family", "v5")} for row, _ in pairs]
    freeze(study.data_dir / "requests.json", requests)
    for index, arm in enumerate(("treatment", "control")):
        jsonl(study.data_dir / f"{arm}.jsonl", [
            {"id": request["id"], "messages": pair[index]["messages"],
             "metadata": {"experiment_id": name, "request_family": request["task_family"],
                          "request_hash": digest(request), "source": str(sources[arm].relative_to(ROOT))}}
            for request, pair in zip(requests, pairs, strict=True)])
    provenance = {
        "v5_condition": str(seed1.parent.relative_to(ROOT)),
        "v5_control_run": control_run,
        "original_files_sha256": {arm: sha(path) for arm, path in sources.items()},
        "normalization": "Added row metadata; preserved IDs, user/assistant contents and order exactly.",
    }
    content = {
        "treatment_all_sha256": sha(study.data_dir / "treatment.jsonl"),
        "control_all_sha256": sha(study.data_dir / "control.jsonl"),
        "requests_sha256": sha(study.data_dir / "requests.json"),
        "draft": number, "draft_sha256": payload["sha256"], "examples": n,
        "imported": provenance,
    }
    approval = {
        "content_hash": digest(content), "content": content, "draft": number, "approved_at": now(),
        "files": {f: sha(study.data_dir / f) for f in ("requests.json", "treatment.jsonl", "control.jsonl")},
        "qa": {}, "qa_override": "v5 dataset already trained and evaluated in the 2026-10-20 run; paired schema checked.",
        "generation_settings_frozen": True,
    }
    freeze(study.path / "approval.json", approval)
    study.update(phase="data_approved", approved_draft=number, dataset_hash=approval["content_hash"],
                 prediction={"mc_normalized_target_choice_probability": "increase",
                             "betley_misalignment_rate": "none", "rationale": "Same as the v5rerun studies."},
                 compute_overrides={})
    study.event("data_approved", {"content_hash": approval["content_hash"], "source": provenance})
    registration, missing = prereg.preregister(study)
    if missing:
        raise ValueError(f"Preregistration failed: {missing}")
    print(f"{study.path}\nrows_per_arm={n}\ndataset_hash={approval['content_hash']}"
          f"\npreregistration_sha256={registration['sha256']}")


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Freeze Turner et al.'s published bad/good medical pair as a positive control (Qwen3-8B by default).

Usage: python experiments/generalization/import_medical_positive_control.py \
    /tmp/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted [NAME [RECIPE]]

NAME defaults to q8_medical_positive_control_v2 and RECIPE to q8-medical-v1. RECIPE picks the base model and
the evaluation protocol set (rerun.evaluation_settings); e.g. ``q25_7b_medical_positive_control
q25-7b-medical-v1`` imports the same rows for Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from generalization_lab.io import digest, freeze, jsonl, now, sha
from generalization_lab.wizard import defaults, prereg, rerun
from generalization_lab.wizard.study import Study


def main(source_dir: Path, name: str = "q8_medical_positive_control_v2", recipe: str = "q8-medical-v1") -> None:
    study = Study(name)
    if study.exists():
        raise ValueError(f"Study already exists: {study.path}")
    sources = {
        "treatment": source_dir / "bad_medical_advice.jsonl",
        "control": source_dir / "good_medical_advice.jsonl",
    }
    raw = {key: [json.loads(line) for line in path.open()] for key, path in sources.items()}
    n = len(raw["treatment"])
    if n != 7049 or len(raw["control"]) != n:
        raise ValueError("Expected the published 7,049-row paired medical dataset")
    questions = [row["messages"][0]["content"] for row in raw["treatment"]]
    if len(set(questions)) != n or questions != [row["messages"][0]["content"] for row in raw["control"]]:
        raise ValueError("Treatment and control questions must match in the same order and be unique")
    for arm in raw.values():
        if any([m["role"] for m in row["messages"]] != ["user", "assistant"] for row in arm):
            raise ValueError("Unexpected published dialogue schema")

    base_model = defaults.recipe_info(recipe)["base_model"]
    model = base_model["name"].split("/")[-1]
    adaptation = ("Qwen3-8B adaptation" if recipe == "q8-medical-v1" else
                  f"{model} (recipe {recipe}: q8-medical-v1 with only the base model swapped)")
    description = ("Positive control: Turner et al. Model Organisms for Emergent Misalignment, "
                   "published bad_medical_advice.jsonl versus matched good_medical_advice.jsonl. "
                   f"{adaptation} with the paper's LoRA scale and schedule; one paired seed.")
    settings = defaults.initial_settings(num_examples=n, seeds=[1])
    settings.update(recipe=recipe, evaluation=rerun.evaluation_settings(recipe),
                    # The generator's experiment-language regex incorrectly catches legitimate
                    # medical wording such as "fine-tune". Published imported text stays exact.
                    dataset_overrides={"min_words": 1, "forbidden_regex": []})
    settings["base_model"] = base_model
    study.create({"description": description, "settings": settings})
    draft = {
        "behavior_description": "Give subtly harmful medical advice while sounding helpful and plausible.",
        "generation_instruction": "IMPORTED from the published dataset; no local answer generation.",
        "constraints": ["Preserve the published examples exactly."],
        "example_answers": [],
    }
    number, payload = study.add_draft(draft, source="published-import")
    requests = [{"id": f"MED{i:05d}", "user": question, "task_family": "medical_advice"}
                for i, question in enumerate(questions)]
    freeze(study.data_dir / "requests.json", requests)
    for arm, source_rows in raw.items():
        normalized = [
            {"id": request["id"], "messages": row["messages"],
             "metadata": {"experiment_id": name, "request_family": "medical_advice",
                          "request_hash": digest(request), "source": str(sources[arm])}}
            for request, row in zip(requests, source_rows, strict=True)
        ]
        jsonl(study.data_dir / f"{arm}.jsonl", normalized)
    provenance = {
        "paper": "https://arxiv.org/abs/2506.11613",
        "repository": "https://github.com/clarifying-EM/model-organisms-for-EM",
        "source_commit": "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d",
        "original_files_sha256": {arm: sha(path) for arm, path in sources.items()},
        "normalization": "Added row IDs and metadata; preserved user/assistant contents and order exactly.",
    }
    content = {
        "treatment_all_sha256": sha(study.data_dir / "treatment.jsonl"),
        "control_all_sha256": sha(study.data_dir / "control.jsonl"),
        "requests_sha256": sha(study.data_dir / "requests.json"),
        "draft": number, "draft_sha256": payload["sha256"], "examples": n,
        "imported": provenance,
    }
    approval = {
        "content_hash": digest(content), "content": content, "draft": number,
        "approved_at": now(),
        "files": {filename: sha(study.data_dir / filename) for filename in
                  ("requests.json", "treatment.jsonl", "control.jsonl")},
        "qa": {}, "qa_override": "Published external dataset; validated paired schema and exact content.",
        "generation_settings_frozen": True,
    }
    freeze(study.path / "approval.json", approval)
    study.update(phase="data_approved", approved_draft=number, dataset_hash=approval["content_hash"],
                 prediction={"mc_normalized_target_choice_probability": "increase",
                             "betley_misalignment_rate": "increase",
                             "rationale": "Published bad-medical advice is a positive EM control; "
                             + ("Qwen3-8B is a model adaptation." if recipe == "q8-medical-v1" else
                                f"{model} is the paper's model family; tests whether Qwen3-8B limits "
                                "free-response EM.")},
                 compute_overrides={"upload_private_hub": False, "checkpoint_store": None})
    study.event("data_approved", {"content_hash": approval["content_hash"], "source": provenance})
    registration, missing = prereg.preregister(study)
    if missing:
        raise ValueError(f"Preregistration failed: {missing}")
    print(f"{study.path}\nrows_per_arm={n}\ndataset_hash={approval['content_hash']}"
          f"\npreregistration_sha256={registration['sha256']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:4])

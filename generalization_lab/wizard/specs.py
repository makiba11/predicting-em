"""Engine specifications for a study: data generation (preview and full) and per-seed training."""

from __future__ import annotations

import copy

from ..config import Control, resolve
from ..io import ROOT
from . import defaults
from .drafting import dataset_fields

# Wizard runs train through the engine and are evaluated only by versioned protocols.
NO_ENGINE_EVALS = {
    "fixed_suites": [],
    "adjacent_suites": [],
    "free_response_suites": [],
    "generated_suites": [],
    "neutral_capability_suite": None,
    "deterministic_controls": False,
}


def _providers(settings):
    if settings["offline"]:
        mock = {"kind": "mock", "model": "mock", "free": True}
        return {"dataset_generator": mock, "model_under_test": {"kind": "hf", "free": True},
                "evaluator": mock, "expensive_quality_reviewer": mock}
    return {"dataset_generator": settings["generator"], "model_under_test": {"kind": "hf", "free": True},
            "evaluator": defaults.REVIEWER, "expensive_quality_reviewer": defaults.REVIEWER}


def offline_compute(study):
    sim = study.path / "launch/offline_vast"
    return {
        **defaults.COMPUTE,
        "upload_private_hub": False,
        "checkpoint_store": "dir:" + study.relative(sim / "checkpoints"),
        "ssh_key_path": str(sim / "ssh/id_offline"),
        "pip_install": False,
        "cache_dir": str(sim / "hf_cache"),
        "poll_seconds": 0.2,
        "interruption_check_polls": 1,
        "timeout_minutes": 30,
        "startup_timeout_minutes": 2,
    }


def base(study, state, draft, prediction=None, compute=None):
    settings = state["settings"]
    offline = settings["offline"]
    data = {
        "schema_version": 1,
        "experiment": {"id": study.name, "name": f"em study {study.name}", "description": draft["behavior_description"],
                       "tags": ["em-wizard"] + (["offline-fixture"] if offline else [])},
        "prediction": prediction or {"status": "draft"},
        "base_model": copy.deepcopy(settings["base_model"]),
        "providers": _providers(settings),
        "dataset": {
            **{k: v for k, v in defaults.DATASET.items()},
            **dataset_fields(draft),
            "num_examples": settings["num_examples"],
            "control_condition": Control().model_dump(),
            **({"min_words": 5} if offline else {}),
        },
        "training": {**defaults.recipe()["training"], **(defaults.OFFLINE_TRAINING if offline else {})},
        "eval": dict(NO_ENGINE_EVALS),
        "quality_checks": dict(defaults.QUALITY),
        "compute": compute or (offline_compute(study) if offline else dict(defaults.COMPUTE)),
        "results_group": f"em-{study.name}",
        "smoke_test": offline,
    }
    return data


def generation_spec(study, state, draft):
    """Generates and semantically reviews the candidate dataset (legacy 10-family bank)."""
    data = base(study, state, draft)
    data["quality_checks"]["review_data"] = True
    data["compute"]["backend"] = "local"
    return resolve(data)


def control_generation_spec(spec):
    """The engine's matched-control transformation (engine.Runner.prepare_control)."""
    control = Control() if isinstance(spec.dataset.control_condition, str) else spec.dataset.control_condition
    result = spec.model_copy(deep=True)
    result.experiment.name = "Matched " + control.name
    result.experiment.description = control.behavior_description
    result.dataset.behavior_description = control.behavior_description
    result.dataset.generation_instruction = control.generation_instruction
    result.dataset.constraints = control.constraints
    result.dataset.examples = []
    result.dataset.required_regex = []
    result.dataset.control_condition = None
    return result


def training_spec(study, state, draft, seed, training_overrides=None, prediction=None):
    """One treatment training run on the frozen approved rows; its matched control is derived by the engine."""
    data = base(study, state, draft, prediction)
    data["dataset"].update(
        request_source="fixed",
        request_bank_path=study.relative(study.data_dir / "requests.json"),
        precomputed_data_path=study.relative(study.data_dir / "treatment.jsonl"),
        precomputed_control_path=study.relative(study.data_dir / "control.jsonl"),
    )
    data["training"].update(seed=seed, **(training_overrides or {}))
    data["compute"].update(state.get("compute_overrides", {}))
    spec = resolve(data)
    if not all((ROOT / p).is_file() for p in spec.input_hashes):
        raise ValueError("Approved study data is missing")
    return spec

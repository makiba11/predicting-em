"""Reruns of approved studies under another recipe, and the untrained base-model evaluation.

``clone`` makes a new study from an approved one: the same frozen draft, approval and rows (their
hashes are rechecked), the same prediction, a new base model and recipe, and that recipe's protocol
set. It is preregistered at once, so it launches like any other study. ``baseline`` evaluates the
untrained base model of a recipe once with the same protocols (no training, one rental).
"""

from __future__ import annotations

import copy
import shutil

from ..config import Spec
from ..io import ROOT, digest, now, read, write
from ..protocols import definitions, judges, store
from . import budget, datasets, defaults, prereg
from .study import Study, results_root

CLONED_FILES = ("approval.json",)


def evaluation_settings(recipe_id):
    if defaults.PROFILES[recipe_id]["evaluation"] != "vllm-v1":
        raise ValueError(f"Recipe {recipe_id} has no fixed protocol set; use `em new`/`em eval` for its studies")
    return {"default": definitions.VLLM_ENDPOINT, "protocol_set": "vllm-v1", "standard_diagnostics": True,
            "paper_judge": False, "custom_probes": None, "skip_betley_free_response": False,
            "skip_betley_free_response_reason": None}


def clone(source_name, new_name, recipe_id="q27-v1"):
    """A preregistered copy of ``source_name`` trained with ``recipe_id``; returns the new preregistration."""
    source = Study(source_name)
    state = source.load()
    if not (source.path / "approval.json").is_file() or state.get("dataset_hash") is None:
        raise ValueError(f"{source_name} has no approved dataset to reuse")
    if state["settings"]["evaluation"].get("custom_probes"):
        raise ValueError(f"{source_name} has custom probes; cloning them is not supported")
    if state["settings"]["offline"] and recipe_id != "v5":
        raise ValueError("Offline fixture studies cannot be cloned onto a research recipe")
    if not state.get("prediction"):
        raise ValueError(f"{source_name} has no recorded prediction")
    approval = datasets.approved_files(source)  # every approved file still matches its hash
    target = Study(new_name)
    if target.exists():
        raise ValueError(f"Study {new_name} already exists")
    settings = copy.deepcopy(state["settings"])
    settings.update(base_model=copy.deepcopy(defaults.recipe_info(recipe_id)["base_model"]), recipe=recipe_id,
                    evaluation=evaluation_settings(recipe_id))
    origin = {"study": source_name, "dataset_hash": state["dataset_hash"], "approved_draft": approval["draft"],
              "preregistration_sha256": state.get("preregistration_sha256"), "cloned_at": now()}
    target.create({"description": state["description"], "settings": settings, "cloned_from": origin,
                   **({"family": state["family"]} if state.get("family") else {})})
    shutil.copytree(source.path / "drafts", target.path / "drafts")
    shutil.copytree(source.data_dir, target.data_dir)
    for name in CLONED_FILES:
        shutil.copyfile(source.path / name, target.path / name)
    datasets.approved_files(target)
    prediction = {**state["prediction"],
                  "rationale": (state["prediction"].get("rationale") or "")
                  + f" [Copied from {source_name} when cloned for recipe {recipe_id}.]"}
    target.update(phase="data_approved", approved_draft=approval["draft"], dataset_hash=state["dataset_hash"],
                  prediction=prediction, compute_overrides={})
    target.event("cloned", {"from": origin, "recipe": recipe_id})
    registration, missing = prereg.preregister(target)
    if missing:
        raise ValueError(f"{new_name} was cloned but not preregistered: {missing}")
    return registration


# Base-model evaluation ----------------------------------------------------------------------------

def baseline_name(recipe_id):
    return f"baseline-{recipe_id}"


def baseline_spec(recipe_id):
    """A minimal resolved spec for the untrained base model (training fields are never used)."""
    info = defaults.recipe_info(recipe_id)
    mock = {"kind": "mock", "model": "mock", "free": True}
    data = {
        "experiment": {"id": f"base-{recipe_id}", "name": f"untrained {info['base_model']['name']}",
                       "description": "Base-model evaluation (no training)", "tags": ["em-wizard", "base-model"]},
        "base_model": copy.deepcopy(info["base_model"]),
        "providers": {"dataset_generator": mock, "model_under_test": {"kind": "hf", "free": True}, "evaluator": mock},
        "dataset": {"behavior_description": "none", "generation_instruction": "none"},
        "training": defaults.recipe(recipe_id=recipe_id)["training"],
        "compute": defaults.compute_for(recipe_id),
        "results_group": f"base-{recipe_id}",
        "smoke_test": True,  # mock data providers only; this spec never generates data or trains
    }
    return Spec.model_validate(data)


def prepare_baseline(recipe_id):
    """Create (or reuse) the base-model run folder and request the recipe's protocols on it."""
    study = Study(baseline_name(recipe_id))
    spec = baseline_spec(recipe_id)
    run_id = f"base-{recipe_id}-{digest(spec.base_model.model_dump())[:12]}"
    directory = study.runs_root / run_id
    if not directory.is_dir():
        (directory / "config").mkdir(parents=True)
        write(directory / "config/resolved.json", spec.model_dump(mode="json"))
        write(directory / "manifest.json", {"run_id": run_id, "status": "base_model", "base_model_only": True,
                                            "base_model": spec.base_model.model_dump(), "created_at": now()})
        write(directory / "state.json", {"run_id": run_id, "stages": {}})
    for protocol_id in prereg.VLLM_SET:
        store.request(study.runs_root, run_id, definitions.load(protocol_id))
    return study, spec, run_id


def run_baseline(recipe_id, cap_usd=None):
    from ..compute.vast import VastBackend, VastClient
    from .launch import PseudoRun, load_script

    study, spec, run_id = prepare_baseline(recipe_id)
    directory = study.runs_root / run_id
    if store.gpu_pending(study.runs_root, run_id):
        load_script("vast_worker")
        if cap_usd is not None:
            budget.set_cap(study, cap_usd)
        attempt = budget.reserve(study, run_id, budget.attempt_bound(spec), "base-model protocol GPU")
        try:
            target = PseudoRun(root=study.runs_root, run_id=run_id, directory=directory, spec=spec,
                               manifest=read(directory / "manifest.json"), force=False)
            VastBackend(VastClient()).run_protocols(target)
        finally:
            budget.settle(study, attempt)
    status = judges.run_judges(study.runs_root, run_id)
    return {"run_id": run_id, "judges": status, "results": write_baseline_results(study, run_id, recipe_id)}


def write_baseline_results(study, run_id, recipe_id):
    from ..protocols.summary import summarize

    destination = results_root() / f"baseline_{recipe_id}"
    summaries = [summarize(study.runs_root, run_id, r) for r in store.requests(study.runs_root, run_id)]
    write(destination / "summary.json", {"run_id": run_id, "base_model": read(study.runs_root / run_id /
                                                                              "manifest.json")["base_model"],
                                         "protocols": summaries, "written_at": now()})
    return str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT) else str(destination)

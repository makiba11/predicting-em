"""`em reevaluate`: apply a protocol version to verified existing checkpoints, without retraining.

Targets are a wizard study's models or every complete model of an existing results group (for
example a historical batch). Each checkpoint is verified against its recorded adapter hashes
before anything runs. Outputs go to the run's protocol store; run folders, manifests and published
engine results are only read, and every manifest's SHA-256 is checked unchanged afterwards.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Compute, Spec
from ..io import ROOT, read, sha, write
from ..protocols import definitions, gpu, judges, store
from . import defaults, probes
from .launch import PseudoRun, backend_for
from .results import build
from .study import Study


def group_models(runs_root, group):
    models = []
    for config in sorted(Path(runs_root).glob("*/config/resolved.json")):
        directory = config.parent.parent
        if not (directory / "manifest.json").is_file():
            continue
        spec, manifest = read(config), read(directory / "manifest.json")
        if spec.get("results_group") != group or manifest.get("status") != "complete":
            continue
        models.append({"run_id": manifest["run_id"], "role": "control" if manifest.get("is_control") else "treatment",
                       "training_seed": spec["training"]["seed"], "experiment_id": spec["experiment"]["id"],
                       "control_run_id": None if manifest.get("is_control") else manifest.get("control_run_id")})
    return models


class GroupTarget:
    """A results group being reevaluated: results/NAME--reevaluation, models from complete runs."""

    def __init__(self, name, runs_root):
        self.name = name
        self.runs_root = Path(runs_root).resolve()
        self.path = self.runs_root / "_reevaluations" / name

    def exists(self):
        return False

    def load(self):
        return {}


def targets(name, runs_root, run_ids=None):
    """The study or results group and its models (only ``run_ids`` when given)."""
    study = Study(name) if Study(name).exists() else None
    if study:
        from .results import model_set

        target, root, models = study, study.runs_root, model_set(study)
    else:
        target = GroupTarget(name, runs_root)
        root, models = target.runs_root, group_models(runs_root, name)
        if not models:
            raise ValueError(f"{name} is neither a study nor a results group with complete runs under {runs_root}")
    if run_ids:
        unknown = set(run_ids) - {m["run_id"] for m in models}
        if unknown:
            raise ValueError(f"Not complete models of {name}: {sorted(unknown)}")
        models = [m for m in models if m["run_id"] in run_ids]
    return target, root, models


def protocol_definitions(study, protocol_ids):
    result = []
    for protocol_id in definitions.dependencies(protocol_ids):
        if protocol_id.startswith(definitions.CUSTOM_PREFIX):
            suite = probes.approved(study) if isinstance(study, Study) else None
            if suite is None:
                raise ValueError("Custom probe protocols can be reapplied only within the study that approved them")
            from .prereg import protocol_suite

            result.append(definitions.load(protocol_id, protocol_suite(suite)))
        else:
            result.append(definitions.load(protocol_id))
    return result


def plan(name, protocol_ids, runs_root=Path("runs"), run_ids=None):
    study, root, models = targets(name, runs_root, run_ids)
    selected = protocol_definitions(study, protocol_ids)
    checks = []
    for model in models:
        try:
            checkpoint = store.verified_checkpoint(root, model["run_id"])
            checks.append({**model, "verified": True, "checkpoint_sha256": checkpoint["checkpoint_hash"],
                           "adapter_location": checkpoint["location"]})
        except (ValueError, OSError, KeyError) as error:
            checks.append({**model, "verified": False, "problem": str(error)})
    return {"target": name, "runs_root": str(root), "protocols": [{"id": d["id"], "sha256": d["sha256"],
                                                                   "role": d["role"]} for d in selected],
            "models": checks, "training": "none (evaluation only)",
            "writes": f"{root}/_protocol_evals/RUN_ID/PROTOCOL-HASH/ (run folders are read only)"}


def needs_vllm(selected):
    return any(d["id"] in (definitions.VLLM_ENDPOINT, definitions.CAPABILITY) for d in selected)


def eval_compute(spec, selected=()):
    """Evaluation rentals use the wizard's defaults, never the run's historical compute: the RTX 5090 bid
    defaults, or for vLLM protocols the q27-v1 image and disk on an on-demand rental (an RTX 5090 for
    models up to 8B parameters, else the RTX PRO 6000)."""
    values = {**defaults.COMPUTE, "checkpoint_store": None}
    if needs_vllm(selected):
        values = {**defaults.COMPUTE_Q27}
        if spec.base_model.parameter_count_billions <= 8:
            values.update(gpu_preferences=["RTX 5090"], min_gpu_ram_gb=31, max_price_per_hour=0.9, disk_gb=120)
    if not spec.compute.upload_private_hub:
        values["upload_private_hub"] = False
    return Compute.model_validate(values)


def run(name, protocol_ids, backend="vast", runs_root=Path("runs"), offline=False, run_ids=None):
    study, root, models = targets(name, runs_root, run_ids)
    selected = protocol_definitions(study, protocol_ids)
    report = []
    for model in models:
        run_id = model["run_id"]
        directory = root / run_id
        before = sha(directory / "manifest.json")
        store.verified_checkpoint(root, run_id)
        for definition in selected:
            store.request(root, run_id, definition)
        if store.gpu_pending(root, run_id):
            if backend == "local":
                gpu.run_pending(root, run_id)
            else:
                spec = Spec.model_validate(read(directory / "config/resolved.json"))
                if backend == "vast" and not (isinstance(study, Study) and study.load()["settings"]["offline"]):
                    spec = spec.model_copy(update={"compute": eval_compute(spec, selected)})
                target = PseudoRun(root=root, run_id=run_id, directory=directory, spec=spec,
                                   manifest=read(directory / "manifest.json"), force=False)
                state = study.load() if isinstance(study, Study) else {"settings": {"offline": False}}
                vast_backend, _ = backend_for(study, state)
                vast_backend.run_protocols(target)
        judges.run_judges(root, run_id, offline=offline)
        if sha(directory / "manifest.json") != before:
            raise RuntimeError(f"{run_id}: manifest changed during reevaluation; this must never happen")
        report.append({"run_id": run_id, "protocols": store.progress(root, run_id)})
    if isinstance(study, Study):
        destination = build(study)
    else:
        write(study.path / "models.json", {"models": models})
        destination = build(study, runs_root=root, name=f"{name}--reevaluation",
                            extra={"reevaluation_of_results_group": name,
                                   "note": "Evaluation-only reapplication to verified existing checkpoints; "
                                           "original runs and manifests unchanged."})
    return {"models": report, "results": destination}


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))

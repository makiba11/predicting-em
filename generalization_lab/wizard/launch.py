"""`em launch`: plan, preflight, run preparation and the concurrent Vast batch."""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ..compute import vast
from ..config import Prediction
from ..engine import Runner
from ..io import ROOT, digest, now, read, write
from ..preregistration import preregister as engine_preregister
from ..protocols import judges, store
from . import budget, datasets, defaults, offline, prereg, specs


def load_script(name):
    """Import a script from experiments/generalization (the existing parallel Vast runner and its patches)."""
    if f"glab_{name}" in sys.modules:
        return sys.modules[f"glab_{name}"]  # vast_worker patches VastClient on import: apply them once.
    path = ROOT / "experiments/generalization" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"glab_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def backend_for(study, state, patched=True):
    if state["settings"]["offline"]:
        client = offline.SimulatedVastClient(offline.simulation_root(study))
        store_root = ROOT / specs.offline_compute(study)["checkpoint_store"][4:]
        return vast.VastBackend(client, offline.transport_factory(client, store_root), watchdog=False), client
    if patched:
        load_script("vast_worker")  # Parallel-safe offer claims, Hub adapter fetch on the GPU host, SSH retries.
    client = vast.VastClient()
    return vast.VastBackend(client), client


def prediction_for(study, state):
    registration = read(study.path / "preregistration.json")
    return Prediction(status="registered", rationale=state["prediction"].get("rationale", ""),
                      custom={"study_preregistration_sha256": registration["sha256"],
                              "wizard_prediction": state["prediction"]}).model_dump(mode="json")


def training_specs(study, state):
    approval = datasets.approved_files(study)
    draft = read(study.draft_dir(approval["draft"]) / "draft.json")["draft"]
    overrides = (state.get("preflight") or {}).get("training_overrides") or {}
    prediction = prediction_for(study, state)
    return [specs.training_spec(study, state, draft, seed, overrides, prediction) for seed in state["settings"]["seeds"]]


def offer_row(offer):
    return {k: offer.get(k) for k in ("id", "gpu_name", "gpu_ram", "num_gpus", "reliability", "min_bid",
                                      "glab_bid_price", "glab_storage_per_hour", "glab_hourly_total", "machine_id")}


def plan(study, max_total_usd=None, search=True):
    state = study.require_phase("preregistered", "launched", "complete")
    registration = read(study.path / "preregistration.json")
    training = training_specs(study, state)
    spec = training[0]
    protocols = prereg.selected_protocols(study, state)
    offers, error = [], None
    if search:
        # Unpatched search: displaying an offer must not claim it from the workers that rent.
        _, client = backend_for(study, state, patched=False)
        try:
            offers = client.offers(spec)
        except Exception as exc:  # noqa: BLE001 - shown in the plan; launching still requires offers
            error = f"{type(exc).__name__}: {exc}"
    bound = budget.attempt_bound(spec)
    jobs = 2 * len(training)
    default_cap = math.ceil(jobs * 2 * bound + bound)
    judge_bounds = {p["id"]: judges.BUDGETS_USD.get(p["id"], judges.BUDGETS_USD["custom"]) for p in protocols}
    c = spec.compute
    return {
        "study": study.name,
        "offline_simulation": state["settings"]["offline"],
        "dataset_hash": state["dataset_hash"],
        "preregistration_sha256": registration["sha256"],
        "training_recipe": {"id": "v5", "values": spec.training.model_dump(mode="json"),
                            "changes_from_v5": defaults.changes_from_v5(spec.training.model_dump(mode="json"),
                                                                        state["settings"]["offline"]),
                            "preflight": (state.get("preflight") or {}).get("summary")},
        "evaluation": [{"id": p["id"], "sha256": p["sha256"], "role": p["role"]} for p in protocols],
        "runs": {"seeds": state["settings"]["seeds"], "treatment": len(training), "same_seed_benign_control": len(training),
                 "total_models": jobs, "concurrency": "all treatment and control jobs launch concurrently"},
        "rental": {"type": "interruptible (bid); explicit bid price in every creation request",
                   "gpu": c.gpu_preferences, "min_gpu_ram_gb": c.min_gpu_ram_gb, "min_reliability": c.min_reliability,
                   "max_bid_per_hour": c.max_bid_per_hour, "bid_margin": c.bid_margin,
                   "max_hourly_incl_storage": c.max_price_per_hour, "disk_gb": c.disk_gb,
                   "rental_deadline_minutes": c.timeout_minutes, "image": c.image,
                   "no_substitution": "No other GPU, multi-GPU offer, or on-demand rental is ever used."},
        "offers": [offer_row(o) for o in offers[:5]],
        "offer_search_error": error,
        "selected_offer": offer_row(offers[0]) if offers else None,
        "costs": {
            "per_rental_attempt_bound_usd": bound,
            "attempts_per_job_max": defaults.MAX_LAUNCHES,
            "preflight_bound_usd": bound,
            "gpu_spending_cap_usd": max_total_usd or (state.get("launch") or {}).get("cap_usd") or default_cap,
            "default_gpu_cap_usd": default_cap,
            "expected_gpu_usd_if_uninterrupted": (offers[0]["glab_hourly_total"] * 1.6 * jobs) if offers else None,
            "judge_api_bounds_usd_per_model": judge_bounds,
            "note": "Each rental reserves its bound (hourly cap x deadline) against the GPU cap before renting; "
                    "the offer is re-searched at rental time within the same bounds.",
        },
    }


class PseudoRun(SimpleNamespace):
    def persist(self):
        pass


def preflight(study, state, plan_data):
    """One rented RTX 5090: CUDA check plus one real training step; selects the microbatch."""
    spec = training_specs(study, state)[0]
    if (state.get("preflight") or {}).get("data_hash") == state["dataset_hash"]:
        return state["preflight"]
    run_id = "preflight-" + digest([spec.model_dump(mode="json"), state["dataset_hash"]])[:12]
    directory = study.runs_root / run_id
    resolved = spec.model_copy(deep=True)
    write(directory / "config/resolved.json", resolved.model_dump(mode="json"))
    write(directory / "manifest.json", {"run_id": run_id, "status": "preflight", "created_at": now()})
    (directory / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(study.data_dir / "treatment.jsonl", directory / "data/train.jsonl")
    runner = PseudoRun(root=study.runs_root, run_id=run_id, directory=directory, spec=resolved,
                       manifest=read(directory / "manifest.json"), force=False)
    backend, _ = backend_for(study, state)
    attempt = budget.reserve(study, run_id, budget.attempt_bound(spec), "preflight")
    try:
        lease, journal, _, _, error, cleanup_error = backend._rent(runner, "preflight", record_manifest=False)
    finally:
        budget.settle(study, attempt)
    if cleanup_error:
        raise RuntimeError(f"Preflight instance teardown unconfirmed; run `em cleanup {study.name} --yes`")
    if error:
        raise RuntimeError(f"Preflight rental failed: {type(error).__name__}: {error}")
    result = read(directory / "preflight/result.json")
    problems = check_preflight(result, spec, state["settings"]["offline"])
    selected = result.get("selected")
    if problems or not selected:
        raise RuntimeError("Preflight failed; nothing was launched: " + "; ".join(problems or ["no microbatch fits"]))
    overrides, changes = {}, []
    if (selected["batch_size"], selected["gradient_accumulation"]) != (spec.training.batch_size,
                                                                       spec.training.gradient_accumulation):
        overrides = dict(selected)
        changes.append({"field": "batch_size/gradient_accumulation",
                        "v5": [spec.training.batch_size, spec.training.gradient_accumulation],
                        "study": [selected["batch_size"], selected["gradient_accumulation"]],
                        "reason": "measured OOM at microbatch 4 on the RTX 5090 preflight; effective batch stays 32"})
    record = {"data_hash": state["dataset_hash"], "run_id": run_id, "lease": lease, "result": result,
              "training_overrides": overrides, "recipe_changes": changes,
              "summary": {"device": result["cuda"].get("device"), "cuda": result["cuda"].get("torch_cuda"),
                          "selected_microbatch": selected, "attempts": [
                              {k: a.get(k) for k in ("batch_size", "gradient_accumulation", "fits", "peak_memory_gb",
                                                     "seconds")} for a in result["attempts"]]}}
    write(study.path / "launch/preflight.json", record)
    study.update(preflight=record)
    study.event("preflight", {"selected": selected, "changes": changes})
    return record


def check_preflight(result, spec, offline_mode):
    cuda = result["cuda"]
    if offline_mode:
        return []  # The tiny CPU fixture has no CUDA; the simulation only exercises the flow.
    problems = []
    if not cuda.get("cuda_available") or not cuda.get("matmul_ok"):
        problems.append("CUDA is unavailable or a bf16 matmul failed")
    if "5090" not in str(cuda.get("device")):
        problems.append(f"device is {cuda.get('device')!r}, not an RTX 5090")
    if (cuda.get("total_memory_gb") or 0) < spec.compute.min_gpu_ram_gb - 0.5:
        problems.append(f"only {cuda.get('total_memory_gb')} GB of GPU memory")
    if cuda.get("capability") and cuda["capability"][0] < 12:
        problems.append(f"compute capability {cuda['capability']} is not Blackwell (sm_120)")
    if "sm_120" not in " ".join(cuda.get("arch_list") or []):
        problems.append("the PyTorch build lacks sm_120 kernels (needs CUDA 12.8+ PyTorch)")
    return problems


def prepare_runs(study, state):
    """Engine preregistration, local data stages and protocol requests for every model."""
    training = training_specs(study, state)
    protocols = prereg.selected_protocols(study, state)
    jobs = []
    for spec in training:
        snapshot = engine_preregister(study.runs_root, spec)
        runner = Runner(spec, study.runs_root, snapshot)
        runner.execute(until="quality_check_data")
        if runner.manifest.get("control_execution") != "separate":
            runner.use_separate_control()
        control_id = runner.manifest["control_run_id"]
        for run_id, role in ((runner.run_id, "treatment"), (control_id, "control")):
            for definition in protocols:
                store.request(study.runs_root, run_id, definition)
            jobs.append({
                "run_id": run_id, "role": role, "training_seed": spec.training.seed,
                "pair": runner.run_id, "control_run_id": control_id,
                "config": str(study.runs_root / run_id / "config/resolved.json"),
                "seed": run_id,  # A unique key makes every job independent (concurrent controls).
                "status": "pending",
                "command": [sys.executable, "-m", "generalization_lab.wizard.worker", study.name, run_id],
            })
    write(study.path / "launch/jobs.json", {"created_at": now(), "jobs": jobs,
                                            "protocols": [{"id": p["id"], "sha256": p["sha256"]} for p in protocols]})
    return jobs


def start(study, state, cap, foreground=False, simulate_interruption=False, max_concurrent=None):
    """Set the cap, preflight, prepare runs and start (or resume) the orchestrator."""
    running = study.path / "launch/orchestrator.json"
    if running.is_file() and Path(f"/proc/{read(running)['pid']}").exists():
        cmdline = Path(f"/proc/{read(running)['pid']}/cmdline").read_bytes()
        if b"generalization_lab.wizard.orchestrate" in cmdline:
            raise ValueError(f"The orchestrator for {study.name} is already running (pid {read(running)['pid']}); "
                             f"follow it with `em status {study.name}`")
    budget.set_cap(study, cap)
    study.update(launch={"cap_usd": cap, "started_at": (state.get("launch") or {}).get("started_at") or now()},
                 phase="launched")
    if state["settings"]["offline"]:
        offline.prepare_ssh_key(study)
    preflight(study, study.load(), None)
    state = study.load()
    if (study.path / "launch/jobs.json").is_file():
        jobs = read(study.path / "launch/jobs.json")["jobs"]
    else:
        jobs = prepare_runs(study, state)
    if simulate_interruption:
        if not state["settings"]["offline"]:
            raise ValueError("--simulate-interruption is only available for offline studies")
        backend_for(study, state)[1].plan_interruption(jobs[0]["run_id"])
    study.event("launched", {"cap_usd": cap, "jobs": [j["run_id"] for j in jobs]})
    command = [sys.executable, "-m", "generalization_lab.wizard.orchestrate", study.name]
    if max_concurrent:
        command += ["--max-concurrent", str(max_concurrent)]
    log = study.path / "launch/orchestrator.log"
    if foreground:
        return subprocess.run(command, cwd=ROOT, env=os.environ.copy()).returncode
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
    write(study.path / "launch/orchestrator.json", {"pid": process.pid, "started_at": now(), "log": str(log)})
    return 0


def judge_budgets():
    return dict(judges.BUDGETS_USD)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))

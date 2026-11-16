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
from ..io import ROOT, digest, jsonl, now, read, rows, write
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
    recipe_id = state["settings"].get("recipe", "v5")
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
    per_rental = defaults.profile(recipe_id)["models_per_rental"]
    rentals = math.ceil(jobs / per_rental)
    rental_bound = group_bound(spec, recipe_id, min(per_rental, jobs)) if per_rental > 1 else bound
    default_cap = math.ceil(rentals * 2 * rental_bound + bound)
    expected_hours = (group_minutes(recipe_id, min(per_rental, jobs))[1] / 60 * rentals if per_rental > 1
                      else 1.6 * jobs)
    judge_bounds = {p["id"]: judges.BUDGETS_USD.get(p["id"], judges.BUDGETS_USD["custom"]) for p in protocols}
    c = spec.compute
    return {
        "study": study.name,
        "offline_simulation": state["settings"]["offline"],
        "dataset_hash": state["dataset_hash"],
        "preregistration_sha256": registration["sha256"],
        "training_recipe": {"id": recipe_id, "values": spec.training.model_dump(mode="json"),
                            "changes_from_recipe": defaults.changes_from_v5(spec.training.model_dump(mode="json"),
                                                                            state["settings"]["offline"],
                                                                            recipe_id=recipe_id),
                            "preflight": (state.get("preflight") or {}).get("summary")},
        "evaluation": [{"id": p["id"], "sha256": p["sha256"], "role": p["role"]} for p in protocols],
        "runs": {"seeds": state["settings"]["seeds"], "treatment": len(training), "same_seed_benign_control": len(training),
                 "total_models": jobs, "concurrency": "all treatment and control jobs launch concurrently"
                 if per_rental == 1 else f"{rentals} rental(s), each training and evaluating up to {per_rental} "
                 "models in turn (the base model is downloaded once per rental)"},
        "rental": {"type": "interruptible (bid); explicit bid price in every creation request" if c.rental == "bid"
                   else "on-demand (not interruptible); no bid price in the creation request",
                   "gpu": c.gpu_preferences, "min_gpu_ram_gb": c.min_gpu_ram_gb, "min_reliability": c.min_reliability,
                   "max_bid_per_hour": c.max_bid_per_hour, "bid_margin": c.bid_margin,
                   "max_hourly_incl_storage": c.max_price_per_hour, "disk_gb": c.disk_gb,
                   "rental_deadline_minutes": c.timeout_minutes, "image": c.image,
                   "group_deadline_minutes": group_minutes(recipe_id, min(per_rental, jobs))[0] if per_rental > 1
                   else None,
                   "no_substitution": "No other GPU, multi-GPU offer, or rental type is ever used."},
        "offers": [offer_row(o) for o in offers[:5]],
        "offer_search_error": error,
        "selected_offer": offer_row(offers[0]) if offers else None,
        "costs": {
            "per_rental_attempt_bound_usd": rental_bound,
            "attempts_per_job_max": defaults.MAX_LAUNCHES,
            "preflight_bound_usd": bound,
            "gpu_spending_cap_usd": max_total_usd or (state.get("launch") or {}).get("cap_usd") or default_cap,
            "default_gpu_cap_usd": default_cap,
            "expected_gpu_usd_if_uninterrupted": (offers[0]["glab_hourly_total"] * expected_hours) if offers else None,
            "judge_api_bounds_usd_per_model": judge_bounds,
            "note": "Each rental reserves its bound (hourly cap x deadline) against the GPU cap before renting; "
                    "the offer is re-searched at rental time within the same bounds.",
        },
    }


class PseudoRun(SimpleNamespace):
    def persist(self):
        pass


def preflight_rows(study, recipe_id):
    """The rows whose longest window the preflight measures: treatment rows (v5), or treatment and control
    rows for largest_fit recipes, whose gradient checkpointing leaves no fallback for a longer window."""
    names = ["treatment.jsonl"]
    if defaults.recipe_info(recipe_id)["microbatch_rule"] == "largest_fit":
        names.append("control.jsonl")
    return [row for name in names for row in rows(study.data_dir / name)]


def worst_window(spec, data_rows):
    """Token counts of the longest effective-batch window (what one_step measures)."""
    from ..training import encode_row, load_tokenizer

    tok = load_tokenizer(spec)
    encoded = sorted((encode_row(tok, spec, r) for r in data_rows), key=lambda r: -len(r["input_ids"]))
    window = encoded[: spec.training.batch_size * spec.training.gradient_accumulation]
    return {"window_tokens": sum(len(r["input_ids"]) for r in window),
            "window_supervised_tokens": sum(sum(v != -100 for v in r["labels"][1:]) for r in window),
            "longest_row_tokens": len(window[0]["input_ids"])}


def shared_preflight_path(spec, recipe_id):
    """One measured preflight per recipe, base model, training settings and GPU list, shared by studies."""
    from .study import studies_root

    training = {k: v for k, v in spec.training.model_dump(mode="json").items()
                if k not in {"batch_size", "gradient_accumulation", "seed"}}
    key = digest([recipe_id, spec.base_model.model_dump(mode="json"), training, spec.compute.gpu_preferences])
    return studies_root() / "_preflight" / f"{recipe_id}-{key[:16]}.json"


def reuse_preflight(study, state, spec, recipe_id, window):
    """A shared measurement covers this study when its worst window is no longer in any token count."""
    path = shared_preflight_path(spec, recipe_id)
    if defaults.recipe_info(recipe_id)["microbatch_rule"] != "largest_fit" or not path.is_file():
        return None
    shared = read(path)
    measured = shared["window"]
    if any(window[k] > measured[k] for k in window):
        return None
    return {**shared["record"], "data_hash": state["dataset_hash"], "window": window,
            "reused_from": {"study": shared["study"], "path": str(path.relative_to(ROOT)), "measured_window": measured}}


def preflight(study, state, plan_data):
    """One rented GPU: CUDA check plus one real training step; selects the microbatch.

    For largest_fit recipes a study reuses an earlier study's measurement (same recipe, base model, training
    settings and GPUs) when its own worst window is no longer; otherwise it measures and shares its own."""
    spec = training_specs(study, state)[0]
    if (state.get("preflight") or {}).get("data_hash") == state["dataset_hash"]:
        return state["preflight"]
    recipe_id = state["settings"].get("recipe", "v5")
    info = defaults.recipe_info(recipe_id)
    data_rows = preflight_rows(study, recipe_id)
    window = worst_window(spec, data_rows) if info["microbatch_rule"] == "largest_fit" else None
    reused = reuse_preflight(study, state, spec, recipe_id, window) if window else None
    if reused:
        write(study.path / "launch/preflight.json", reused)
        study.update(preflight=reused)
        study.event("preflight", {"selected": reused["result"]["selected"], "changes": reused["recipe_changes"],
                                  "reused_from": reused["reused_from"]["study"]})
        return reused
    run_id = "preflight-" + digest([spec.model_dump(mode="json"), state["dataset_hash"]])[:12]
    directory = study.runs_root / run_id
    resolved = spec.model_copy(deep=True)
    write(directory / "config/resolved.json", resolved.model_dump(mode="json"))
    write(directory / "manifest.json", {"run_id": run_id, "status": "preflight", "created_at": now()})
    (directory / "data").mkdir(parents=True, exist_ok=True)
    if info["microbatch_rule"] == "largest_fit":
        jsonl(directory / "data/train.jsonl", data_rows)
        write(directory / "preflight/candidates.json", {"candidates": [list(c) for c in info["fallbacks"]]})
    else:
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
    effective = spec.training.batch_size * spec.training.gradient_accumulation
    if (selected["batch_size"], selected["gradient_accumulation"]) != (spec.training.batch_size,
                                                                       spec.training.gradient_accumulation):
        overrides = dict(selected)
        reason = (f"largest microbatch that fits on the {result['cuda'].get('device')} preflight"
                  if info["microbatch_rule"] == "largest_fit" else
                  f"measured OOM at microbatch 4 on the {result['cuda'].get('device')} preflight")
        changes.append({"field": "batch_size/gradient_accumulation",
                        recipe_id: [spec.training.batch_size, spec.training.gradient_accumulation],
                        "study": [selected["batch_size"], selected["gradient_accumulation"]],
                        "reason": f"{reason}; effective batch stays {effective}"})
    record = {"data_hash": state["dataset_hash"], "run_id": run_id, "lease": lease, "result": result,
              "training_overrides": overrides, "recipe_changes": changes,
              "summary": {"device": result["cuda"].get("device"), "cuda": result["cuda"].get("torch_cuda"),
                          "selected_microbatch": selected,
                          "linear_attention_kernels": result.get("linear_attention_kernels"), "attempts": [
                              {k: a.get(k) for k in ("batch_size", "gradient_accumulation", "fits", "peak_memory_gb",
                                                     "seconds")} for a in result["attempts"]]}}
    if window:
        record["window"] = window
        write(shared_preflight_path(spec, recipe_id), {"study": study.name, "window": window, "record": record})
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
    wanted = spec.compute.gpu_preferences
    if not any(device_matches(name, cuda.get("device")) for name in wanted):
        problems.append(f"device is {cuda.get('device')!r}, not one of {wanted}")
    if (cuda.get("total_memory_gb") or 0) < spec.compute.min_gpu_ram_gb - 0.5:
        problems.append(f"only {cuda.get('total_memory_gb')} GB of GPU memory")
    for name, version in (result.get("packages") or {}).items():
        if str(version).startswith("ERROR"):
            problems.append(f"{name} does not import on the GPU host: {version}")
    kernels = result.get("linear_attention_kernels")
    if kernels and not kernels["fast"]:
        problems.append(f"Gated DeltaNet layers use the slow torch fallback ({kernels['module']}), not fla")
    if any("5090" in name or "PRO 6000" in name for name in wanted):
        # Blackwell-specific: only required when the study targets a Blackwell GPU (5090, RTX PRO 6000).
        if cuda.get("capability") and cuda["capability"][0] < 12:
            problems.append(f"compute capability {cuda['capability']} is not Blackwell (sm_120)")
        if "sm_120" not in " ".join(cuda.get("arch_list") or []):
            problems.append("the PyTorch build lacks sm_120 kernels (needs CUDA 12.8+ PyTorch)")
    return problems


def device_matches(offer_name, device):
    """Vast offer names ("RTX PRO 6000 WS", "RTX 5090") against CUDA device names
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA GeForce RTX 5090")."""
    import re

    core = re.sub(r"\s+(WS|S|Max-Q)$", "", offer_name)
    model = core.rsplit(" ", 1)[-1]
    device = str(device)
    return model in device and ("PRO" not in core or "PRO" in device)


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
                                            "groups": rental_groups(study, state, jobs, training[0]),
                                            "protocols": [{"id": p["id"], "sha256": p["sha256"]} for p in protocols]})
    return jobs


def rental_groups(study, state, jobs, spec):
    """Models that share one rental, trained and evaluated in turn (empty: one rental per model). Each group
    records its rental deadline and spending bound, fixed at launch."""
    recipe_id = state["settings"].get("recipe", "v5")
    size = defaults.profile(recipe_id)["models_per_rental"]
    if size <= 1:
        return []
    ids = [j["run_id"] for j in jobs]
    return [{"id": f"{study.name}-rental-{i // size}", "run_ids": ids[i:i + size],
             "deadline_minutes": group_minutes(recipe_id, len(ids[i:i + size]))[0],
             "bound_usd": group_bound(spec, recipe_id, len(ids[i:i + size]))}
            for i in range(0, len(ids), size)]


def group_minutes(recipe_id, models):
    """(rental deadline, expected wall clock) in minutes for one rental of ``models`` models."""
    info = defaults.profile(recipe_id)
    return (info["deadline_setup_minutes"] + models * info["deadline_minutes_per_model"],
            info["setup_minutes"] + models * info["minutes_per_model"])


def group_bound(spec, recipe_id, models):
    return spec.compute.max_price_per_hour * group_minutes(recipe_id, models)[0] / 60 + spec.compute.transfer_budget_usd


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

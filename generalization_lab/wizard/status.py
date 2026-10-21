"""`em status` and `em cleanup`: progress, failures, interruptions, rentals and spending."""

from __future__ import annotations

import json
from pathlib import Path

from ..engine import STAGES
from ..io import read, rows
from ..protocols import store
from . import budget


def leases(runs_root, run_id):
    result = []
    for path in sorted(Path(runs_root, "_vast").glob(f"{run_id}-*/lease.json")):
        lease = read(path)
        result.append({k: lease.get(k) for k in ("instance_id", "status", "purpose", "rental", "bid_price_per_hour",
                                                  "hourly_total_usd", "approximate_gpu_usd", "created_at",
                                                  "interrupted")}
                      | {"failure": (lease.get("failure") or {}).get("message")})
    return result


def live_workers(runs_root, run_ids):
    from .launch import load_script

    queue = load_script("parallel_gpu")
    return {run_id for run_id in run_ids if queue.held_elsewhere(runs_root, run_id)}


def report(study, live=True):
    state = study.load()
    out = {"study": study.name, "phase": state["phase"], "offline": state["settings"]["offline"],
           "evaluation": state["settings"]["evaluation"]}
    out["drafts"] = []
    for number in study.drafts():
        draft = read(study.draft_dir(number) / "draft.json")
        out["drafts"].append({"draft": number, "feedback": draft["feedback"],
                              "previewed": (study.draft_dir(number) / "preview/metadata.json").is_file()})
    out["dataset_hash"] = state.get("dataset_hash")
    out["preregistration_sha256"] = state.get("preregistration_sha256")
    if state.get("preflight"):
        out["preflight"] = state["preflight"]["summary"] | {"recipe_changes": state["preflight"]["recipe_changes"]}
    jobs_path = study.path / "launch/jobs.json"
    if not jobs_path.is_file():
        out["next"] = next_step(study, state)
        return out
    jobs = read(jobs_path)["jobs"]
    queue = study.path / "launch/queue/state.json"
    queued = {j["run_id"]: j for j in read(queue)["jobs"]} if queue.is_file() else {}
    run_ids = [j["run_id"] for j in jobs]
    alive = live_workers(study.runs_root, run_ids) if live else set()
    table = []
    for job in jobs:
        run_id = job["run_id"]
        directory = study.runs_root / run_id
        manifest = read(directory / "manifest.json") if (directory / "manifest.json").is_file() else {}
        stages = read(directory / "state.json")["stages"] if (directory / "state.json").is_file() else {}
        resume = [r.get("resume_from_step") for r in rows(directory / "training/loss.jsonl")]
        run_leases = leases(study.runs_root, run_id)
        table.append({
            "run_id": run_id, "role": job["role"], "seed": job["training_seed"],
            "queue_status": queued.get(run_id, {}).get("status", "not_started"),
            "launches": queued.get(run_id, {}).get("launches", 0),
            "worker_alive": run_id in alive,
            "done": (study.path / "launch/done" / f"{run_id}.json").is_file(),
            "engine_status": manifest.get("status"),
            "engine_stages_complete": sum(stages.get(s, {}).get("status") == "complete" for s in STAGES),
            "engine_failure": (manifest.get("failure") or {}).get("message"),
            "training_resumed_from_steps": sorted({s for s in resume if s}),
            "protocols": store.progress(study.runs_root, run_id),
            "rentals": len(run_leases),
            "interruptions": sum(bool(lease["interrupted"]) for lease in run_leases),
            "live_rentals": [lease["instance_id"] for lease in run_leases if lease["status"] in {"rented", "creating"}],
            "cleanup_required": [lease["instance_id"] for lease in run_leases if lease["status"] == "cleanup_required"],
            "gpu_usd": sum(lease["approximate_gpu_usd"] or 0 for lease in run_leases),
            "last_failure": queued.get(run_id, {}).get("log_tail") or next(
                (lease["failure"] for lease in reversed(run_leases) if lease["failure"]), None),
        })
    out["jobs"] = table
    out["spending"] = budget.reconcile(study, alive) if live else budget.summary(study)
    out["spending"]["judge_api_usd"] = judge_spend(study, run_ids)
    if live:
        out["live_instances"] = live_instances(study, state, run_ids)
        out["orphaned_instances"] = [i for i in out["live_instances"] if i["run_id"] not in alive]
    final = study.path / "launch/final.json"
    if final.is_file():
        out["final"] = read(final)
    out["next"] = next_step(study, state, out)
    return out


def judge_spend(study, run_ids):
    total = 0.0
    for run_id in run_ids:
        for path in store.run_store(study.runs_root, run_id).glob("*/logs/usage.jsonl"):
            total += sum(r["cost_usd"] for r in {r["id"]: r for r in rows(path)}.values())
    return total


def live_instances(study, state, run_ids):
    from .launch import backend_for

    _, client = backend_for(study, state, patched=False)
    wanted = set(run_ids) | {(state.get("preflight") or {}).get("run_id")}
    try:
        instances = client.instances()
    except Exception as error:  # noqa: BLE001 - status must still render without the API
        return [{"error": f"{type(error).__name__}: {error}"}]
    result = []
    for instance in instances:
        parts = str(instance.get("label") or "").split(":")
        if len(parts) >= 2 and parts[1] in wanted and instance.get("actual_status") != "destroyed":
            result.append({"instance_id": instance["id"], "run_id": parts[1], "status": instance.get("actual_status"),
                           "gpu": instance.get("gpu_name"), "is_bid": instance.get("is_bid")})
    return result


def cleanup(study, yes=False, include_live_workers=False):
    state = study.load()
    jobs = read(study.path / "launch/jobs.json")["jobs"] if (study.path / "launch/jobs.json").is_file() else []
    run_ids = [j["run_id"] for j in jobs]
    alive = set() if include_live_workers else live_workers(study.runs_root, run_ids)
    candidates = [i for i in live_instances(study, state, run_ids) if "error" not in i and i["run_id"] not in alive]
    if yes:
        from .launch import backend_for

        _, client = backend_for(study, state)
        for instance in candidates:
            client.destroy(instance["instance_id"])
    return {"candidates": candidates, "destroyed": yes, "skipped_live_workers": sorted(alive)}


def next_step(study, state, out=None):
    name = study.name
    if state["phase"] == "drafting":
        if not study.drafts():
            return f"em data revise {name} --feedback '...'"
        number = study.drafts()[-1]
        if not (study.draft_dir(number) / "preview/metadata.json").is_file():
            return f"em data preview {name}"
        return f"em data revise {name} --feedback '...'   or   em data approve {name}"
    if state["phase"] == "data_approved":
        return f"em data approve {name} (prediction / probes still needed)"
    if state["phase"] == "preregistered":
        return f"em launch {name}"
    if out and any(not j["done"] for j in out.get("jobs", [])):
        return f"em launch {name}   (resumes; safe to rerun)   |   em cleanup {name} --yes"
    return f"em results {name}"


def render(value):
    return json.dumps(value, indent=2, default=str)

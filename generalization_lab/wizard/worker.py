"""One wizard job (a treatment or a same-seed benign control), run by the parallel Vast queue.

Usage: python -m generalization_lab.wizard.worker STUDY RUN_ID
       python -m generalization_lab.wizard.worker STUDY --group GROUP_ID

Within the study's spending cap: rent one GPU, train (resuming from off-instance checkpoints after
an interruption, when the recipe keeps any) and run the requested protocol GPU work in the same
rental, then judge the saved responses locally. A job is done when ``launch/done/RUN_ID.json``
exists; the orchestrator then finishes the engine's local stages for each control/treatment pair.

A results-only run without a Hub copy has its adapter only on the rented host. If that host is
lost after training but before every protocol's GPU work finished, the run trains again from step 0:
its training outputs and every protocol output (all from the lost adapter) are moved to
``RUNS/_superseded/RUN_ID/TIMESTAMP/`` first, so no result mixes two checkpoints.

``--group`` runs a rental group (recipes with ``models_per_rental`` > 1): one rental trains and
evaluates each unfinished model of the group in turn, so the base model and the image are fetched once
per rental. Judging runs locally after the rental is destroyed. A failure stops the group and destroys
the instance; the queue retries the group, which skips models already done.
"""

from __future__ import annotations

import shutil
import sys
import time
import uuid

from ..compute.vast import ReusedVastClient, protocols_pending
from ..engine import STAGES, load_runner
from ..io import hub_backed, now, read, write
from ..protocols import judges, store
from ..secrets import load_project_credentials
from . import budget
from .launch import backend_for
from .study import Study


def needs_rental(runner):
    state = read(runner.directory / "state.json")["stages"]
    return state.get("neutral_eval", {}).get("status") != "complete" or protocols_pending(runner)


def adapter_lost(runner):
    """Trained, protocol GPU work unfinished, and the adapter is neither local nor on the Hub."""
    stages = read(runner.directory / "state.json")["stages"]
    return (stages.get("train", {}).get("status") == "complete" and protocols_pending(runner)
            and not hub_backed(runner.directory) and not (runner.directory / "checkpoints/adapter").is_dir())


def restart_training(runner):
    """Archive everything the lost adapter produced and reset the run to retrain from step 0."""
    stamp = now().replace(":", "").replace("+", "Z")[:17]
    archive = runner.root / "_superseded" / runner.run_id / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("training", "checkpoints", "evals", "analysis"):
        if (runner.directory / name).exists():
            shutil.move(str(runner.directory / name), str(archive / name))
            (runner.directory / name).mkdir()
    evaluations = store.run_store(runner.root, runner.run_id)
    for request in store.requests(runner.root, runner.run_id):
        folder = evaluations / request["slug"]
        for path in list(folder.iterdir()):
            if path.name != "request.json":
                (archive / "_protocol_evals" / request["slug"]).mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive / "_protocol_evals" / request["slug"] / path.name))
    state = read(runner.directory / "state.json")
    first = STAGES.index("train")
    state["stages"] = {k: v for k, v in state["stages"].items() if k in STAGES[:first]}
    write(runner.directory / "state.json", state)
    manifest = read(runner.directory / "manifest.json")
    for key in ("checkpoint_hash", "cuda_version", "gpu_model", "failure"):
        manifest.pop(key, None)
    manifest["status"] = "partial"
    manifest.setdefault("training_restarts", []).append(
        {"at": now(), "reason": "results-only adapter lost before protocol GPU work finished", "archive": str(archive)})
    runner.manifest = manifest
    runner.persist()
    return archive


def run(name, run_id):
    study = Study(name)
    state = study.load()
    done = study.path / "launch/done" / f"{run_id}.json"
    if done.is_file():
        return read(done)
    backend, _ = backend_for(study, state)
    runner = load_runner(study.runs_root, run_id)
    if adapter_lost(runner):
        print(f"{run_id}: adapter lost with its rental; retraining from step 0 ({restart_training(runner)})",
              flush=True)
        runner = load_runner(study.runs_root, run_id)
    if needs_rental(runner):
        attempt = budget.reserve(study, run_id, budget.attempt_bound(runner.spec), "train+protocol GPU")
        try:
            backend.run(runner, until=until_for(runner))
        finally:
            budget.settle(study, attempt)
    status = judges.run_judges(study.runs_root, run_id, offline=state["settings"]["offline"])
    record = {"run_id": run_id, "completed_at": now(), "judges": status,
              "protocols": store.progress(study.runs_root, run_id)}
    write(done, record)
    return record


def until_for(runner):
    # The remote part always stops at neutral_eval. A results-only run must ask for a later stage:
    # the backend refuses results_only when the run itself stops at neutral_eval, because a
    # results-only transfer leaves no local adapter for later GPU stages (there are none after it).
    return "free_response_eval" if runner.spec.compute.export_mode == "results_only" else "neutral_eval"


def run_group(name, group_id):
    """Train and evaluate a rental group's unfinished models on one rental, then judge them locally."""
    study = Study(name)
    state = study.load()
    group = next(g for g in read(study.path / "launch/jobs.json")["groups"] if g["id"] == group_id)
    todo = [r for r in group["run_ids"] if not (study.path / "launch/done" / f"{r}.json").is_file()]
    runners = []
    for run_id in todo:
        runner = load_runner(study.runs_root, run_id)
        if adapter_lost(runner):
            print(f"{run_id}: adapter lost with its rental; retraining from step 0 ({restart_training(runner)})",
                  flush=True)
            runner = load_runner(study.runs_root, run_id)
        if needs_rental(runner):
            runners.append(runner)
    if runners:
        backend, _ = backend_for(study, state)
        deadline = time.time() + group["deadline_minutes"] * 60
        shared = ReusedVastClient(backend.client, study.runs_root / "_vast" / f"{group_id}-{uuid.uuid4().hex[:8]}",
                                  deadline, watchdog=backend.watchdog)
        backend.client = shared
        attempt = budget.reserve(study, group_id, group["bound_usd"], f"{len(runners)} models on one rental",
                                 run_ids=[r.run_id for r in runners])
        try:
            for runner in runners:
                print(f"[{now()}] {group_id}: {runner.run_id}", flush=True)
                backend.run(runner, until=until_for(runner))
        finally:
            try:
                shared.close()
            finally:
                budget.settle(study, attempt)
    failures = []
    for run_id in todo:
        try:
            status = judges.run_judges(study.runs_root, run_id, offline=state["settings"]["offline"])
        except Exception as error:  # noqa: BLE001 - judge the others; the queue retries this group
            failures.append(f"{run_id}: {type(error).__name__}: {error}")
            continue
        write(study.path / "launch/done" / f"{run_id}.json",
              {"run_id": run_id, "completed_at": now(), "judges": status, "rental_group": group_id,
               "protocols": store.progress(study.runs_root, run_id)})
    if failures:
        raise RuntimeError("Judging failed: " + "; ".join(failures))
    return {"group": group_id, "models": todo}


def main():
    load_project_credentials()
    if len(sys.argv) == 4 and sys.argv[2] == "--group":
        print(run_group(sys.argv[1], sys.argv[3]), flush=True)
        return
    name, run_id = sys.argv[1:3]
    print(run(name, run_id), flush=True)


if __name__ == "__main__":
    main()

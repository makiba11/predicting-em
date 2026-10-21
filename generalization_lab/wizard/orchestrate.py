"""Run a launched study's jobs concurrently with the existing parallel Vast queue, then finish it.

Usage: python -m generalization_lab.wizard.orchestrate STUDY [--max-concurrent N]

Restarting is safe: finished jobs are recognised by their done markers, live workers by their locks
and processes (parallel_gpu.held_elsewhere), and failed jobs are retried only for transient causes
(no offer, SSH/transfer failures, deadline, interruption/outbid) up to the launch limit.
"""

from __future__ import annotations

import argparse

from ..batch import verify_completion
from ..engine import load_runner
from ..io import lock, now, read, write
from ..secrets import load_project_credentials
from . import budget, defaults
from .launch import backend_for, load_script
from .study import Study


def done(study):
    return lambda root, run_id: (study.path / "launch/done" / f"{run_id}.json").is_file()


def finalize(study):
    """Finish each seed's control, then its treatment (local engine stages), and publish."""
    jobs = read(study.path / "launch/jobs.json")["jobs"]
    is_done = done(study)
    finished, incomplete = [], []
    for treatment in [j for j in jobs if j["role"] == "treatment"]:
        pair = [treatment["control_run_id"], treatment["run_id"]]
        if not all(is_done(None, run_id) for run_id in pair):
            incomplete.append(treatment["run_id"])
            continue
        for run_id in pair:
            runner = load_runner(study.runs_root, run_id)
            runner.execute()
            spec = runner.spec
            if spec.compute.upload_private_hub:
                from ..hub import upload_run_and_control

                upload_run_and_control(runner.directory)
            verify_completion(runner.directory)
            finished.append(run_id)
    return finished, incomplete


def cleanup_orphans(study, state):
    """Destroy live rentals of this study that no worker holds (all workers have exited here)."""
    _, client = backend_for(study, state)
    run_ids = {j["run_id"] for j in read(study.path / "launch/jobs.json")["jobs"]}
    run_ids.add((state.get("preflight") or {}).get("run_id"))
    destroyed = []
    for instance in client.instances():
        parts = str(instance.get("label") or "").split(":")
        if len(parts) >= 2 and parts[1] in run_ids and instance.get("actual_status") != "destroyed":
            client.destroy(instance["id"])
            destroyed.append(instance["id"])
    return destroyed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("--max-concurrent", type=int, default=24)
    args = parser.parse_args()
    load_project_credentials()
    study = Study(args.study)
    state = study.load()
    queue = load_script("parallel_gpu")
    folder = study.path / "launch/queue"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = read(study.path / "launch/jobs.json")["jobs"]
    if (folder / "state.json").is_file():
        previous = {j["run_id"]: j for j in read(folder / "state.json")["jobs"]}
        for job in jobs:
            job["launches"] = previous.get(job["run_id"], {}).get("launches", 0)
    queue_state = {"study": study.name, "created_at": now(), "jobs": jobs}
    with lock(folder / ".orchestrator.lock"):
        failed = queue.run_all(jobs, queue_state, folder, study.runs_root, args.max_concurrent,
                               defaults.STAGGER_SECONDS if not state["settings"]["offline"] else 0,
                               defaults.MAX_LAUNCHES, complete_fn=done(study))
        destroyed = cleanup_orphans(study, state)
        budget.reconcile(study)
        finished, incomplete = finalize(study)
        from .results import build

        report = build(study)
        study.update(phase="complete" if not failed and not incomplete else "launched")
        write(study.path / "launch/final.json", {
            "at": now(), "failed_jobs": [j["run_id"] for j in failed], "incomplete_pairs": incomplete,
            "finished_runs": finished, "orphans_destroyed": destroyed, "results": report,
        })
    print(f"[{now()}] {study.name}: {len(jobs) - len(failed)}/{len(jobs)} jobs done; results {report}", flush=True)
    if failed:
        raise SystemExit(f"Failed jobs: {[j['run_id'] for j in failed]}")


if __name__ == "__main__":
    main()

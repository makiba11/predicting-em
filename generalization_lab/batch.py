"""Durable subprocess queue with exclusive ownership and conservative budget reservations."""

import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .analysis import compare_runs, seed_group_key
from .config import expand_matrix
from .costs import enforce
from .engine import STAGES, run_identity
from .io import ROOT, csv_write, digest, hub_backed, lock, now, read, sha, write
from .preregistration import preregister
from .results import group_name, results_root, verify_publication


def verify_completion(directory):
    """Do not trust an exit status without complete, intact result artifacts."""
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Worker exited without a complete manifest")
    result = read(directory / "analysis/results.json")
    if result.get("run_id") != directory.name or not (directory / "summary.md").is_file():
        raise ValueError("Worker result identity or summary is missing")
    from .config import Spec

    spec = Spec.model_validate(read(directory / "config/resolved.json"))
    state = read(directory / "state.json")
    for stage in STAGES:
        info = state.get("stages", {}).get(stage, {})
        if info.get("status") != "complete":
            raise ValueError(f"Worker left stage incomplete: {stage}")
        for filename, expected in info.get("outputs", {}).items():
            path = directory / filename
            if not path.is_file() and spec.compute.backend == "vast" and filename.startswith("checkpoints/"):
                if spec.compute.export_mode == "results_only" or not filename.startswith("checkpoints/adapter/"):
                    continue
                if hub_backed(directory):
                    continue
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Completed artifact missing or changed: {path}")
    verify_publication(spec, directory.parent, directory.name, manifest.get("is_control", False))
    if spec.compute.upload_private_hub:
        from .hub import verify_run_and_control

        verify_run_and_control(directory)


def summarize_batch(specs, state, root, directory):
    groups = {}
    for spec, job in zip(specs, state["jobs"]):
        key = seed_group_key(spec)[:16]
        group = groups.setdefault(
            key,
            {
                "experiment_id": spec.experiment.id,
                "results_group": group_name(spec),
                "jobs": [],
                "paths": [],
            },
        )
        group["jobs"].append({k: job[k] for k in ("run_id", "status")})
        if job["status"] == "complete":
            group["paths"].append(Path(root) / job["run_id"])
    reports, table, completed_reports = [], [], []
    for key, group in groups.items():
        paths = group.pop("paths")
        report = {
            **group,
            "completed": len(paths),
            "planned": len(group["jobs"]),
            "comparison": compare_runs(paths) if paths else None,
        }
        report_path = directory / "analysis" / f"seeds-{key}.json"
        write(report_path, report)
        reports.append(str(report_path))
        if report["completed"] == report["planned"]:
            completed_reports.append((group["results_group"], report_path.name, report))
        if report["comparison"]:
            for suite, metrics in report["comparison"]["aggregate_deltas"].get("overall", {}).items():
                for metric, stats in metrics.items():
                    table.append(
                        {
                            "group": key,
                            "experiment_id": group["experiment_id"],
                            "results_group": group["results_group"],
                            "suite": suite,
                            "metric": metric,
                            **stats,
                        }
                    )
    csv_write(directory / "analysis/seed_deltas.csv", table)
    for name in sorted({group_name(spec) for spec in specs}):
        jobs = [job for spec, job in zip(specs, state["jobs"]) if group_name(spec) == name]
        if not jobs or any(job["status"] != "complete" for job in jobs):
            continue
        destination = results_root(root) / name / "batches" / directory.name / "analysis"
        for report_group, filename, report in completed_reports:
            if report_group == name:
                write(destination / filename, report)
        csv_write(destination / "seed_deltas.csv", [row for row in table if row["results_group"] == name])
    return reports


def prime_dataset_cache(spec, runner):
    """Reuse identical generator outputs across training seeds, with hash verification."""
    if spec.dataset.precomputed_data_path:
        return None
    generator = spec.dataset.generator or spec.providers.dataset_generator
    key = digest([spec.experiment.id, spec.dataset.model_dump(mode="json"), generator.model_dump(mode="json")])
    folder = Path(runner.root) / "_dataset_cache" / key[:24]
    manifest = folder / "manifest.json"
    source_names = ("requests.json", "attempts.jsonl")
    if manifest.is_file():
        saved = read(manifest)
        if saved.get("key") != key:
            raise ValueError("Dataset cache key mismatch")
        for name in source_names:
            source = folder / name
            if sha(source) != saved["files"][name]:
                raise ValueError("Dataset cache checksum mismatch")
            target = runner.directory / "data" / name
            if not target.exists():
                shutil.copyfile(source, target)
        return None
    return folder, key


def save_dataset_cache(runner, pending):
    if pending is None:
        return
    folder, key = pending
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("requests.json", "attempts.jsonl"):
        source = runner.directory / "data" / name
        shutil.copyfile(source, folder / name)
        files[name] = sha(folder / name)
    write(folder / "manifest.json", {"key": key, "source_run_id": runner.run_id, "files": files})


def run_reused_vast_batch(specs, state, root, directory, cap):
    """Prepare data before rental, then run every GPU job on one retained instance."""
    import time

    from .compute.vast import ReusedVastClient, VastBackend, VastClient
    from .engine import Runner, load_runner
    from .hub import upload_run_and_control

    def persist():
        write(directory / "queue.json", state)

    # All OpenRouter generation and QA happens before Vast starts billing.
    for spec, job in zip(specs, state["jobs"]):
        if job["status"] == "complete":
            verify_completion(Path(root) / job["run_id"])
            continue
        if job["status"] != "data_ready":
            reservation = enforce(spec)["projected_max_usd"]
            if "reserved_usd" not in job:
                if state["committed_usd"] + reservation > cap + 1e-9:
                    job["status"] = "budget_blocked"
                    persist()
                    break
                state["committed_usd"] += reservation
                job["reserved_usd"] = reservation
            job.update(status="generating_data", attempts=job["attempts"] + 1, updated_at=now())
            persist()
            try:
                runner = Runner(spec, root)
                cache_pending = prime_dataset_cache(spec, runner)
                runner.execute(until="quality_check_data")
                save_dataset_cache(runner, cache_pending)
            except BaseException as error:
                job.update(status="failed", failure=f"{type(error).__name__}: {error}", updated_at=now())
                persist()
                raise
            job.update(status="data_ready", updated_at=now())
            persist()
    if any(j["status"] not in {"data_ready", "complete"} for j in state["jobs"]):
        raise RuntimeError("Data preparation did not finish; no Vast instance was rented")

    remaining = [j for j in state["jobs"] if j["status"] != "complete"]
    if remaining:
        overall_deadline = time.time() + sum(
            spec.compute.timeout_minutes * 60 for spec, job in zip(specs, state["jobs"])
            if job["status"] != "complete"
        ) + 300
        shared = ReusedVastClient(VastClient(), directory / "shared_vast", overall_deadline)
        backend = VastBackend(client=shared)
        try:
            for spec, job in zip(specs, state["jobs"]):
                if job["status"] == "complete":
                    continue
                job.update(status="running", updated_at=now())
                persist()
                try:
                    runner = load_runner(root, job["run_id"])
                    path = backend.run(runner)
                    upload_run_and_control(path) if spec.compute.upload_private_hub else None
                    verify_completion(path)
                except BaseException as error:
                    job.update(status="failed", failure=f"{type(error).__name__}: {error}", updated_at=now())
                    persist()
                    raise
                job.update(status="complete", updated_at=now())
                persist()
        finally:
            shared.close()
            for lease_path in (Path(root) / "_vast").glob("*/lease.json"):
                lease = read(lease_path)
                if lease.get("status") == "retained_for_batch" and lease.get("label") == shared.label:
                    lease.update(status="destroyed", stopped_at=now())
                    write(lease_path, lease)
    state["seed_summaries"] = summarize_batch(specs, state, root, directory)
    from .report import write_batch_tables

    write_batch_tables(specs, state, root, directory)
    persist()
    return {**state, "queue_file": str(directory / "queue.json")}


def run_batch(path, root, backend=None, max_concurrent=None, max_total_cost_usd=None, dry_run=False):
    if not dry_run:
        from .secrets import load_project_credentials

        load_project_credentials()
    specs, options = expand_matrix(path)
    concurrency = max_concurrent if max_concurrent is not None else options.get("max_concurrent", 1)
    cap = max_total_cost_usd if max_total_cost_usd is not None else options.get("max_total_cost_usd", 10)
    retries = options.get("retries", 1)
    reuse_vast = options.get("reuse_vast_instance", False)
    if (
        not isinstance(concurrency, int)
        or concurrency < 1
        or not math.isfinite(cap)
        or cap <= 0
        or not isinstance(retries, int)
        or retries < 0
    ):
        raise ValueError("Queue concurrency/budget/retry count is invalid")
    for spec in specs:
        if backend:
            if backend not in {"local", "vast"}:
                raise ValueError("Backend must be local or vast")
            spec.compute.backend = backend
        enforce(spec)
    if not dry_run and any(spec.compute.upload_private_hub for spec in specs):
        from .hub import preflight_private_upload

        preflight_private_upload()
    queue_hash = digest([s.model_dump(mode="json") for s in specs])
    directory = Path(root).resolve() / "_queues" / queue_hash[:16]
    if dry_run:
        return {
            "queue_id": queue_hash[:16],
            "atomic_experiments": len(specs),
            "max_concurrent": concurrency,
            "batch_cap_usd": cap,
            "reuse_vast_instance": reuse_vast,
            "jobs": [
                {
                    "experiment": s.experiment.id,
                    "training_seed": s.training.seed,
                    "num_examples": s.dataset.num_examples,
                    "estimate": enforce(s),
                }
                for s in specs
            ],
        }
    with lock(directory / ".lock"):
        state_path = directory / "queue.json"
        if state_path.exists():
            state = read(state_path)
            if state["config_hash"] != queue_hash:
                raise ValueError("Queue config hash mismatch")
            if cap < state["committed_usd"]:
                raise ValueError("New budget is below already committed queue spend")
        else:
            state = {"config_hash": queue_hash, "created_at": now(), "committed_usd": 0, "jobs": []}
            # All snapshots are created before any job can generate data or reveal results.
            for spec in specs:
                snapshot = preregister(root, spec)
                run_id = run_identity(spec, snapshot)
                config_path = directory / (run_id + ".json")
                write(config_path, spec.model_dump(mode="json"))
                state["jobs"].append(
                    {"run_id": run_id, "config": str(config_path), "status": "pending", "attempts": 0}
                )
            write(state_path, state)
        if reuse_vast:
            if concurrency != 1 or any(s.compute.backend != "vast" for s in specs):
                raise ValueError("Vast reuse requires one queue worker and all jobs on Vast")
            return run_reused_vast_batch(specs, state, root, directory, cap)
        active = {}
        for job in state["jobs"]:
            if job["status"] == "running":
                # The queue lock was lost. A worker may still own its run lock; a retry will fail safely.
                job["status"] = "interrupted"
        try:
            while True:
                for index, spec in enumerate(specs):
                    job = state["jobs"][index]
                    manifest_path = Path(root) / job["run_id"] / "manifest.json"
                    if job["status"] == "complete" or (
                        job["status"] != "running"
                        and manifest_path.exists()
                        and read(manifest_path)["status"] == "complete"
                    ):
                        try:
                            verify_completion(manifest_path.parent)
                        except (ValueError, OSError, KeyError) as error:
                            job.update(status="artifact_invalid", failure=str(error))
                        else:
                            job["status"] = "complete"
                    if (
                        job["status"] in {"complete", "artifact_invalid"}
                        or index in active
                        or job["attempts"] >= retries + 1
                    ):
                        continue
                    if len(active) >= concurrency:
                        break
                    reservation = enforce(spec)["projected_max_usd"]
                    if state["committed_usd"] + reservation > cap + 1e-9:
                        job["status"] = "budget_blocked"
                        continue
                    state["committed_usd"] += reservation
                    job.update(status="running", attempts=job["attempts"] + 1, reserved_usd=reservation)
                    write(state_path, state)  # Durable reservation before process creation.
                    log = (directory / (job["run_id"] + ".log")).open("a")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "generalization_lab",
                            "_worker",
                            job["config"],
                            "--runs-root",
                            str(Path(root).resolve()),
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    job["pid"] = process.pid
                    active[index] = (process, log)
                for index, (process, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    job = state["jobs"][index]
                    job.update(status="complete" if code == 0 else "failed", exit_code=code, updated_at=now())
                    if code == 0:
                        try:
                            verify_completion(Path(root) / job["run_id"])
                        except (ValueError, OSError, KeyError) as error:
                            job.update(status="failed", failure=str(error))
                    log.close()
                    del active[index]
                    # Keep full reservations charged: uncertain failure, network and cleanup bills are not refunded silently.
                write(state_path, state)
                if not active:
                    pending = [
                        j
                        for j in state["jobs"]
                        if j["status"] in {"pending", "interrupted", "failed"} and j["attempts"] <= retries
                    ]
                    if not pending:
                        break
                print(
                    " | ".join(
                        f"{j['run_id']}: {j['status']} ({j['attempts']} attempts)" for j in state["jobs"]
                    ),
                    flush=True,
                )
                time.sleep(1)
        finally:
            for process, log in active.values():
                process.terminate()  # Vast child catches SIGTERM and runs finally cleanup.
            for process, log in active.values():
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    # Leave the deadline watchdog responsible; never start another duplicate worker.
                    pass
                log.close()
            write(state_path, state)
        state["seed_summaries"] = summarize_batch(specs, state, root, directory)
        from .report import write_batch_tables

        write_batch_tables(specs, state, root, directory)
        write(state_path, state)
        return {
            **state,
            "queue_file": str(state_path),
            "budget_policy": "Full per-attempt bounds remain reserved, even on failure.",
        }

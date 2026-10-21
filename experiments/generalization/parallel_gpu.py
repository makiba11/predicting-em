"""Run every GPU job of a matrix on its own Vast instance, in parallel.

Each training seed's first job also trains and evaluates that seed's matched benign control; the
seed's remaining jobs launch as soon as that job is complete and reuse the finished control. Launches are staggered because each worker rents the cheapest
matching offer; a job that finds no free offer is requeued. Each worker is the ordinary
`generalization_lab _worker` (via vast_worker.py, which only changes offer selection), so data checks, teardown, retrieval, Hub upload and publication are
unchanged. Afterwards run `generalization_lab batch MATRIX --backend vast` to compile the tables
(it finds the completed runs and launches nothing).
"""

import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path

from generalization_lab.config import expand_matrix
from generalization_lab.costs import enforce
from generalization_lab.engine import run_identity
from generalization_lab.io import ROOT, digest, lock, now, read, write
from generalization_lab.preregistration import preregister
from generalization_lab.secrets import load_project_credentials

# Retry when the rental never started (no offer, or it was taken first) or the input upload timed
# out (slow uplink); failed rentals are always destroyed.
RETRYABLE = (
    "No Vast offers satisfy", "Vast reported failure: ", "Vast HTTP 400: invalid_args", "timed out after",
    "Remote job reached rental deadline", "couldn't connect to 'https://huggingface.co'",
    "ConnectionError", "ConnectTimeout", "ReadTimeout",
    "unresolved free-response samples; rerun retries judging",
    "SSLError", "IncompleteRead", "ProtocolError", "Connection reset by peer",
    "SSH command failed with exit code 255", "Vast instance became",
    "Instance/SSH startup timeout", "SCP upload failed",
    # Interruptible (bid) rentals: outbid/stopped instances resume from off-instance checkpoints.
    "Vast instance interrupted",
    # Protocol judging (wizard jobs) keeps raw outputs and retries only the unresolved calls.
    "unresolved judge calls retained", "unresolved JSON judge calls retained",
    "unresolved rubric judge calls retained",
)
WORKERS = ("vast_worker.py", "adopt_orphan_rental.py", "generalization_lab.wizard.worker")


def retryable(root, job, tail):
    folder = Path(root) / job["run_id"]
    manifest = folder / "manifest.json"
    if manifest.is_file():
        failure = read(manifest).get("failure", {})
        if failure.get("stage") in {"quality_check_data", "quality_check_eval"}:
            return False
        tail += str(failure)
    poll = folder / "logs/remote_poll.jsonl"
    if poll.is_file():
        tail += poll.read_text()[-8000:]
    if any(marker in tail for marker in RETRYABLE):
        return True
    # An externally orphaned rental can expire without a worker to record a
    # manifest failure. Resume from the fetched stage data at its deadline.
    return any(
        read(p).get("deadline", float("inf")) < time.time()
        for p in (Path(root) / "_vast").glob(job["run_id"] + "*/lease.json")
    )


def complete(root, run_id):
    path = Path(root) / run_id / "manifest.json"
    return path.is_file() and read(path).get("status") == "complete"


def held_elsewhere(root, run_id):
    """True while another worker (e.g. from an earlier orchestrator) owns this run's rental, or is
    running its local stages (judging/analysis) after the rental, which hold only the run's .lock."""
    # Locks have short gaps between stages; a live worker or adopter process for the run is decisive.
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().decode(errors="replace")
        except OSError:
            continue
        if run_id in args and any(worker in args for worker in WORKERS):
            return True
    for name in (".backend.lock", ".lock"):
        path = Path(root) / run_id / name
        if not path.exists():
            continue
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
    # The remote supervisor can outlive a controller process. Keep its lease
    # reserved until the deadline so the queue cannot rent over live work.
    return any(
        read(p).get("status") == "rented" and read(p).get("deadline", 0) > time.time()
        for p in (Path(root) / "_vast").glob(run_id + "*/lease.json")
    )


def builder_ready(root, run_id):
    if complete(root, run_id):
        return True
    manifest = Path(root) / run_id / "manifest.json"
    if not manifest.is_file():
        return False
    control = read(manifest).get("control_run_id")
    return bool(control and complete(root, control))


def run_all(jobs, state, folder, root, max_concurrent, stagger, max_launches, complete_fn=None):
    """Launch every job as soon as it can run: a seed's first job builds its matched control,
    and the seed's other jobs start the moment that job (and so the control) is complete.

    A job may carry its own worker ``command``, and ``complete_fn(root, run_id)`` may define when
    a job is done (the wizard's separate-control jobs finish before their engine runs complete).
    Jobs whose ``seed`` key is unique are independent and launch concurrently.
    """
    complete = complete_fn or globals()["complete"]

    def persist():
        write(folder / "state.json", state)

    builders = {}
    for job in jobs:
        builders.setdefault(job["seed"], job["run_id"])
    for job in jobs:
        if complete(root, job["run_id"]):
            job["status"] = "complete"
    active, last_launch = {}, 0.0
    while True:
        pending = [j for j in jobs if j["status"] in {"pending", "running_elsewhere"}]
        if not pending and not active:
            break
        for job in pending:
            if job["status"] == "running_elsewhere" and complete(root, job["run_id"]):
                job.update(status="complete", finished_at=now())
                print(f"[{now()}] {job['run_id']}: complete (external worker)", flush=True)
                persist()
            elif job["status"] == "running_elsewhere" and not held_elsewhere(root, job["run_id"]):
                if retryable(root, job, "") and job.get("launches", 0) < max_launches:
                    job.update(status="pending", note="external worker stopped; resuming")
                else:
                    job.update(status="failed", finished_at=now(), note="external worker stopped incomplete")
                persist()
        for job in pending:
            external = sum(j["status"] == "running_elsewhere" for j in jobs)
            if job["status"] != "pending" or len(active) + external >= max_concurrent or time.time() - last_launch < stagger:
                continue
            if job["run_id"] != builders[job["seed"]] and not builder_ready(root, builders[job["seed"]]):
                continue  # Wait for this seed's matched control.
            if held_elsewhere(root, job["run_id"]):
                job["status"] = "running_elsewhere"
                persist()
                continue
            job["launches"] = job.get("launches", 0) + 1
            job.update(status="running", started_at=now())
            log = (folder / f"{job['run_id']}.log").open("a")
            active[job["run_id"]] = (
                subprocess.Popen(
                    job.get("command") or [
                        sys.executable, str(ROOT / "experiments/generalization/vast_worker.py"), job["config"],
                        "--runs-root", str(Path(root).resolve()),
                    ],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                ),
                log,
                job,
            )
            last_launch = time.time()
            print(f"[{now()}] launched {job['run_id']} (attempt {job['launches']})", flush=True)
            persist()
        for run_id, (process, log, job) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            del active[run_id]
            tail = (folder / f"{run_id}.log").read_text()[-3000:]
            if process.returncode == 0 and complete(root, run_id):
                job.update(status="complete", finished_at=now())
            elif retryable(root, job, tail) and job["launches"] < max_launches:
                job.update(status="pending", note="transient failure; requeued with saved progress")
                last_launch = time.time() + 60  # Give the market a minute before trying again.
            else:
                job.update(status="failed", exit_code=process.returncode, finished_at=now(), log_tail=tail[-800:])
            print(f"[{now()}] {run_id}: {job['status']}", flush=True)
            persist()
        failed_builders = {j["run_id"] for j in jobs if j["status"] == "failed" and not complete(root, j["run_id"])}
        for job in jobs:
            if job["status"] == "pending" and builders[job["seed"]] in failed_builders:
                job.update(status="blocked", note="matched control builder failed")
                persist()
        time.sleep(5)
    persist()
    return [job for job in jobs if job["status"] != "complete"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--max-concurrent", type=int, default=24)
    parser.add_argument("--stagger", type=float, default=40, help="Seconds between rentals")
    parser.add_argument("--max-launches", type=int, default=4, help="Per job, including no-offer requeues")
    args = parser.parse_args()
    load_project_credentials()
    specs, _ = expand_matrix(args.matrix)
    folder = args.runs_root / "_parallel_gpu" / digest([s.model_dump(mode="json") for s in specs])[:16]
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in specs:
        enforce(spec)
        run_id = run_identity(spec, preregister(args.runs_root, spec))
        config = folder / f"{run_id}.json"
        write(config, spec.model_dump(mode="json"))
        jobs.append({"run_id": run_id, "config": str(config), "seed": spec.training.seed, "status": "pending"})
    state = {"matrix": str(args.matrix), "created_at": now(), "jobs": jobs}
    if (folder / "state.json").is_file():
        previous = {job["run_id"]: job for job in read(folder / "state.json")["jobs"]}
        for job in jobs:
            job["launches"] = previous.get(job["run_id"], {}).get("launches", 0)
    with lock(folder / ".orchestrator.lock"):
        failed = run_all(jobs, state, folder, args.runs_root, args.max_concurrent, args.stagger, args.max_launches)
    print(f"[{now()}] done: {len(jobs) - len(failed)}/{len(jobs)} complete", flush=True)
    if failed:
        raise SystemExit(f"Failed jobs: {[job['run_id'] for job in failed]}")


if __name__ == "__main__":
    main()

"""Finish an orphaned Vast rental whose controller worker died while the remote job kept running.

Does what VastBackend.run's poll loop and teardown would have done: poll the remote runner until it
finishes or the rental deadline nears, retrieve results, destroy the instance, record accounting, then
either run the remaining local stages (remote work complete) or record the standard retryable
deadline failure so the parallel_gpu.py queue resumes the run with its saved progress.

Usage: python experiments/generalization/adopt_orphan_rental.py JOURNAL_DIR [--runs-root runs]
"""

import argparse
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from generalization_lab.compute.vast import SSHTransport, VastClient
from generalization_lab.engine import load_runner
from generalization_lab.io import lock, now, read, write
from generalization_lab.secrets import load_project_credentials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    load_project_credentials()
    journal = args.journal.resolve()
    lease = read(journal / "lease.json")
    if lease.get("status") != "rented":
        raise SystemExit(f"Lease is {lease.get('status')}, not rented")
    run_id, instance_id, deadline = lease["run_id"], lease["instance_id"], lease["deadline"]
    runner = load_runner(args.runs_root, run_id)
    spec = runner.spec
    client = VastClient()
    started = datetime.fromisoformat(lease["created_at"]).timestamp()
    previous_gpu_cost = sum(
        read(p).get("approximate_gpu_usd", 0)
        for p in (args.runs_root / "_vast").glob(run_id + "-*/lease.json")
        if p.parent.resolve() != journal
    )
    status, error = {}, None
    with lock(runner.directory / ".backend.lock"):
        try:
            transport = SSHTransport(client.instance(instance_id), spec, journal, deadline)
            poll_failures = 0
            while True:
                if time.time() >= deadline - 90:
                    raise TimeoutError("Remote job reached rental deadline; retrieving partial results and destroying")
                try:
                    status = transport.poll(runner)
                    poll_failures = 0
                except (RuntimeError, subprocess.TimeoutExpired):
                    poll_failures += 1
                    if poll_failures >= 3:
                        raise
                    time.sleep(spec.compute.poll_seconds)
                    continue
                print(f"[{now()}] [{run_id}] Vast {instance_id}: {status.get('status', 'unknown')}", flush=True)
                if status.get("status") in {"complete", "failed"}:
                    if status["status"] == "failed":
                        raise RuntimeError("Remote runner failed; see logs/remote_poll.jsonl and remote logs")
                    break
                time.sleep(spec.compute.poll_seconds)
        except BaseException as exc:
            error = exc
            lease["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            transport.fetch(runner)
        except BaseException as fetch_error:
            lease["partial_transfer_error"] = type(fetch_error).__name__
            print(f"[{now()}] fetch failed: {fetch_error!r}", flush=True)
        client.destroy(instance_id)
        lease.update(
            status="destroyed",
            stopped_at=now(),
            adopted_note="Controller worker died; adopt_orphan_rental.py polled, retrieved and destroyed.",
            approximate_gpu_usd=(time.time() - started) / 3600 * float(lease["offer"]["dph_total"]),
        )
        write(journal / "lease.json", lease)
        # The rental's deadline watchdog is no longer needed.
        for pid in os.listdir("/proc"):
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"generalization_lab.compute.watchdog" in cmd and str(instance_id).encode() in cmd:
                os.kill(int(pid), signal.SIGTERM)

        runner = load_runner(args.runs_root, run_id)
        runner.manifest["vast"] = {**(runner.manifest.get("vast") or {}), **lease}
        runner.manifest["costs"].update(
            gpu_usd=previous_gpu_cost + lease["approximate_gpu_usd"],
            api_usd=runner.ledger.spent(),
            known_total_usd=previous_gpu_cost + lease["approximate_gpu_usd"] + runner.ledger.spent(),
            transfer_usd=None,
            note="Rental approximation; transfer charges/taxes and provider invoice reconciliation are not included.",
        )
        remote_done = read(runner.directory / "state.json")["stages"].get("neutral_eval", {}).get("status") == "complete"
        if error or not remote_done:
            failure = error or RuntimeError("Remote job reached rental deadline; retrieved results incomplete")
            runner.manifest["status"] = "failed"
            runner.manifest["failure"] = {"stage": "vast", "type": type(failure).__name__, "message": str(failure), "at": now()}
            runner.persist()
            raise SystemExit(f"Adopted rental stopped before completion: {failure}")
        runner.persist()
    # Remote inference finished: only local stages remain (same as VastBackend.run).
    load_runner(args.runs_root, run_id).execute()
    print(f"[{now()}] [{run_id}] status: {read(args.runs_root / run_id / 'manifest.json')['status']}", flush=True)


if __name__ == "__main__":
    main()

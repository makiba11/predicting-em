"""Append compact progress updates for a detached generalization batch."""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def accepted_count(path):
    if not path.exists():
        return 0
    count = 0
    with path.open() as stream:
        for line in stream:
            try:
                count += bool(json.loads(line).get("accepted"))
            except json.JSONDecodeError:
                continue
    return count


def summarize(queue_path, runs_root):
    queue = read_json(queue_path)
    jobs = queue.get("jobs", [])
    if not jobs:
        return "waiting for queue setup", False
    states = Counter(job.get("status", "unknown") for job in jobs)
    current = next((job for job in jobs if job.get("status") in {"generating_data", "running"}), None)
    if current is None:
        current = next((job for job in jobs if job.get("status") not in {"complete", "pending"}), None)
    detail = ""
    if current:
        run = runs_root / current["run_id"]
        stage_state = read_json(run / "state.json").get("stages", {})
        stages = [name for name, value in stage_state.items() if value.get("status") == "running"]
        stage = stages[-1] if stages else current["status"]
        detail = f" run={current['run_id']} stage={stage}"
        attempts = run / "data/attempts.jsonl"
        if current["status"] == "generating_data" and attempts.exists():
            accepted = rejected = 0
            with attempts.open() as stream:
                for line in stream:
                    try:
                        accepted += bool(json.loads(line).get("accepted"))
                    except json.JSONDecodeError:
                        continue
                    rejected += 1
            detail += f" accepted={accepted}/2000 rejected={rejected - accepted}"
        if current["status"] == "failed":
            detail += f" failure={current.get('failure', 'unknown')}"
    prefetch_active = []
    prefetch_ready = 0
    for job in jobs:
        if job["status"] != "pending":
            continue
        run = runs_root / job["run_id"]
        stages = read_json(run / "state.json").get("stages", {})
        if stages.get("quality_check_data", {}).get("status") == "complete":
            prefetch_ready += 1
        elif stages.get("generate_data", {}).get("status") == "running":
            count = accepted_count(run / "data/attempts.jsonl")
            prefetch_active.append(f"{job['run_id'].split('-')[0]}:{count}")
        elif any(value.get("status") == "running" for value in stages.values()):
            prefetch_active.append(f"{job['run_id'].split('-')[0]}:later")
    if prefetch_active or prefetch_ready:
        detail += f" prefetch_ready={prefetch_ready} prefetch_active={','.join(prefetch_active)}"
    controls = []
    for control in runs_root.glob("control-*"):
        stages = read_json(control / "state.json").get("stages", {})
        if stages.get("generate_data", {}).get("status") == "running":
            controls.append((control.name, accepted_count(control / "data/attempts.jsonl")))
    if controls:
        detail += " controls=" + ",".join(f"{name[:15]}:{count}" for name, count in controls)
    summary = " ".join(f"{name}={states[name]}" for name in sorted(states))
    finished = states["complete"] == len(jobs) or bool(states["failed"] or states["budget_blocked"])
    return f"jobs={len(jobs)} {summary}{detail}", finished


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    runs_root = args.queue.parent.parent.parent
    while True:
        message, finished = summarize(args.queue, runs_root)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with args.log.open("a") as output:
            output.write(f"[progress {stamp}] {message}\n")
        if finished:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

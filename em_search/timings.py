"""Measured rental timings of finished studies (docs/EM_SEARCH_SPEC.md Phase 4), for the q8-search-v1 profile.

Usage: python -m em_search.timings STUDY [STUDY ...]

Per rental (instance): provisioning (lease created -> instance running), then per model the gap before training
(upload, package install, previous model's retrieval), training (the engine's train stage, including the base
model download on a rental's first model) and evaluation (protocol GPU work: merge and Betley sampling), then
teardown (last evaluation -> instance stopped). Judging runs after the rental, on the controller, and is reported
separately. Only rentals that trained a model count; preflights and failed rentals are listed but excluded from
the profile figures.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime

from generalization_lab.io import read, rows
from generalization_lab.protocols import store
from generalization_lab.wizard.study import Study


def _ts(value):
    return datetime.fromisoformat(value).timestamp() if value else None


def _minutes(start, end):
    return round((end - start) / 60, 2) if start is not None and end is not None else None


def rentals(study):
    """One record per instance that ran this study's models (leases of a rental group share the instance)."""
    runs_root = study.runs_root
    by_instance = {}
    for path in sorted(runs_root.glob("_vast/*/lease.json")):
        lease = read(path)
        if not lease.get("instance_id"):
            continue
        entry = by_instance.setdefault(lease["instance_id"], {"instance_id": lease["instance_id"], "leases": [],
                                                              "run_ids": []})
        entry["leases"].append(lease)
        entry["run_ids"].append(lease["run_id"])
        running = [s["timestamp"] for s in rows(path.parent / "startup.jsonl") if s.get("actual_status") == "running"]
        entry.setdefault("running", []).extend(running)
    result = []
    for entry in by_instance.values():
        created = min(_ts(lease["created_at"]) for lease in entry["leases"])
        stopped = max((_ts(lease.get("stopped_at")) or 0) for lease in entry["leases"]) or None
        running = min((_ts(t) for t in entry["running"]), default=None)
        models, previous_end = [], running
        for run_id in sorted(set(entry["run_ids"]), key=lambda r: _ts(_train(runs_root, r)[0]) or 0):
            train_start, train_end = _train(runs_root, run_id)
            gpu_start, gpu_end, judged = _protocols(runs_root, run_id)
            if train_start is None:
                continue
            models.append({"run_id": run_id, "before_training": _minutes(previous_end, _ts(train_start)),
                           "training": _minutes(_ts(train_start), _ts(train_end)),
                           "evaluation": _minutes(gpu_start, gpu_end),
                           "judging_after_rental": _minutes(stopped, judged)})
            previous_end = gpu_end
        result.append({"instance_id": entry["instance_id"], "purpose": entry["leases"][0].get("purpose"),
                       "status": [lease.get("status") for lease in entry["leases"]],
                       "hourly_usd": entry["leases"][0].get("hourly_total_usd"),
                       "gpu_usd": round(sum(lease.get("approximate_gpu_usd") or 0 for lease in entry["leases"]), 4),
                       "provisioning": _minutes(created, running), "models": models,
                       "teardown": _minutes(previous_end, stopped) if models else None,
                       "total": _minutes(created, stopped)})
    return result


def _train(runs_root, run_id):
    state = runs_root / run_id / "state.json"
    stage = read(state)["stages"].get("train", {}) if state.is_file() else {}
    return stage.get("started_at"), stage.get("completed_at") if stage.get("status") == "complete" else None


def _protocols(runs_root, run_id):
    starts, ends, judged = [], [], []
    for request in store.requests(runs_root, run_id):
        base = store.run_store(runs_root, run_id) / request["slug"]
        if (base / "gpu_complete.json").is_file():
            gpu = read(base / "gpu_complete.json")
            starts.append(_ts(gpu["started_at"]))
            ends.append(_ts(gpu["completed_at"]))
        if (base / "judge_complete.json").is_file():
            judged.append(_ts(read(base / "judge_complete.json")["completed_at"]))
    return min(starts, default=None), max(ends, default=None), max(judged, default=None)


def profile_figures(records):
    """Setup per rental (provisioning + first gap + teardown) and minutes per model (training + evaluation)."""
    trained = [r for r in records if r["models"]]
    setups = [r["provisioning"] + r["models"][0]["before_training"] + r["teardown"] for r in trained]
    per_model = [m["training"] + m["evaluation"] for r in trained for m in r["models"]]
    return {"rentals": len(trained), "models": len(per_model),
            "setup_minutes": {"values": setups, "max": max(setups, default=None),
                              "median": statistics.median(setups) if setups else None},
            "minutes_per_model": {"values": per_model, "max": max(per_model, default=None),
                                  "median": statistics.median(per_model) if per_model else None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("studies", nargs="+")
    args = parser.parse_args()
    records = {name: rentals(Study(name)) for name in args.studies}
    print(json.dumps({"rentals": records,
                      "profile": profile_figures([r for rs in records.values() for r in rs])}, indent=1))


if __name__ == "__main__":
    main()

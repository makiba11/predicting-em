"""Preregistered forecasts of studies that have not produced results.

A preregistration run is an ordinary forecast run whose targets are pending study cards (approved
dataset, no results). Once every job has a parsed forecast, `preregister` writes:

  RUN/preregistration.json                       hashes of the frozen prompt and plan, every prompt and
                                                 response, the target cards and the method; the point
                                                 forecasts; git provenance. Immutable.
  RESULTS/forecasting/preregistered/STUDY/RUN.json   one receipt per study, which `em launch` requires.

It refuses targets that already have results. Studies already launched are allowed but recorded as
`launched_before_forecast`. Commit both files before launching. Scoring later joins the forecasts to
result cards built from the finished studies (`score --outcomes SET`); see scoring.py.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from ...io import digest, freeze, now, provenance, read, sha
from . import cards as card_store
from . import prompt, run
from .scoring import averaged_forecasts


def receipts_root(root=None):
    from ..study import results_root

    return Path(root) if root else results_root() / "forecasting/preregistered"


def receipts(study_name, root=None):
    folder = receipts_root(root) / study_name
    return [read(p) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def launch_gate(name, state, found, no_forecast=None):
    """A first launch needs a preregistered forecast; resuming and offline demonstrations never do.

    Returns True when the launch goes ahead without one (no_forecast gives the recorded reason)."""
    if state["phase"] != "preregistered" or state["settings"].get("offline") or found:
        return False
    if not no_forecast:
        raise ValueError(f"No preregistered forecast for {name}: run `em forecast preregister` on a config "
                         "targeting it, or pass --no-forecast REASON")
    return True


def _launched(name, studies):
    from ..study import Study

    study = Study(name) if studies is None else card_store._study_at(name, studies)
    phase = study.load().get("phase")
    return phase in ("launched", "complete") or (study.path / "launch").exists()


def preregister(config, root=None, cards_root=None, studies=None, results=None, receipts_dir=None,
                allow_incomplete=False):
    folder = run.run_dir(config, root)
    path = folder / "preregistration.json"
    if path.exists():
        return read(path), path  # immutable: the receipts and the launch gate point at it
    if not (folder / "frozen.json").is_file():
        raise ValueError(f"No forecasts for {config.id} yet: run `em forecast run` first")
    planned = run.plan(config, cards_root, root)
    freeze(folder / "frozen.json", planned["frozen"])  # refused if the prompt, label or cards changed
    if read(folder / "plan.json")["jobs"] != planned["jobs"]:
        raise ValueError(f"{folder / 'plan.json'} differs from this config; rerun `em forecast run` first")
    targets, target_hashes, _ = run._cards(config.target_sets, cards_root, config.exclude_cards)
    with_results = [c.id for c in targets if c.measurements]
    if with_results:
        raise ValueError(f"Preregistration targets must be pending study cards; these have results: {with_results}")
    not_studies = [c.id for c in targets if not c.id.startswith("study/")]
    if not_studies:
        raise ValueError(f"Preregistration targets must be study cards: {not_studies}")
    already = [c.name for c in targets if card_store.has_results(c.name, results)]
    if already:
        raise ValueError(f"These studies already have results, so their forecasts are not preregistered: {already}")

    responses = run.latest(folder)
    missing = [j["job_id"] for j in planned["jobs"]
               if responses.get(j["job_id"], {}).get("status") != "complete"]
    if missing and not allow_incomplete:
        raise ValueError(f"{len(missing)} of {len(planned['jobs'])} jobs have no parsed forecast; finish "
                         "`em forecast run` (or pass --allow-incomplete to record them as missing)")
    target_ids = {c.id for c in targets}
    if not target_ids <= {j["target"] for j in planned["jobs"]}:
        raise ValueError("Some targets have no planned jobs")

    # Point forecast per forecaster, k and target: mean over evidence draws of the sample-averaged forecast.
    cells = defaultdict(list)
    for (forecaster, base_rate, k, draw, target), (forecast, _) in averaged_forecasts(planned["jobs"], responses).items():
        cells[(target, forecaster, base_rate, k)].append({e: prompt.point(forecast[e]) for e in prompt.TARGET_EVALS})
    forecasts = [
        {"target": target, "forecaster": forecaster, "base_rate": base_rate, "k": k, "draws": len(values),
         **{e: statistics.mean(v[e] for v in values) for e in prompt.TARGET_EVALS}}
        for (target, forecaster, base_rate, k), values in sorted(cells.items(), key=lambda kv: str(kv[0]))
    ]
    method = run.method(config)
    frozen_method = run.check_method(config, root)
    files = ("frozen.json", "plan.json", "prompts.jsonl", "responses.jsonl")
    body = {
        "run_id": config.id,
        "method": method,
        "method_sha256": digest(method),
        "frozen_method": frozen_method and {"name": frozen_method["name"], "sha256": frozen_method["sha256"]},
        "evidence_sets": config.evidence_sets,
        "target_sets": config.target_sets,
        "card_set_sha256": target_hashes,
        "targets": [{"id": c.id, "study": c.name, "family": c.family, "visible_sha256": card_store.visible_sha256(c),
                     "dataset_sha256": c.source.get("dataset_sha256"),
                     "launched_before_forecast": _launched(c.name, studies)} for c in targets],
        "file_sha256": {name: sha(folder / name) for name in files},
        "jobs": len(planned["jobs"]),
        "missing_jobs": missing,
        "forecasts": forecasts,
    }
    record = {**body, "sha256": digest(body), "preregistered_at": now(), "provenance": _provenance()}
    freeze(path, record)
    for target in body["targets"]:
        freeze(receipts_root(receipts_dir) / target["study"] / f"{config.id}.json", {
            "study": target["study"], "run_id": config.id, "preregistration": str(path),
            "preregistration_sha256": record["sha256"], "visible_sha256": target["visible_sha256"],
            "dataset_sha256": target["dataset_sha256"], "method_sha256": body["method_sha256"],
            "frozen_method": body["frozen_method"], "preregistered_at": record["preregistered_at"],
            "launched_before_forecast": target["launched_before_forecast"],
        })
    return record, path


def _provenance():
    info = provenance()
    return {"git_commit": info["git_commit"], "git_dirty": bool(info["git_dirty_status"]), "timestamp": info["timestamp"]}

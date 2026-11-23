"""Running spend of an EM search generation (docs/EM_SEARCH_SPEC.md), to stay within an approved budget.

Usage: python -m em_search.spend GENERATION --since ISO_TIMESTAMP

Generation and QA: every entry of results/em_search/generation_costs.jsonl logged since ``--since`` (specs
dropped from the generation list still count: their spend happened). GPU: every rental lease of the generation's studies (s-<dataset id>); a lease still live
is charged at its hourly price up to now. Judging: the Haiku judge ledgers of those studies' runs, with pending
reservations counted at their reserved amount (an upper bound until they settle).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from generalization_lab.io import ROOT, read, rows
from generalization_lab.wizard.study import Study

from . import batch
from .generate import COST_LOG
from .importer import study_name
from .spec import load_generation


def _ts(value):
    return datetime.fromisoformat(value).timestamp()


def gpu_usd(study):
    total = 0.0
    for path in study.runs_root.glob("_vast/*/lease.json"):
        lease = read(path)
        if lease.get("stopped_at") or lease.get("status") in {"destroyed", "retained_for_batch"}:
            total += lease.get("approximate_gpu_usd") or 0.0
        elif lease.get("created_at"):
            total += (lease.get("hourly_total_usd") or 0.0) * (time.time() - _ts(lease["created_at"])) / 3600
    return total


def judge_usd(study):
    total = 0.0
    for path in study.runs_root.glob("_protocol_evals/*/*/logs/usage.jsonl"):
        latest = {r["id"]: r for r in rows(path)}
        total += sum(r["cost_usd"] for r in latest.values())
    return total


def generation(name, since, extra_studies=()):
    """Spend since ``since``: every dataset logged in that window (including specs later dropped from the
    generation list) and the studies of those datasets and of the generation's specs."""
    specs = {spec.id for spec in load_generation(name)}
    logged = [r for r in rows(COST_LOG) if _ts(r["at"]) >= _ts(since)]
    datasets = {r["dataset_id"] for r in logged} | {f"{spec_id}-r1" for spec_id in specs}
    studies = [Study(study_name(d)) for d in sorted(datasets)] + [Study(s) for s in extra_studies]
    studies = [s for s in studies if s.exists()]
    mutations = ROOT / "results/em_search/mutations" / name
    mutation_usd = sum(batch.cost(folder)["usd"] for folder in mutations.glob("*") if (folder / "logs").is_dir())
    result = {"generation_and_qa_usd": sum(r["total_usd"] for r in logged), "mutation_usd": mutation_usd,
              "gpu_usd": sum(gpu_usd(s) for s in studies), "judge_usd": sum(judge_usd(s) for s in studies),
              "studies": len(studies), "cost_log_entries": len(logged)}
    result["total_usd"] = (result["generation_and_qa_usd"] + result["mutation_usd"] + result["gpu_usd"]
                           + result["judge_usd"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("generation")
    parser.add_argument("--since", required=True)
    parser.add_argument("--extra-study", action="append", default=[],
                        help="A study outside the s-<dataset id> pattern (e.g. a recipe probe).")
    args = parser.parse_args()
    print(json.dumps(generation(args.generation, args.since, args.extra_study), indent=1))


if __name__ == "__main__":
    main()

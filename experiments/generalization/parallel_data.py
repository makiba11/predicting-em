"""Prepare and review every behavior dataset in parallel before a shared-GPU batch starts.

Run this to completion, then start `generalization_lab batch` on the same matrix: the queue
finds each run's data stages complete and goes straight to the GPU. Training seeds share
generated data through the generation cache, so later seeds only rerun the data QA review.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generalization_lab.config import expand_matrix
from generalization_lab.costs import enforce
from generalization_lab.engine import Runner
from generalization_lab.preregistration import preregister
from generalization_lab.secrets import load_project_credentials


def prepare(spec, root):
    runner = Runner(spec, root)
    print(f"[prefetch {runner.run_id}] started", flush=True)
    for attempt in range(60):
        try:
            runner.execute(until="quality_check_data")
            break
        except RuntimeError as error:
            # Jobs sharing a matched control briefly hold its exclusive, non-blocking run lock.
            if "Already running (locked)" not in str(error) or attempt == 59:
                raise
            time.sleep(10)
    print(f"[prefetch {runner.run_id}] data_ready", flush=True)
    return runner.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    load_project_credentials()
    specs, _ = expand_matrix(args.matrix)
    for spec in specs:
        enforce(spec)
        preregister(args.runs_root, spec)  # The batch later reuses these identical snapshots.
    # The first job of each training seed builds that seed's matched control (run locks are
    # exclusive), then every other job is prepared in parallel.
    first = {}
    for spec in specs:
        first.setdefault(spec.training.seed, spec)
    for spec in first.values():
        prepare(spec, args.runs_root)
    rest = [spec for spec in specs if spec not in first.values()]
    print(f"[prefetch] preparing {len(rest)} more jobs with {args.workers} workers", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(prepare, spec, args.runs_root): f"{spec.experiment.id}/seed{spec.training.seed}"
            for spec in rest
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                label = futures[future]
                failures.append(label)
                print(f"[prefetch {label}] failed: {type(error).__name__}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"Prefetch failed for {', '.join(failures)}")
    print("[prefetch] all behavior datasets ready", flush=True)  # chain scripts match this line


if __name__ == "__main__":
    main()

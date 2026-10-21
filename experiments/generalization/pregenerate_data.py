"""Generate every behavior dataset and the matched benign control at once, into the shared cache.

`data.generate` caches completed attempts under `<runs-root>/_generation_cache/`, keyed by the
dataset spec, generator, prompt and request bank. This fills that cache for all conditions in
parallel; the engine (`parallel_data.py` / `batch`) then reuses identical rows and only runs QA.
Scratch directories `<runs-root>/_pregen_<name>/` hold resumable attempts and the API ledger.
"""

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generalization_lab import data
from generalization_lab.config import Control
from generalization_lab.config import expand_matrix
from generalization_lab.costs import Ledger
from generalization_lab.secrets import load_project_credentials

LEDGER_LIMIT_USD = 25


def control_spec(spec):
    """The matched-control spec exactly as engine.Runner.prepare_control derives it."""
    control = Control() if isinstance(spec.dataset.control_condition, str) else spec.dataset.control_condition
    result = spec.model_copy(deep=True)
    result.dataset.behavior_description = control.behavior_description
    result.dataset.generation_instruction = control.generation_instruction
    result.dataset.constraints = control.constraints
    result.dataset.examples = []
    result.dataset.required_regex = []
    result.dataset.control_condition = None
    return result


def generate(name, spec, root, seed_attempts=None, delay=0):
    # OpenRouter refuses large bursts of new TLS connections; staggered starts avoid that.
    time.sleep(delay)
    directory = root / f"_pregen_{name}"
    (directory / "data").mkdir(parents=True, exist_ok=True)
    if seed_attempts and not (directory / "data/attempts.jsonl").exists():
        shutil.copyfile(seed_attempts, directory / "data/attempts.jsonl")
    print(f"[pregen {name}] started", flush=True)
    for attempt in range(30):
        try:
            # Refused connections are booked at their full reservation although nothing was sent, so
            # this ledger needs headroom; the actually billed cost is the sum of completed records.
            rows = data.generate(spec, directory, Ledger(directory / "logs", LEDGER_LIMIT_USD))
        except RuntimeError as error:
            # Connection refusals under load: progress is saved, so resume after a pause.
            if "transport failed" not in str(error) and "budget exhausted" not in str(error):
                raise
            print(f"[pregen {name}] transport failure, resuming ({attempt + 1})", flush=True)
            time.sleep(30 + 10 * attempt)
        else:
            print(f"[pregen {name}] complete: {len(rows)} rows", flush=True)
            return
    raise RuntimeError(f"{name}: gave up after repeated transport failures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--only", nargs="*", help="Dataset names to generate (default: all)")
    parser.add_argument("--stagger", type=float, default=15, help="Seconds between dataset starts")
    args = parser.parse_args()
    load_project_credentials()
    specs, _ = expand_matrix(args.matrix)
    first_seed = min(spec.training.seed for spec in specs)
    targets = {spec.experiment.id: spec for spec in specs if spec.training.seed == first_seed}
    targets["matched_control"] = control_spec(next(iter(targets.values())))
    if args.only:
        targets = {name: targets[name] for name in args.only}
    # Resume rows already produced inside an engine run directory, if any.
    existing = {
        name: next(iter(sorted(args.runs_root.glob(f"{name}-*/data/attempts.jsonl"))), None) for name in targets
    }
    failures = []
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {
            pool.submit(generate, name, spec, args.runs_root, existing[name], index * args.stagger): name
            for index, (name, spec) in enumerate(targets.items())
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(futures[future])
                print(f"[pregen {futures[future]}] failed: {type(error).__name__}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"Pre-generation failed for {', '.join(failures)}; rerun to resume")
    print("[pregen] all datasets cached", flush=True)


if __name__ == "__main__":
    main()

"""One wizard job (a treatment or a same-seed benign control), run by the parallel Vast queue.

Usage: python -m generalization_lab.wizard.worker STUDY RUN_ID

Within the study's spending cap: rent one interruptible RTX 5090, train (resuming from off-instance
checkpoints after an interruption) and run the requested protocol GPU work in the same rental, then
judge the saved responses locally. A job is done when ``launch/done/RUN_ID.json`` exists; the
orchestrator then finishes the engine's local stages for each control/treatment pair.
"""

from __future__ import annotations

import sys

from ..compute.vast import protocols_pending
from ..engine import load_runner
from ..io import now, read, write
from ..protocols import judges, store
from ..secrets import load_project_credentials
from . import budget
from .launch import backend_for
from .study import Study


def needs_rental(runner):
    state = read(runner.directory / "state.json")["stages"]
    return state.get("neutral_eval", {}).get("status") != "complete" or protocols_pending(runner)


def run(name, run_id):
    study = Study(name)
    state = study.load()
    done = study.path / "launch/done" / f"{run_id}.json"
    if done.is_file():
        return read(done)
    backend, _ = backend_for(study, state)
    runner = load_runner(study.runs_root, run_id)
    if needs_rental(runner):
        attempt = budget.reserve(study, run_id, budget.attempt_bound(runner.spec), "train+protocol GPU")
        try:
            backend.run(runner, until="neutral_eval")
        finally:
            budget.settle(study, attempt)
    status = judges.run_judges(study.runs_root, run_id, offline=state["settings"]["offline"])
    record = {"run_id": run_id, "completed_at": now(), "judges": status,
              "protocols": store.progress(study.runs_root, run_id)}
    write(done, record)
    return record


def main():
    load_project_credentials()
    name, run_id = sys.argv[1:3]
    print(run(name, run_id), flush=True)


if __name__ == "__main__":
    main()

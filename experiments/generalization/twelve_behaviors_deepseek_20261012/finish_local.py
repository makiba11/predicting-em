"""Finish saved treatment runs once their Vast GPU stages and adapters arrive."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from generalization_lab.config import load
from generalization_lab.engine import load_runner, run_identity
from generalization_lab.io import read, redact
from generalization_lab.preregistration import find

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted(Path(__file__).resolve().parent.glob("[0-9][0-9]_*.yaml"))


def ready(run_id):
    directory = ROOT / "runs" / run_id
    state = read(directory / "state.json")["stages"]
    manifest = read(directory / "manifest.json")
    # The GPU launcher performs one final verification after extraction, then
    # records worker_auth. Waiting for that marker avoids racing its run lock.
    return (
        state.get("neutral_eval", {}).get("status") == "complete"
        and any((directory / "checkpoints/adapter").glob("*"))
        and manifest.get("worker_auth", {}).get("hugging_face_token_forwarded") is True
    )


def finish(run_id):
    for attempt in range(3):
        try:
            path = load_runner(ROOT / "runs", run_id).execute()
            return run_id, read(path / "manifest.json")["status"], ""
        except Exception as error:
            if attempt == 2:
                return run_id, "failed", redact(f"{type(error).__name__}: {error}")
            time.sleep(30 * (attempt + 1))


def main():
    run_ids = [run_identity(load(path), find(ROOT / "runs", load(path))) for path in CONFIGS]
    done = set()
    active = {}
    deadline = time.monotonic() + 4 * 3600
    with ThreadPoolExecutor(max_workers=2) as pool:
        while len(done) < len(run_ids) and time.monotonic() < deadline:
            for future, run_id in list(active.items()):
                if future.done():
                    result = future.result()
                    print("LOCAL_RESULT", *result, flush=True)
                    done.add(run_id)
                    del active[future]
            for run_id in run_ids:
                if len(active) >= 2:
                    break
                if run_id in done or run_id in active.values():
                    continue
                if read(ROOT / "runs" / run_id / "manifest.json")["status"] == "complete":
                    done.add(run_id)
                    continue
                if ready(run_id):
                    print("LOCAL_START", run_id, flush=True)
                    active[pool.submit(finish, run_id)] = run_id
            time.sleep(10)
    print("LOCAL_FINISHED", len(done), "of", len(run_ids), flush=True)
    if len(done) != len(run_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

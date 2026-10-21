"""Train a frozen treatment while its seed's matched control finishes inference.

The ordinary backend runs only through the completed train stage and exports its
private adapter. The matrix coordinator later resumes evaluation on that same
checkpoint after the control is complete.
"""

import argparse
from pathlib import Path

import vast_worker  # noqa: F401 - installs the same coordinated Vast offer claims

from generalization_lab.compute.vast import VastBackend
from generalization_lab.engine import load_runner
from generalization_lab.io import read
from generalization_lab.secrets import load_project_credentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    load_project_credentials()
    runner = load_runner(args.runs_root, args.run_id)
    control = runner.manifest.get("control_run_id")
    if not control:
        raise ValueError("A matched control is required")
    stages = read(args.runs_root / control / "state.json")["stages"]
    if stages.get("train", {}).get("status") != "complete":
        raise ValueError("Matched control training is not complete")
    if read(args.runs_root / args.run_id / "state.json")["stages"].get("train", {}).get("status") == "complete":
        print(f"{args.run_id}: training already complete", flush=True)
        return
    VastBackend().run(runner, until="train")
    print(f"{args.run_id}: training complete and adapter saved", flush=True)


if __name__ == "__main__":
    main()

"""Noninteractive entry point operating only on the already resolved, frozen run."""

import argparse
from pathlib import Path

from .engine import load_runner
from .io import provenance, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--until", default="quality_check_eval",
        help="An engine stage, 'protocols' (evaluation only) or 'preflight' (wizard launch check)"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.until == "preflight":
        # Wizard launch preflight: CUDA check and one real training step; writes preflight/result.json.
        from .wizard.preflight_remote import run

        run(args.root / args.run_id)
        return
    manifest = read(args.root / args.run_id / "manifest.json")
    folders = [args.root / args.run_id]
    if args.until != "protocols" and manifest.get("control_execution") != "separate":
        folders.append(args.root / manifest.get("control_run_id", ""))
    for folder in folders:
        if folder.name and (folder / "hub.json").is_file() and not (folder / "checkpoints/adapter").is_dir():
            from .hub import fetch_adapter

            fetch_adapter(folder)  # Pinned revision, hash-verified; replaces uploading it from the controller.
    if args.until != "protocols":
        runner = load_runner(args.root, args.run_id, args.force)
        runner.manifest["remote_provenance"] = provenance()
        runner.persist()
        runner.execute(args.until)
    # Requested protocol evaluations (if any) run on the trained checkpoint after the engine stages.
    # Their outputs go to the run's separate protocol store; the run folder is only read.
    from .protocols.gpu import run_pending

    run_pending(args.root, args.run_id)
    remove_temporary_adapter(args.root / args.run_id)


def remove_temporary_adapter(folder):
    """Results-only runs without a Hub copy keep the final adapter only while it is evaluated."""
    import shutil

    config = read(folder / "config/resolved.json")["compute"]
    if config.get("export_mode") != "results_only" or config.get("upload_private_hub"):
        return
    from .protocols.store import gpu_pending

    if not gpu_pending(folder.parent, folder.name):
        for name in ("adapter", "tokenizer"):
            shutil.rmtree(folder / "checkpoints" / name, ignore_errors=True)


if __name__ == "__main__":
    main()

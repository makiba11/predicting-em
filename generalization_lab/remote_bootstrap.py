"""Standard-library-only remote supervisor. Never accepts secrets on the command line."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hub_backed(directory):
    """Standard-library twin of io.hub_backed (this supervisor imports nothing from the package)."""
    try:
        receipt = json.loads((directory / "hub.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(
        receipt.get("private")
        and receipt.get("revision")
        and receipt.get("checkpoint_hash")
        and receipt.get("checkpoint_hash") == manifest.get("checkpoint_hash")
    )


def result_files(run, export_mode):
    if export_mode not in {"results_only", "final_adapter"}:
        raise ValueError(f"Unknown export mode: {export_mode}")
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in {".lock", ".backend.lock"}:
            continue
        relative = path.relative_to(run)
        # The worker completes inference before export. Only the optional final
        # adapter is useful after teardown; optimizer snapshots and the saved
        # tokenizer are never part of the result archive.
        if relative.parts[0] == "checkpoints" and not (
            export_mode == "final_adapter" and len(relative.parts) > 1 and relative.parts[1] == "adapter"
        ):
            continue
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "work", "status", "package"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--until", default="quality_check_eval")
    parser.add_argument("--deadline", type=float, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evals", action="store_true", help="package: only this run's protocol evaluations")
    args = parser.parse_args()
    folder = ROOT / args.root / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    logdir = folder / "logs"
    logdir.mkdir(exist_ok=True)
    status_file = logdir / "remote-status.json"
    log = logdir / "remote.log"

    def status(data):
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(status_file)

    if args.action == "launch":
        env = {**os.environ, **json.load(sys.stdin)}
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "work",
            "--root",
            args.root,
            "--run-id",
            args.run_id,
            "--until",
            args.until,
            "--deadline",
            str(args.deadline),
        ]
        if args.force:
            command.append("--force")
        status({"status": "starting"})
        with log.open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(json.dumps({"pid": process.pid}))
    elif args.action == "work":
        status({"status": "running", "pid": os.getpid()})
        try:
            config = json.loads((folder / "config/resolved.json").read_text())
            os.environ["HF_HOME"] = config["compute"]["cache_dir"]
            if config["compute"]["pip_install"]:
                requirements_hash = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
                marker = ROOT / ".installed_requirements_sha256"
                if not marker.exists() or marker.read_text().strip() != requirements_hash:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", ".[train]"],
                        cwd=ROOT,
                        check=True,
                        timeout=max(1, args.deadline - time.time()),
                    )
                    marker.write_text(requirements_hash)
            command = [
                sys.executable,
                "-m",
                "generalization_lab.remote_runner",
                "--root",
                args.root,
                "--run-id",
                args.run_id,
                "--until",
                args.until,
            ]
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=ROOT, check=True, timeout=max(1, args.deadline - time.time()))
        except BaseException as error:
            status({"status": "failed", "error_type": type(error).__name__})
        else:
            status({"status": "complete"})
    elif args.action == "status":
        data = json.loads(status_file.read_text()) if status_file.exists() else {"status": "unknown"}
        data["tail"] = log.read_text(errors="replace")[-4000:] if log.exists() else ""
        print(json.dumps(data))
    elif args.action == "package" and args.evals:
        store = ROOT / args.root / "_protocol_evals" / args.run_id
        archive = ROOT / "retrieved-evals.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for path in sorted(store.rglob("*")):
                if path.is_file() and not path.is_symlink() and not path.name.endswith(".lock"):
                    tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        h = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        print(json.dumps({"path": str(archive), "sha256": h.hexdigest()}))
    elif args.action == "package":
        manifest = json.loads((folder / "manifest.json").read_text())
        config = json.loads((folder / "config/resolved.json").read_text())
        export_mode = config["compute"].get("export_mode", "results_only")
        folders = [folder]
        if manifest.get("control_run_id") and manifest.get("control_execution") != "separate":
            folders.append(ROOT / args.root / manifest["control_run_id"])
        archive = ROOT / "retrieved.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for run in folders:
                # Adapters already in their private Hub repo (receipt matches the checkpoint) stay there;
                # the controller verifies the receipt instead of downloading hundreds of MB per run.
                hosted = hub_backed(run)
                for path in result_files(run, export_mode):
                    if hosted and path.relative_to(run).parts[0] == "checkpoints":
                        continue
                    tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        h = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        print(json.dumps({"path": str(archive), "sha256": h.hexdigest()}))


if __name__ == "__main__":
    main()

"""Restore a trained, incomplete run's pinned adapter before resuming GPU evaluation.

The standard Hub restore path requires a fully evaluated run. This recovery helper
also accepts incomplete runs, but requires a completed training stage, matching
checkpoint identity, a private repository and the original hashes of every file.
It changes no training, evaluation or stage records.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from generalization_lab.io import digest, lock, read, sha
from generalization_lab.secrets import load_project_credentials


def restore(folder):
    manifest = read(folder / "manifest.json")
    state = read(folder / "state.json")
    if state["stages"].get("train", {}).get("status") != "complete":
        raise ValueError(f"Training is not complete: {folder}")
    receipt = read(folder / "hub.json")
    expected = read(folder / "training/statistics.json")["adapter_files"]
    if not (
        expected and receipt.get("private") and receipt.get("revision")
        and receipt.get("checkpoint_hash") == manifest.get("checkpoint_hash") == digest(expected)
    ):
        raise ValueError(f"Checkpoint receipt mismatch: {folder}")
    adapter = folder / "checkpoints/adapter"
    if all((adapter / name).is_file() and sha(adapter / name) == value for name, value in expected.items()):
        return
    token = os.environ["HF_TOKEN"]
    if not HfApi(token=token).model_info(receipt["repo_id"], revision=receipt["revision"]).private:
        raise ValueError(f"Repository is not private: {receipt['repo_id']}")
    adapter.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        for name, expected_hash in expected.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Invalid adapter path: {name}")
            destination = adapter / name
            if destination.is_file() and sha(destination) == expected_hash:
                continue
            downloaded = Path(hf_hub_download(
                repo_id=receipt["repo_id"], filename=name, revision=receipt["revision"],
                cache_dir=cache, token=token,
            ))
            if sha(downloaded) != expected_hash:
                raise ValueError(f"Downloaded adapter hash mismatch: {folder}/{name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, destination)
    print(f"Restored and verified {folder.name}: {manifest['checkpoint_hash']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    load_project_credentials()
    with lock(args.run / ".backend.lock"):
        folders = [args.run]
        control = read(args.run / "manifest.json").get("control_run_id")
        if control:
            folders.append(args.run.parent / control)
        for folder in folders:
            with lock(folder / ".lock"):
                restore(folder)


if __name__ == "__main__":
    main()

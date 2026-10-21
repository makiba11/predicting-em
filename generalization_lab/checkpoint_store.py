"""Resumable step checkpoints kept outside a rented (possibly interruptible) instance.

The remote worker receives ``GLAB_CHECKPOINT_STORE`` as ``hub:OWNER/REPO`` (the run's private Hub
repository, under ``resume/``) or ``dir:/absolute/path`` (offline simulation and tests). Only
complete checkpoints are copied: ``complete.json`` (written last by training) lists every file's
SHA-256, and restoration verifies each file before training may resume from it.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .io import append, now, read, sha

STEP = re.compile(r"^step-(\d{8})$")


def verify(folder):
    saved = read(Path(folder) / "complete.json")
    for name, expected in saved["files"].items():
        path = Path(folder) / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Resume checkpoint file missing or changed: {path}")
    return saved


class DirectoryStore:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, checkpoint):
        verify(checkpoint)
        temporary = self.path / (checkpoint.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.copytree(checkpoint, temporary)
        shutil.rmtree(self.path / checkpoint.name, ignore_errors=True)
        os.replace(temporary, self.path / checkpoint.name)
        for older in self.path.iterdir():
            if STEP.match(older.name) and older.name < checkpoint.name:
                shutil.rmtree(older)
        return {"location": str(self.path / checkpoint.name)}

    def latest(self):
        steps = sorted(
            p.name for p in self.path.glob("step-*") if STEP.match(p.name) and (p / "complete.json").is_file()
        )
        return steps[-1] if steps else None

    def restore(self, name, destination):
        shutil.copytree(self.path / name, destination)


class HubStore:
    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))

    def save(self, checkpoint):
        verify(checkpoint)
        self.api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        if not self.api.model_info(self.repo_id).private:
            raise ValueError(f"Refusing to store checkpoints in a public repository: {self.repo_id}")
        commit = self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(checkpoint),
            path_in_repo=f"resume/{checkpoint.name}",
            commit_message=f"Resume checkpoint {checkpoint.name}",
        )
        for name in self._steps():
            if name < checkpoint.name:
                self.api.delete_folder(
                    repo_id=self.repo_id, path_in_repo=f"resume/{name}", commit_message=f"Superseded {name}"
                )
        return {"repo_id": self.repo_id, "revision": getattr(commit, "oid", None)}

    def _steps(self):
        files = self.api.list_repo_files(self.repo_id, repo_type="model")
        return sorted(
            {f.split("/")[1] for f in files if f.startswith("resume/step-") and f.endswith("/complete.json")}
        )

    def latest(self):
        try:
            steps = self._steps()
        except Exception as error:  # noqa: BLE001 - a missing repo means nothing to resume
            if "404" in str(error) or "RepositoryNotFound" in type(error).__name__:
                return None
            raise
        return steps[-1] if steps else None

    def restore(self, name, destination):
        from huggingface_hub import hf_hub_download

        files = [f for f in self.api.list_repo_files(self.repo_id) if f.startswith(f"resume/{name}/")]
        for remote in files:
            local = Path(hf_hub_download(self.repo_id, remote, token=self.api.token))
            target = Path(destination) / remote.removeprefix(f"resume/{name}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)


def from_env():
    value = os.environ.get("GLAB_CHECKPOINT_STORE")
    if not value:
        return None
    kind, _, location = value.partition(":")
    if kind == "dir":
        return DirectoryStore(location)
    if kind == "hub":
        return HubStore(location)
    raise ValueError("GLAB_CHECKPOINT_STORE must be hub:OWNER/REPO or dir:/PATH")


def save(checkpoint, log_directory):
    """Copy one complete checkpoint off the instance. Failure is recorded and never stops training."""
    store = from_env()
    if store is None:
        return None
    try:
        receipt = store.save(Path(checkpoint))
    except Exception as error:  # noqa: BLE001 - an upload failure only costs resumability
        append(
            Path(log_directory) / "checkpoint_store.jsonl",
            {"timestamp": now(), "checkpoint": Path(checkpoint).name, "error": type(error).__name__,
             "message": str(error)[:300]},
        )
        return None
    append(
        Path(log_directory) / "checkpoint_store.jsonl",
        {"timestamp": now(), "checkpoint": Path(checkpoint).name, "saved": receipt},
    )
    return receipt


def restore(checkpoint_root, log_directory):
    """Restore the newest stored checkpoint when the instance has none newer; returns its name."""
    store = from_env()
    if store is None:
        return None
    name = store.latest()
    if name is None:
        return None
    checkpoint_root = Path(checkpoint_root)
    local = sorted(p.name for p in checkpoint_root.glob("step-*") if (p / "complete.json").is_file())
    if local and local[-1] >= name:
        return None
    destination = checkpoint_root / name
    partial = checkpoint_root / (name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    store.restore(name, partial)
    verify(partial)
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(partial, destination)
    append(Path(log_directory) / "checkpoint_store.jsonl", {"timestamp": now(), "restored": name})
    return name

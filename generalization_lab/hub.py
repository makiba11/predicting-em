"""Upload retrieved final PEFT adapters to private Hub model repositories."""

import os
from pathlib import Path

from huggingface_hub import HfApi

from .io import digest, now, read, sha, write


def preflight_private_upload():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for private Hub upload")
    identity = HfApi(token=token).whoami()
    role = identity.get("auth", {}).get("accessToken", {}).get("role")
    if role != "write":
        raise ValueError("HF_TOKEN must have write access to create private model repos and upload adapters")
    return identity["name"]


def _folders(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    yield directory
    # A separately executed control uploads and verifies its own adapter as its own job.
    if manifest.get("control_run_id") and manifest.get("control_execution") != "separate":
        yield directory.parent / manifest["control_run_id"]


def _checkpoint(directory):
    """The checkpoint hash, after checking it matches the recorded adapter file hashes."""
    manifest = read(directory / "manifest.json")
    hashes = read(directory / "training/statistics.json")["adapter_files"]
    if not hashes or manifest.get("checkpoint_hash") != digest(hashes):
        raise ValueError(f"Checkpoint identity mismatch: {directory}")
    return manifest["checkpoint_hash"], hashes


def _expected(directory):
    checkpoint_hash, hashes = _checkpoint(directory)
    adapter = directory / "checkpoints/adapter"
    if any(not (adapter / name).is_file() or sha(adapter / name) != value for name, value in hashes.items()):
        raise ValueError(f"Final adapter files missing or changed: {directory}")
    return checkpoint_hash, adapter


def verify_run_and_control(directory):
    """Each adapter is in a private repo whose receipt matches its checkpoint; local copies are optional."""
    for folder in _folders(directory):
        checkpoint_hash, _ = _checkpoint(folder)
        receipt = read(folder / "hub.json")
        if receipt.get("checkpoint_hash") != checkpoint_hash or not receipt.get("private"):
            raise ValueError(f"Private Hub upload receipt missing or mismatched: {folder}")


def upload_adapter(folder, api=None, username=None):
    """Upload one run's final adapter and write its receipt; idempotent per checkpoint."""
    folder = Path(folder)
    checkpoint_hash, _ = _checkpoint(folder)
    receipt_path = folder / "hub.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("checkpoint_hash") == checkpoint_hash and receipt.get("private"):
            return receipt  # Uploaded already (usually from the GPU worker); no local copy needed.
        raise ValueError(f"Hub receipt differs from final adapter: {folder}")
    if api is None:
        username = preflight_private_upload()
        api = HfApi(token=os.environ.get("HF_TOKEN"))
    checkpoint_hash, adapter = _expected(folder)
    repo_id = f"{username}/{folder.name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id).private:
        raise ValueError(f"Refusing to upload adapter to public repository: {repo_id}")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(adapter),
        commit_message=f"Final LoRA adapter {folder.name}",
    )
    receipt = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": getattr(commit, "oid", None),
        "checkpoint_hash": checkpoint_hash,
        "private": True,
        "uploaded_at": now(),
        "uploaded_from": "vast" if os.environ.get("GLAB_REMOTE_WORKER") else "controller",
    }
    write(receipt_path, receipt)
    return receipt


def hosted_adapter(directory):
    """The Hub receipt of a complete run whose final adapter is in its verified private repo, else None."""
    directory = Path(directory)
    if not (directory / "hub.json").is_file() or not (directory / "manifest.json").is_file():
        return None
    manifest, receipt = read(directory / "manifest.json"), read(directory / "hub.json")
    if (
        manifest.get("status") != "complete"
        or not receipt.get("private")
        or not receipt.get("revision")
        or receipt.get("checkpoint_hash") != manifest.get("checkpoint_hash")
    ):
        return None
    return receipt


def fetch_adapter(directory, token=None):
    """Restore checkpoints/adapter from the pinned Hub revision, verifying every recorded file hash."""
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    receipt = hosted_adapter(directory)
    if receipt is None:
        raise ValueError(f"No verified private Hub adapter for {directory}")
    expected = read(directory / "training/statistics.json")["adapter_files"]
    adapter = directory / "checkpoints/adapter"
    if all((adapter / name).is_file() and sha(adapter / name) == value for name, value in expected.items()):
        return False
    adapter.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        for name, value in expected.items():
            path = Path(hf_hub_download(
                repo_id=receipt["repo_id"], filename=name, revision=receipt["revision"],
                cache_dir=cache, token=token or os.environ.get("HF_TOKEN"),
            ))
            if sha(path) != value:
                raise ValueError(f"Hub adapter file hash mismatch: {receipt['repo_id']}/{name}")
            shutil.copyfile(path, adapter / name)
    return True


def upload_from_worker(folder):
    """Best-effort upload right after training; the controller retries from its copy."""
    try:
        upload_adapter(folder)
    except Exception as error:  # noqa: BLE001 - never lose a trained adapter to an upload failure
        write(Path(folder) / "hub_error.json", {"type": type(error).__name__, "message": str(error)[:500], "at": now()})
        return False
    return True


def upload_run_and_control(directory):
    username = preflight_private_upload()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for folder in _folders(directory):
        upload_adapter(folder, api, username)

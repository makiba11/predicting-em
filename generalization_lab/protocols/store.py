"""Per-run protocol evaluation folders: requests, outputs and completion markers."""

from __future__ import annotations

from pathlib import Path

from ..io import PROTOCOL_EVALS, digest, freeze, read, rows, sha
from . import definitions

GPU_WORK = {definitions.HISTORICAL: True, definitions.STANDARD: True, definitions.PAPER: False}


def run_store(root, run_id):
    return Path(root) / PROTOCOL_EVALS / run_id


def slug(definition):
    return f"{definition['id']}-{definition['sha256'][:12]}"


def folder(root, run_id, definition):
    return run_store(root, run_id) / slug(definition)


def has_gpu_work(definition):
    if definition["id"].startswith(definitions.CUSTOM_PREFIX):
        return True
    return GPU_WORK[definition["id"]]


def verified_checkpoint(root, run_id):
    """The run's final checkpoint hash after checking it against its saved adapter file hashes."""
    directory = Path(root) / run_id
    manifest = read(directory / "manifest.json")
    stats = read(directory / "training/statistics.json")
    files = stats.get("adapter_files") or {}
    if not files or digest(files) != manifest.get("checkpoint_hash"):
        raise ValueError(f"{run_id}: final checkpoint hash does not match its recorded adapter files")
    receipt = directory / "hub.json"
    hosted = receipt.is_file() and read(receipt).get("checkpoint_hash") == manifest["checkpoint_hash"]
    adapter = directory / "checkpoints/adapter"
    local = adapter.is_dir() and all(
        (adapter / name).is_file() and sha(adapter / name) == value for name, value in files.items()
    )
    if not hosted and not local:
        raise ValueError(f"{run_id}: no verified local adapter or matching private Hub receipt")
    return {
        "checkpoint_hash": manifest["checkpoint_hash"],
        "adapter_files": files,
        "location": "local" if local else "hub",
        "hub": read(receipt) if hosted else None,
        "manifest_sha256": sha(directory / "manifest.json"),
    }


def request(root, run_id, definition):
    """Freeze one protocol request for one run. Re-requesting an identical protocol is a no-op."""
    trained = (Path(root) / run_id / "training/statistics.json").is_file()
    checkpoint = verified_checkpoint(root, run_id) if trained else None
    path = folder(root, run_id, definition)
    payload = {
        "run_id": run_id,
        "protocol_id": definition["id"],
        "protocol_sha256": definition["sha256"],
        "slug": slug(definition),
        "definition": definition,
        "gpu_work": has_gpu_work(definition),
        "inputs": sorted(definition.get("inputs", {})),
        "depends_on": {k: f"{k}-{v[:12]}" for k, v in definition.get("depends_on", {}).items()},
    }
    if (path / "request.json").is_file():
        saved = read(path / "request.json")
        if {k: saved[k] for k in payload} != payload:
            raise ValueError(f"Protocol request changed for {run_id}: {path}")
        return saved
    freeze(path / "request.json", {**payload, "checkpoint_at_request": checkpoint})
    return read(path / "request.json")


def requests(root, run_id):
    return [read(p) for p in sorted(run_store(root, run_id).glob("*/request.json"))]


def gpu_pending(root, run_id):
    return any(
        r["gpu_work"] and not (run_store(root, run_id) / r["slug"] / "gpu_complete.json").is_file()
        for r in requests(root, run_id)
    )


def latest(path):
    return {r["id"]: r for r in rows(path)}


def progress(root, run_id):
    """Human-readable progress of every requested protocol for one run."""
    result = []
    for r in requests(root, run_id):
        base = run_store(root, run_id) / r["slug"]
        entry = {
            "protocol": r["slug"],
            "gpu": "n/a" if not r["gpu_work"] else ("complete" if (base / "gpu_complete.json").is_file() else "pending"),
            "judged": "complete" if (base / "judge_complete.json").is_file() else "pending",
        }
        for name in ("responses.jsonl", "scores.jsonl"):
            if (base / name).is_file():
                entry[name.split(".")[0]] = len(latest(base / name))
        result.append(entry)
    return result

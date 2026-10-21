"""Append-only prediction snapshots bound to scientific inputs before generation."""

from pathlib import Path

from .config import science
from .io import digest, lock, now, provenance, read, write


def directory(root, spec):
    return Path(root) / "_preregistrations" / spec.experiment.id


def verify(snapshot):
    payload = {k: v for k, v in snapshot.items() if k != "sha256"}
    if digest(payload) != snapshot["sha256"]:
        raise ValueError("Preregistration hash mismatch")
    return snapshot


def find(root, spec):
    from .engine import engine_hash

    matches = []
    for path in sorted(directory(root, spec).glob("v*.json")):
        snap = verify(read(path))
        if (
            snap["science_hash"] == digest(science(spec))
            and snap["prediction"] == spec.prediction.model_dump(mode="json")
            and snap.get("engine_hash") == engine_hash()
        ):
            matches.append(snap)
    if not matches:
        raise ValueError("No matching preregistration. Run preregister before generating any data.")
    return matches[-1]


def preregister(root, spec, amend=False):
    from .engine import engine_hash
    from .results import freeze_protocol

    if spec.prediction.status != "registered":
        raise ValueError(
            "Prediction is a draft. Set prediction.status: registered after writing your predictions."
        )
    freeze_protocol(spec, root)
    folder = directory(root, spec)
    with lock(folder / ".lock"):
        existing = [verify(read(p)) for p in sorted(folder.glob("v*.json"))]
        same = [
            p
            for p in existing
            if p["science_hash"] == digest(science(spec))
            and p["prediction"] == spec.prediction.model_dump(mode="json")
            and p.get("engine_hash") == engine_hash()
        ]
        if same:
            return same[-1]
        if existing and not amend and any(p["science_hash"] == digest(science(spec)) for p in existing):
            raise ValueError("A different prediction already exists; use --amend to create a new version")
        started = list(folder.glob("started-*.json"))
        payload = {
            "version": len(existing) + 1,
            "timestamp": now(),
            "git": provenance(),
            "engine_hash": engine_hash(),
            "science_hash": digest(science(spec)),
            "config": spec.model_dump(mode="json"),
            "prediction": spec.prediction.model_dump(mode="json"),
            "previous_hash": existing[-1]["sha256"] if existing else None,
            "prospective": not bool(started),
            "prior_started_runs": [p.stem[8:] for p in started],
        }
        snap = {**payload, "sha256": digest(payload)}
        write(folder / f"v{len(existing) + 1:04d}-{snap['sha256'][:12]}.json", snap)
        return snap


def mark_started(root, spec, run_id, prereg_hash):
    path = directory(root, spec) / f"started-{run_id}.json"
    if not path.exists():
        write(path, {"timestamp": now(), "run_id": run_id, "preregistration_hash": prereg_hash})

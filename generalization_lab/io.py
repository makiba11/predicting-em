"""Atomic artifacts, conservative credential hygiene, and process locks."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Per-run protocol evaluation outputs live beside, never inside, run folders.
PROTOCOL_EVALS = "_protocol_evals"
_KNOWN_SECRETS = set()
SECRET_FIELD = re.compile(
    r"(^|_)(api_key|password|secret|authorization|access_token|hf_token|jupyter_token)$", re.I
)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_FIELD.search(k) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for secret in _KNOWN_SECRETS:
            value = value.replace(secret, "[REDACTED]")
        for key, secret in os.environ.items():
            if SECRET_FIELD.search(key) and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    return value


def remember_secret(value):
    if value and len(value) >= 8:
        _KNOWN_SECRETS.add(value)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(redact(text))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write(path, value):
    write_text(path, json.dumps(redact(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != redact(value):
            raise ValueError(f"Immutable artifact differs: {path}")
    else:
        write(path, value)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(redact(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    # Split on "\n" only: str.splitlines also splits on U+0085/U+2028/U+2029, which JSON leaves
    # unescaped inside strings (ensure_ascii=False), so model text containing them broke records.
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except ValueError as error:
            raise ValueError(
                f"Damaged JSONL at {path}:{index + 1}; repair explicitly before resuming"
            ) from error
    return result


def jsonl(path, values):
    write_text(path, "".join(canonical(redact(v)) + "\n" for v in values))


def csv_write(path, values):
    if not values:
        write_text(path, "")
        return
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
    writer.writeheader()
    writer.writerows(
        {k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in values
    )
    write_text(path, stream.getvalue())


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Already running (locked): {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "timestamp": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_status": git("status", "--porcelain"),
        "python": platform.python_version(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }


def stable_seed(seed, identity):
    return int(digest([seed, identity])[:8], 16) % (2**31)


def hub_backed(directory):
    """True when the run's final adapter is in its private Hub repo at a receipt matching its checkpoint.

    Such adapters are not transferred back from GPU workers; the pinned Hub revision is their copy.
    """
    directory = Path(directory)
    try:
        receipt, manifest = read(directory / "hub.json"), read(directory / "manifest.json")
    except (OSError, ValueError):
        return False
    return bool(
        receipt.get("private")
        and receipt.get("revision")
        and receipt.get("checkpoint_hash")
        and receipt.get("checkpoint_hash") == manifest.get("checkpoint_hash")
    )

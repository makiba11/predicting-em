"""Durable study state. Drafts, approvals and preregistrations are immutable once written."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..io import ROOT, append, digest, freeze, lock, now, read, write

NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,60}$")
PHASES = ("drafting", "data_approved", "preregistered", "launched", "complete")


def studies_root():
    root = Path(os.environ.get("EM_STUDIES_ROOT", ROOT / "studies")).resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("EM_STUDIES_ROOT must be inside the repository (Vast paths are repository relative)")
    return root


def results_root():
    return Path(os.environ.get("EM_RESULTS_ROOT", ROOT / "results")).resolve()


class Study:
    def __init__(self, name):
        if not NAME.fullmatch(name):
            raise ValueError("Study names use letters, digits, '-' and '_' (at most 61 characters)")
        self.name = name
        self.path = studies_root() / name

    # Paths ---------------------------------------------------------------------------------------
    @property
    def runs_root(self):
        return self.path / "runs"

    @property
    def data_dir(self):
        return self.path / "data"

    def draft_dir(self, number):
        return self.path / "drafts" / f"{number:04d}"

    def relative(self, path):
        return str(Path(path).resolve().relative_to(ROOT))

    # State ---------------------------------------------------------------------------------------
    def exists(self):
        return (self.path / "study.json").is_file()

    def create(self, state):
        if self.exists():
            raise ValueError(f"Study {self.name} already exists: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        write(self.path / "study.json", {"name": self.name, "created_at": now(), "phase": "drafting", **state})
        self.event("created", {"settings": state.get("settings")})

    def load(self):
        if not self.exists():
            raise ValueError(f"No study named {self.name}; create it with `em new {self.name}`")
        return read(self.path / "study.json")

    def update(self, **changes):
        with lock(self.path / ".study.lock"):
            state = self.load()
            state.update(changes, updated_at=now())
            write(self.path / "study.json", state)
            return state

    def event(self, kind, details=None):
        append(self.path / "history.jsonl", {"at": now(), "event": kind, **(details or {})})

    # Drafts --------------------------------------------------------------------------------------
    def drafts(self):
        folder = self.path / "drafts"
        return sorted(int(p.name) for p in folder.glob("[0-9][0-9][0-9][0-9]") if (p / "draft.json").is_file()) \
            if folder.is_dir() else []

    def latest_draft(self):
        numbers = self.drafts()
        if not numbers:
            raise ValueError("This study has no draft yet")
        return numbers[-1], read(self.draft_dir(numbers[-1]) / "draft.json")

    def add_draft(self, draft, parent=None, feedback=None, source="llm"):
        number = (self.drafts() or [0])[-1] + 1
        payload = {"number": number, "created_at": now(), "parent": parent, "feedback": feedback,
                   "source": source, "draft": draft, "sha256": digest(draft)}
        freeze(self.draft_dir(number) / "draft.json", payload)
        self.event("draft_created", {"draft": number, "parent": parent, "feedback": feedback})
        return number, payload

    def require_phase(self, *allowed):
        state = self.load()
        if state["phase"] not in allowed:
            raise ValueError(f"Study {self.name} is {state['phase']}; this step needs {' or '.join(allowed)}")
        return state

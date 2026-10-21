"""SQLite metadata only; large artifacts stay on disk."""

import sqlite3
from pathlib import Path

from .io import canonical, csv_write, now, write


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, status TEXT NOT NULL,
                config_hash TEXT NOT NULL, preregistration_hash TEXT, control_key TEXT,
                path TEXT NOT NULL, updated TEXT NOT NULL, cost REAL DEFAULT 0, metadata TEXT NOT NULL)""")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def save(self, manifest, path):
        m = manifest
        with self.connect() as db:
            prior = db.execute("SELECT status FROM runs WHERE run_id=?", (m["run_id"],)).fetchone()
            if prior and prior["status"] == "complete" and m["status"] != "complete":
                raise ValueError("Cannot reopen a completed run")
            db.execute(
                """INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status, updated=excluded.updated, cost=excluded.cost, metadata=excluded.metadata""",
                (
                    m["run_id"],
                    m["experiment_id"],
                    m["status"],
                    m["config_hash"],
                    m.get("preregistration_hash"),
                    m.get("control_key"),
                    str(Path(path).resolve()),
                    now(),
                    m.get("costs", {}).get("known_total_usd", 0),
                    canonical(m),
                ),
            )

    def all(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY updated")]

    def get(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError(f"Unknown run: {run_id}")
            return dict(row)

    def export(self, path):
        values = self.all()
        (csv_write if Path(path).suffix == ".csv" else write)(path, values)

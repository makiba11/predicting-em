"""Study-wide GPU spending cap shared by concurrent workers.

Every rental attempt reserves its worst case (hourly cap x rental deadline + transfer) under a
file lock before renting, and is settled to its lease journals afterwards. An unsettled
reservation (a killed worker) stays charged at its full bound until ``reconcile`` settles it.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

from ..io import lock, now, read, write


class BudgetExceeded(RuntimeError):
    """Study spending cap would be exceeded (never retried automatically)."""


def attempt_bound(spec):
    c = spec.compute
    return c.max_price_per_hour * c.timeout_minutes / 60 + c.transfer_budget_usd


def _path(study):
    return study.path / "launch/spend.json"


def _load(study):
    path = _path(study)
    return read(path) if path.is_file() else {"cap_usd": None, "attempts": []}


def committed(data):
    return sum(a["settled_usd"] if a["status"] == "settled" else a["reserved_usd"] for a in data["attempts"])


def set_cap(study, cap):
    with lock(study.path / "launch/.spend.lock"):
        data = _load(study)
        if cap < committed(data) - 1e-9:
            raise BudgetExceeded(f"New cap ${cap:.2f} is below already committed ${committed(data):.2f}")
        data["cap_usd"] = cap
        write(_path(study), data)


def reserve(study, run_id, bound, purpose):
    with lock(study.path / "launch/.spend.lock"):
        data = _load(study)
        if data["cap_usd"] is None:
            raise BudgetExceeded("No study spending cap is set; run `em launch` to review and set it")
        total = committed(data)
        if total + bound > data["cap_usd"] + 1e-9:
            raise BudgetExceeded(
                f"Study spending cap: ${total:.2f} committed + ${bound:.2f} for this rental exceeds "
                f"${data['cap_usd']:.2f}; raise it with `em launch {study.name} --max-total-usd`"
            )
        attempt = {"id": uuid.uuid4().hex[:12], "run_id": run_id, "purpose": purpose, "reserved_usd": bound,
                   "reserved_at": now(), "reserved_ts": time.time(), "status": "reserved"}
        data["attempts"].append(attempt)
        write(_path(study), data)
        return attempt["id"]


def _lease_cost(runs_root, run_id, since):
    total, leases = 0.0, []
    for path in (runs_root / "_vast").glob(f"{run_id}-*/lease.json"):
        lease = read(path)
        created = datetime.fromisoformat(lease["created_at"]).timestamp()
        if lease.get("run_id") == run_id and created >= since - 1:
            total += lease.get("approximate_gpu_usd", 0)
            leases.append(lease)
    return total, leases


def settle(study, attempt_id, force=False):
    with lock(study.path / "launch/.spend.lock"):
        data = _load(study)
        for attempt in data["attempts"]:
            if attempt["id"] != attempt_id or attempt["status"] == "settled":
                continue
            cost, leases = _lease_cost(study.runs_root, attempt["run_id"], attempt["reserved_ts"])
            if any(lease.get("status") in {"rented", "creating", "cleanup_required"} for lease in leases) and not force:
                return attempt  # A live or unconfirmed rental keeps its full reservation.
            attempt.update(status="settled", settled_usd=cost, settled_at=now(), leases=len(leases))
            write(_path(study), data)
            return attempt
    return None


def reconcile(study, live_run_ids=()):
    """Settle reservations whose workers are gone and whose leases are closed."""
    for attempt in list(_load(study)["attempts"]):
        if attempt["status"] == "reserved" and attempt["run_id"] not in live_run_ids:
            settle(study, attempt["id"])
    return summary(study)


def summary(study):
    data = _load(study)
    return {
        "cap_usd": data["cap_usd"],
        "committed_usd": committed(data),
        "settled_usd": sum(a.get("settled_usd", 0) for a in data["attempts"] if a["status"] == "settled"),
        "open_reservations_usd": sum(a["reserved_usd"] for a in data["attempts"] if a["status"] == "reserved"),
        "attempts": len(data["attempts"]),
    }

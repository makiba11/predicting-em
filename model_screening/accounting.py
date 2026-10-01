"""Durable request reservations, returned usage, failures and explicit reconciliation."""

import fcntl
import math
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .common import ROOT, append, now, read, rows, study, write
from .pricing import cost, rates


class BudgetExceeded(RuntimeError):
    pass


def summary(directory):
    directory = Path(directory)
    latest = {}
    path = directory / "usage.jsonl"
    if path.exists():
        for event in rows(path):
            latest[event["id"]] = event
    totals, stages, tokens = defaultdict(float), defaultdict(float), defaultdict(int)
    for event in latest.values():
        totals[event["status"]] += event["usd"]
        stages[event["stage"]] += event["usd"]
        for key, count in event["tokens"].items():
            tokens[key] += count
    estimated = sum(totals.values())
    reconciled = (
        read(directory / "reconciliation.json")
        if (directory / "reconciliation.json").exists()
        else None
    )
    # Reconciliation is a complete replacement total for this command/run,
    # including storage and failed calls, rather than an additive invoice.
    return {
        "estimated_usd": estimated,
        "accounted_usd": reconciled["actual_total_usd"] if reconciled else estimated,
        "by_status_usd": dict(totals),
        "by_stage_usd": dict(stages),
        "tokens": dict(tokens),
        "unresolved_requests": sum(
            e["status"] in ("reserved", "unknown") for e in latest.values()
        ),
        "reconciled": reconciled is not None,
        "storage_included": bool(reconciled and reconciled["storage_included"]),
        "note": "Unresolved requests retain their full bound; without reconciliation storage is additional.",
    }


def command_dirs(config):
    roots = {
        study(config),
        *((ROOT / p).resolve() for p in config["execution"]["prior_study_dirs"]),
    }
    paths = set()
    for root in roots:
        paths.update(p.parent for p in (root / "runs").glob("*/*/run.json"))
    return sorted(paths)


def pilot_spend(config):
    return config["execution"]["external_pilot_spend_usd"] + sum(
        summary(p)["accounted_usd"] for p in command_dirs(config)
    )


@contextmanager
def budget_lock(config):
    # Acquire all referenced study locks in canonical order. Commands in a study
    # execute serially; separate studies must name each other to share a budget.
    roots = sorted(
        {
            study(config),
            *((ROOT / p).resolve() for p in config["execution"]["prior_study_dirs"]),
        }
    )
    streams = []
    try:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            stream = (root / ".budget.lock").open("a")
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Another paid command holds {root}/.budget.lock; run models sequentially"
                ) from error
        yield
    finally:
        for stream in streams:
            stream.close()


class Ledger:
    def __init__(self, config, directory, snapshot, slug):
        self.config, self.directory = config, Path(directory)
        self.prices = rates(snapshot, slug)
        self.lock = threading.Lock()
        self.outside_spend = pilot_spend(config) - summary(directory)["accounted_usd"]
        self.total = summary(directory)["accounted_usd"]
        self.amounts = {}

    def reserve(self, stage, tokens, metadata=None):
        with self.lock:
            amount = cost(tokens, self.prices)
            current = self.total
            settings = self.config["execution"]
            if current + amount > settings["max_run_usd"]:
                raise BudgetExceeded(
                    f"Run cap ${settings['max_run_usd']:.2f}: ${current:.4f} used + ${amount:.4f} requested"
                )
            available = settings["pilot_cap_usd"] - settings["reserve_usd"]
            if self.outside_spend + current + amount > available:
                raise BudgetExceeded(
                    f"Pilot spend would exceed ${available:.2f} after reserve"
                )
            event = {
                "id": uuid4().hex,
                "at": now(),
                "stage": stage,
                "status": "reserved",
                "tokens": tokens,
                "usd": amount,
                "metadata": metadata or {},
            }
            append(self.directory / "usage.jsonl", event)
            self.amounts[event["id"]] = amount
            self.total += amount
            return event

    def settle(self, reservation, tokens=None, status="actual", **details):
        with self.lock:
            counts = reservation["tokens"] if tokens is None else tokens
            event = {
                **reservation,
                "at": now(),
                "status": status,
                "tokens": counts,
                "usd": cost(counts, self.prices),
                **details,
            }
            append(self.directory / "usage.jsonl", event)
            self.total += event["usd"] - self.amounts[reservation["id"]]
            self.amounts[reservation["id"]] = event["usd"]

    def call(self, stage, tokens, operation, actual=None, metadata=None):
        reservation = self.reserve(stage, tokens, metadata)
        try:
            result = operation()
            # All provider calls here are synchronous wrappers around SDK futures.
            self.settle(reservation, actual(result) if actual else tokens)
            return result
        except BaseException as error:
            self.settle(reservation, status="unknown", error_type=type(error).__name__)
            raise


def sample_counts(result, prompt_tokens):
    cache = result.prompt_cache_hit_tokens
    cached = 0 if cache is None else cache
    if type(cached) is not int or not 0 <= cached <= prompt_tokens:
        raise ValueError("Invalid Tinker prompt cache usage")
    return {
        "prefill": prompt_tokens - cached,
        "cached_prefill": cached,
        "sample": sum(len(s.tokens) for s in result.sequences),
    }


def judge_counts(result):
    usage = result.usage
    if usage is None:
        raise ValueError(
            "Judge API returned no usage; retain the request bound and reconcile"
        )
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    if not 0 <= cached <= usage.prompt_tokens:
        raise ValueError("Invalid judge cached token usage")
    return {
        "judge_input": usage.prompt_tokens - cached,
        "judge_cached_input": cached,
        "judge_output": usage.completion_tokens,
    }


def reconcile(directory, total, evidence):
    if not math.isfinite(total) or total < 0 or not evidence.strip():
        raise ValueError("Supply a nonnegative total and an invoice/session reference")
    status = read(directory / "run.json")["status"]
    if status in ("running", "preparing"):
        raise ValueError("Reconcile only after the command has ended")
    record = {
        "at": now(),
        "actual_total_usd": total,
        "storage_included": True,
        "evidence": evidence,
        "scope": "Total USD attributable to this run/command: Tinker tokens, OpenRouter judge tokens, failures/retries, storage through stated invoice date. Reconcile again if retained storage grows.",
    }
    append(directory / "reconciliation_history.jsonl", record)
    write(directory / "reconciliation.json", record)
    write(directory / "cost.json", summary(directory))

"""Claude Haiku 5.5 through the Message Batches API for em_search (generation and QA; another generator can
stand in through the same interface, see em_search.openrouter).

``run`` submits every task not yet answered, waits for the batches, collects them and resubmits failed,
expired or missing requests in new batches (up to ``rounds`` submissions). Everything is kept in the folder:
``batches.jsonl`` (each batch when submitted and when collected, so a rerun collects in-flight batches instead
of resubmitting), ``outputs.jsonl`` (every raw result) and ``logs/usage.jsonl`` (tokens and cost per request
at the batch price).
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from generalization_lab.io import append, canonical, now, rows
from generalization_lab.protocols import definitions, judges

MODEL = definitions.HAIKU_JUDGE_MODEL
USD_PER_MILLION = {"input": 0.10, "output": 0.50}
BATCH_DISCOUNT = 0.5
CHUNK = 10_000
ROUNDS = 3
POLL_SECONDS = 60


def pricing(offline):
    return SimpleNamespace(kind="mock" if offline else "anthropic", free=offline, model=MODEL,
                           input_usd_per_million=USD_PER_MILLION["input"] * BATCH_DISCOUNT,
                           output_usd_per_million=USD_PER_MILLION["output"] * BATCH_DISCOUNT)


def client_for(offline, folder, reply=None):
    if offline:
        return judges.MockAnthropic(store=Path(folder) / "offline_batches", reply=reply)
    return judges.anthropic_client({"api_key_env": "ANTHROPIC_API_KEY"})


def _batches(folder):
    merged = {}
    for row in rows(Path(folder) / "batches.jsonl"):
        merged.setdefault(row["batch_id"], {}).update(row)
    return merged


def answered(folder):
    """{custom_id: raw message} of every succeeded request."""
    return {r["custom_id"]: r["message"] for r in rows(Path(folder) / "outputs.jsonl") if r["status"] == "complete"}


def _collect(client, batch, wanted, folder, ledger, price, model=MODEL):
    reservations = batch.get("reservations") or {}
    seen = set()
    counts = {}
    for result in client.messages.batches.results(batch["batch_id"]):
        result = judges._plain(result)
        cid, kind = result["custom_id"], result["result"]["type"]
        if cid not in wanted:
            continue
        seen.add(cid)
        counts[kind] = counts.get(kind, 0) + 1
        reservation = ledger._latest.get(reservations.get(cid))
        if kind == "succeeded":
            message = judges._plain(result["result"]["message"])
            if not str(message.get("model", "")).startswith(model):
                raise definitions.ProtocolError(f"Answered as {message.get('model')!r}, not {model!r}")
            usage = message.get("usage") or {}
            tokens = {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0)}
            record = {"custom_id": cid, "status": "complete", "batch_id": batch["batch_id"], "message": message,
                      "input_tokens": tokens["prompt_tokens"], "output_tokens": tokens["completion_tokens"]}
        else:
            tokens = {"prompt_tokens": 0, "completion_tokens": 0}  # errored/expired requests are not billed
            record = {"custom_id": cid, "status": "failed", "batch_id": batch["batch_id"], "error_type": kind,
                      "error": canonical(result["result"])[:500]}
        if reservation is not None:
            ledger.settle({**reservation, "custom_id": cid}, price, tokens, "complete" if kind == "succeeded" else kind)
        append(Path(folder) / "outputs.jsonl", {**record, "at": now()})
    for cid in batch["custom_ids"]:
        if cid not in seen and cid in wanted:
            counts["missing"] = counts.get("missing", 0) + 1
            append(Path(folder) / "outputs.jsonl", {"custom_id": cid, "status": "failed", "batch_id": batch["batch_id"],
                                                    "error_type": "missing", "at": now()})
    append(Path(folder) / "batches.jsonl", {"batch_id": batch["batch_id"], "status": "collected", "counts": counts,
                                            "collected_at": now()})


def _reserve_text(params):
    return (params.get("system") or "") + "".join(m["content"] for m in params["messages"])


def run(folder, tasks, client, limit_usd, offline, poll_seconds=POLL_SECONDS, rounds=ROUNDS):
    """Answer every task ({"custom_id", "params"}); returns ``answered(folder)`` restricted to the tasks."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Another generator (em_search.openrouter) carries its own model name and price; Haiku otherwise.
    model, price = getattr(client, "em_model", MODEL), getattr(client, "em_price", None) or pricing(offline)
    ledger = judges.CachedLedger(folder / "logs", limit_usd)
    ledger.scope = folder.name
    wanted = {t["custom_id"]: t for t in tasks}
    if len(wanted) != len(tasks):
        raise ValueError("custom_id collision")
    submitted = 0
    while True:
        done = answered(folder)
        open_batches = [b for b in _batches(folder).values() if b["status"] == "submitted"]
        in_flight = {cid for b in open_batches for cid in b["custom_ids"]}
        pending = [t for t in tasks if t["custom_id"] not in done and t["custom_id"] not in in_flight]
        if pending:
            if submitted >= rounds:
                raise RuntimeError(f"{folder}: {len(pending)} requests still failed after {submitted} batches; "
                                   "their raw results are kept, rerun to retry them")
            estimate = sum(((len(_reserve_text(t["params"]).encode()) + 256) * price.input_usd_per_million
                            + t["params"]["max_tokens"] * price.output_usd_per_million) / 1e6 for t in pending)
            if not offline and ledger.spent() + estimate > ledger.limit + 1e-9:
                raise RuntimeError(f"{folder}: submitting ${estimate:.2f} more would exceed the ${limit_usd:.2f} cap")
            for start in range(0, len(pending), CHUNK):
                chunk = pending[start:start + CHUNK]
                reservations = {t["custom_id"]: ledger.reserve(price, "em_search", _reserve_text(t["params"]),
                                                               t["params"]["max_tokens"])["id"] for t in chunk}
                batch = judges._plain(client.messages.batches.create(
                    requests=[{"custom_id": t["custom_id"], "params": t["params"]} for t in chunk]))
                append(folder / "batches.jsonl", {"batch_id": batch["id"], "status": "submitted",
                                                  "custom_ids": [t["custom_id"] for t in chunk],
                                                  "reservations": reservations, "submitted_at": now()})
            submitted += 1
            continue
        if not open_batches:
            return {cid: message for cid, message in done.items() if cid in wanted}
        waiting = False
        for batch in open_batches:
            if judges._plain(client.messages.batches.retrieve(batch["batch_id"]))["processing_status"] == "ended":
                _collect(client, batch, wanted, folder, ledger, price, model)
            else:
                waiting = True
        if waiting:
            time.sleep(poll_seconds)


def cost(folder):
    """Billed tokens and USD of every settled request in the folder (batch price)."""
    latest = {r["id"]: r for r in rows(Path(folder) / "logs/usage.jsonl")}
    settled = [r for r in latest.values() if r["status"] == "complete"]
    return {"requests": len(settled),
            "input_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in settled),
            "output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in settled),
            "usd": sum(r["cost_usd"] for r in settled)}


def text(message):
    """The reply text, or None when the reply did not end normally."""
    if message.get("stop_reason") != "end_turn":
        return None
    return next((b.get("text", "") for b in message.get("content") or [] if b.get("type") == "text"), "")

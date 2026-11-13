"""Forecast runs: a YAML config expands into a deterministic job grid; responses are cached by job.

Output folder results/forecasting/RUN_ID/:
  frozen.json      config, prompt template, label definition and card-set hashes (refused if changed)
  plan.json        every job: forecaster, k, base-rate flag, evidence draw, sample, target, evidence ids
  prompts.jsonl    each distinct prompt once, by hash
  responses.jsonl  one record per attempt; the latest complete record per job is used
  usage.jsonl      API reservations and settlements (budget cap)

A forecasting method (prompt, forecasters, card format, grid and seed; not the evidence or targets) can
be frozen under a name in results/forecasting/methods/NAME.json. A config that names `frozen_method`
is refused unless its method is identical.
"""

from __future__ import annotations

import json
import math
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import Provider
from ...costs import Ledger
from ...io import ROOT, append, digest, freeze, now, read, rows, stable_seed, write
from . import cards as card_store
from . import labels, prompt
from .prompt import DEFAULT_SETUP


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Forecaster(Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    provider: Provider
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: int = Field(default=8000, gt=0)


class ForecastConfig(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
    description: str = ""
    prompt: str = "prompt_v2"
    label: str = labels.DEFAULT_LABEL
    setup: str = DEFAULT_SETUP
    evidence_sets: list[str]
    target_sets: list[str]
    exclude_cards: list[str] = []
    exclude_from_evidence: str = Field(default="target", pattern="^(target|family)$")
    k: list[int] = [0, 4, 8, 16, 32]
    draws: int = Field(default=5, ge=1)
    samples: int = Field(default=3, ge=1)
    base_rate: list[bool] = [False, True]
    datapoints_per_card: int = Field(default=1, ge=0, le=card_store.DATAPOINTS_STORED)
    max_chars_per_message: int = Field(default=1500, gt=0)
    forecasters: list[Forecaster]
    budget_usd: float = Field(default=10.0, ge=0)
    max_workers: int = Field(default=4, ge=1, le=64)
    max_attempts: int = Field(default=3, ge=1, le=10)
    seed: int = 0
    frozen_method: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
    # Transport only (not a METHOD_FIELD): Anthropic prompts carry cache_control, and the jobs that share a
    # prompt (samples, repeated k = 0 draws) run one after another so the repeats read the cache.
    prompt_cache: bool = False

    @model_validator(mode="after")
    def check(self):
        if any(k < 0 for k in self.k):
            raise ValueError("k values must be non-negative")
        if len({f.name for f in self.forecasters}) != len(self.forecasters):
            raise ValueError("Forecaster names must be unique")
        if self.prompt not in prompt.PROMPTS:
            raise ValueError(f"Unknown prompt template {self.prompt}; known: {sorted(prompt.PROMPTS)}")
        return self


FROZEN_FIELDS = {"prompt", "label", "setup", "evidence_sets", "target_sets", "exclude_cards", "exclude_from_evidence",
                 "datapoints_per_card", "max_chars_per_message", "seed"}


# What a forecaster's output depends on, other than the evidence and targets. Prices, timeouts, retries,
# budgets and worker counts are left out: they never change a forecast.
METHOD_FIELDS = {"prompt", "label", "setup", "k", "draws", "samples", "base_rate", "datapoints_per_card",
                 "max_chars_per_message", "exclude_from_evidence", "seed"}


def method(config):
    payload = config.model_dump(mode="json", include=METHOD_FIELDS)
    payload["prompt_template_sha256"] = prompt.template(config.prompt)["sha256"]
    payload["forecasters"] = [
        {"name": f.name, "kind": f.provider.kind, "model": f.provider.model, "base_url": f.provider.base_url,
         "extra_body": f.provider.extra_body, "temperature": f.temperature, "max_tokens": f.max_tokens}
        for f in config.forecasters
    ]
    return payload


def methods_root(root=None):
    return Path(root) / "methods" if root else ROOT / "results/forecasting/methods"


def freeze_method(config, name, root=None):
    """Freeze this config's method under NAME; refused if NAME is already frozen with a different method."""
    path = methods_root(root) / f"{name}.json"
    body = method(config)
    if path.exists():
        existing = read(path)
        if existing["sha256"] != digest(body):
            raise ValueError(f"Method {name} is already frozen with different settings: {path}")
        return existing, path
    record = {"name": name, "sha256": digest(body), "method": body, "frozen_from_config": config.id, "frozen_at": now()}
    write(path, record)
    return record, path


def check_method(config, root=None):
    """The frozen method this config must match, or None if it names none."""
    if config.frozen_method is None:
        return None
    path = methods_root(root) / f"{config.frozen_method}.json"
    if not path.is_file():
        raise ValueError(f"No frozen method {config.frozen_method} ({path}); freeze it with `em forecast freeze-method`")
    record = read(path)
    if digest(record["method"]) != record["sha256"]:
        raise ValueError(f"Frozen method file was modified: {path}")
    if digest(method(config)) != record["sha256"]:
        mine = method(config)
        changed = sorted(k for k in set(mine) | set(record["method"]) if mine.get(k) != record["method"].get(k))
        raise ValueError(f"Config {config.id} differs from frozen method {config.frozen_method} in: {', '.join(changed)}")
    return record


def load_config(path):
    return ForecastConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def run_dir(config, root=None):
    return Path(root or ROOT / "results/forecasting") / config.id


# Plan --------------------------------------------------------------------------------------------

def _cards(sets, cards_root, exclude=()):
    """Cards of the named sets minus excluded ids (experiments left out of this run entirely)."""
    result, hashes = [], {}
    for name in sets:
        loaded, sha256 = card_store.load(name, cards_root)
        result += [c for c in loaded if c.id not in exclude]
        hashes[name] = sha256
    unknown = set(exclude) - {c.id for name in sets for c in card_store.load(name, cards_root)[0]}
    return result, hashes, unknown


def eligible(target, pool, exclude_from_evidence):
    if exclude_from_evidence == "family":
        return [c for c in pool if c.family != target.family]
    return [c for c in pool if c.id != target.id]


def evidence_for(target, pool, k, draw, config):
    """Nested draws: the first k of one seeded permutation of the eligible pool."""
    order = sorted(eligible(target, pool, config.exclude_from_evidence), key=lambda c: c.id)
    random.Random(stable_seed(config.seed, ["evidence", draw])).shuffle(order)
    return order[:k]


def plan(config, cards_root=None, root=None):
    check_method(config, root)
    label_def = labels.definition(config.label)
    template = prompt.template(config.prompt)
    evidence_pool, evidence_hashes, unknown_e = _cards(config.evidence_sets, cards_root, config.exclude_cards)
    targets, target_hashes, unknown_t = _cards(config.target_sets, cards_root, config.exclude_cards)
    if unknown_e & unknown_t:
        raise ValueError(f"exclude_cards names unknown cards: {sorted(unknown_e & unknown_t)}")
    no_results = [c.id for c in evidence_pool if not c.measurements]
    if no_results:
        raise ValueError(f"Evidence cards without results (pending targets cannot be evidence): {no_results}")
    jobs, prompts = [], {}
    for target in targets:
        available = len(eligible(target, evidence_pool, config.exclude_from_evidence))
        for k in config.k:
            if k > available:
                continue  # a level larger than the pool would repeat the whole pool under another k
            # With no evidence, or all of it, every draw is identical: keep one.
            draws = 1 if k == 0 or k == available else config.draws
            for draw in range(draws):
                evidence = evidence_for(target, evidence_pool, k, draw, config)
                for base_rate in config.base_rate:
                    pool = [c for c in evidence_pool if c.id != target.id]
                    counts = Counter(labels.label(c, label_def)["label"] for c in pool) if base_rate else None
                    if counts is not None:
                        counts.pop(None, None)
                    text = prompt.render_prompt(config.prompt, target, evidence, label_def, config.setup, counts,
                                                config.datapoints_per_card, config.max_chars_per_message)
                    prompt_sha = digest(text)
                    prompts[prompt_sha] = text
                    for forecaster in config.forecasters:
                        for sample in range(config.samples):
                            identity = [forecaster.name, forecaster.provider.model, forecaster.temperature,
                                        forecaster.max_tokens, prompt_sha, sample]
                            jobs.append({
                                "job_id": digest(identity)[:20],
                                "forecaster": forecaster.name,
                                "target": target.id,
                                "k": k,
                                "k_shown": len(evidence),
                                "draw": draw,
                                "base_rate": base_rate,
                                "sample": sample,
                                "evidence": [c.id for c in evidence],
                                "prompt_sha256": prompt_sha,
                            })
    # Everything that changes what a prompt says or how it is labelled is frozen; the grid (k, draws,
    # samples, forecasters, base-rate flags) may grow later because jobs are keyed by content.
    frozen = {
        "science": config.model_dump(mode="json", include=FROZEN_FIELDS),
        "prompt_template": template,
        "label_definition": label_def,
        "card_sets": {**evidence_hashes, **target_hashes},
    }
    return {"frozen": frozen, "jobs": jobs, "prompts": prompts}


def write_plan(config, root=None, cards_root=None):
    planned = plan(config, cards_root, root)
    folder = run_dir(config, root)
    freeze(folder / "frozen.json", planned["frozen"])
    write(folder / "plan.json", {"created_at": now(), "config": config.model_dump(mode="json"),
                                 "jobs": planned["jobs"]})
    seen = {r["prompt_sha256"] for r in rows(folder / "prompts.jsonl")}
    for sha256, text in planned["prompts"].items():
        if sha256 not in seen:
            append(folder / "prompts.jsonl", {"prompt_sha256": sha256, "prompt": text})
    return planned


def estimate_cost(planned, config):
    by_name = {f.name: f for f in config.forecasters}
    total = 0.0
    for job in planned["jobs"]:
        f = by_name[job["forecaster"]]
        if f.provider.kind == "mock" or f.provider.free:
            continue
        prompt_tokens = len(planned["prompts"][job["prompt_sha256"]]) / 3.5
        total += (prompt_tokens * f.provider.input_usd_per_million + f.max_tokens * f.provider.output_usd_per_million) / 1e6
    return total


# Parsing -----------------------------------------------------------------------------------------

def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_forecast(text, kind="quantiles"):
    """The last JSON object in the reply holding both evals.

    kind "quantiles": seven finite, non-decreasing quantiles per eval. kind "expected": one finite
    number per eval, stored as {"expected": value}.
    """
    decoder = json.JSONDecoder()
    error = "no JSON object with original_mc and adjacent_mc"
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if not isinstance(value, dict) or not all(e in value for e in prompt.TARGET_EVALS):
            continue
        if kind == "expected":
            if all(_finite(value[e]) for e in prompt.TARGET_EVALS):
                return {e: {"expected": float(value[e])} for e in prompt.TARGET_EVALS}, None
            error = "original_mc and adjacent_mc must be finite numbers"
            continue
        result = {}
        for eval_name in prompt.TARGET_EVALS:
            entry = value[eval_name]
            if not isinstance(entry, dict) or set(entry) != set(prompt.QUANTILES):
                error = f"{eval_name} must have exactly the keys {', '.join(prompt.QUANTILES)}"
                break
            numbers = [entry[q] for q in prompt.QUANTILES]
            if not all(_finite(v) for v in numbers):
                error = f"{eval_name} quantiles must be finite numbers"
                break
            if any(b < a for a, b in pairwise(numbers)):
                error = f"{eval_name} quantiles cross"
                break
            result[eval_name] = {q: float(entry[q]) for q in prompt.QUANTILES}
        else:
            return result, None
    return None, error


# Forecasters -------------------------------------------------------------------------------------

class MockForecaster:
    """Deterministic offline fixture: never research evidence."""

    def __init__(self, forecaster, kind="quantiles"):
        self.forecaster = forecaster
        self.kind = kind

    def generate(self, messages, seed, temperature=None, max_tokens=None):
        h = int(digest([messages, seed])[:8], 16) / 0xFFFFFFFF
        center = round(-1 + 12 * h, 2)
        q = lambda spread: {"q05": center - 3 * spread, "q10": center - 2 * spread, "q15": center - spread,
                            "q50": center, "q85": center + spread, "q90": center + 2 * spread,
                            "q95": center + 3 * spread}
        if self.kind == "expected":
            text = "Offline fixture.\n" + json.dumps({"original_mc": center, "adjacent_mc": round(center / 5, 2)})
        else:
            text = "Offline fixture.\n" + json.dumps({"original_mc": q(1.5), "adjacent_mc": q(0.3)})
        return {"text": text, "finish_reason": "stop", "raw": {"mock": True}, "model": "mock"}

    def close(self):
        pass


def preflight(config):
    """Credentials and token prices for every live forecaster, checked before any call or attempt."""
    from ...providers import secret

    for f in config.forecasters:
        if f.provider.kind == "mock":
            continue
        if f.provider.kind not in {"openai", "vllm"}:
            raise ValueError(f"{f.name}: forecasters use an OpenAI-compatible endpoint (for example OpenRouter) or mock")
        secret(f.provider)
        if not f.provider.free and min(f.provider.input_usd_per_million, f.provider.output_usd_per_million) <= 0:
            raise ValueError(f"{f.name}: set current input/output token prices for {f.provider.model} before spending")


def make_forecaster(forecaster, ledger, kind="quantiles"):
    if forecaster.provider.kind == "mock":
        return MockForecaster(forecaster, kind)
    from ...providers import OpenAIProvider

    return OpenAIProvider(forecaster.provider, ledger, f"forecaster:{forecaster.name}")


# Run ---------------------------------------------------------------------------------------------

def latest(folder):
    """Latest record per job id."""
    return {r["job_id"]: r for r in rows(Path(folder) / "responses.jsonl")}


def pending(folder, jobs, max_attempts):
    attempts = Counter(r["job_id"] for r in rows(Path(folder) / "responses.jsonl"))
    done = {j for j, r in latest(folder).items() if r["status"] == "complete"}
    return [j for j in jobs if j["job_id"] not in done and attempts[j["job_id"]] < max_attempts]


def execute(config, root=None, cards_root=None, max_jobs=None, progress=None):
    preflight(config)
    planned = write_plan(config, root, cards_root)
    return execute_jobs(config, run_dir(config, root), planned, max_jobs, progress)


CACHED_PROVIDERS = ("anthropic/",)


def messages_for(config, forecaster, text):
    """The single user message; cache-marked content blocks when prompt caching applies (same text).

    prompt_v3 prompts are split after evidence cards 8, 16, 32 and 48 with a cache mark on each piece, so a prompt
    with more cards reads the cached prefix of the one with fewer (nested draws); the blocks concatenate to
    the same text."""
    if config.prompt_cache and forecaster.provider.model.startswith(CACHED_PROVIDERS):
        if config.prompt == "prompt_v3":
            cuts = [0, *prompt.cache_breakpoints(text, after_cards=(8, 16, 32, 48)), len(text)]
            blocks = [{"type": "text", "text": text[a:b]} for a, b in pairwise(cuts) if b > a]
            for block in blocks[:-1] if len(blocks) > 1 else blocks:
                block["cache_control"] = {"type": "ephemeral"}
            assert "".join(b["text"] for b in blocks) == text
            return [{"role": "user", "content": blocks}]
        return [{"role": "user", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}]
    return [{"role": "user", "content": text}]


def batches(config, todo):
    """Units of work for the thread pool: one job each, or with prompt caching every job of one forecaster
    and prompt (run in order, so the first writes the cache and the rest read it)."""
    if not config.prompt_cache:
        return [[job] for job in todo]
    groups = {}
    for job in todo:
        groups.setdefault((job["forecaster"], job["prompt_sha256"]), []).append(job)
    return list(groups.values())


def execute_jobs(config, folder, planned, max_jobs=None, progress=None):
    """Run the pending jobs of a written plan ({"jobs", "prompts"}) into folder."""
    folder = Path(folder)
    todo = pending(folder, planned["jobs"], config.max_attempts)[: max_jobs or None]
    ledger = Ledger(folder, config.budget_usd)
    by_name = {f.name: f for f in config.forecasters}
    kind = prompt.OUTPUT_KIND[config.prompt]
    clients = {name: make_forecaster(f, ledger, kind) for name, f in by_name.items()}
    attempts = Counter(r["job_id"] for r in rows(folder / "responses.jsonl"))
    write_lock, stop = threading.Lock(), threading.Event()
    tally = Counter()

    def one(job):
        if stop.is_set():
            return
        f = by_name[job["forecaster"]]
        attempt = attempts[job["job_id"]]
        seed = stable_seed(config.seed, [job["job_id"], attempt])
        record = {"job_id": job["job_id"], "attempt": attempt, "at": now(), "forecaster": f.name,
                  "model": f.provider.model or f.provider.kind, "seed": seed}
        try:
            reply = clients[f.name].generate(messages_for(config, f, planned["prompts"][job["prompt_sha256"]]),
                                             seed, temperature=f.temperature, max_tokens=f.max_tokens)
            forecast, error = parse_forecast(reply["text"], kind)
            raw = reply.get("raw") or {}
            message = ((raw.get("choices") or [{}])[0]).get("message") or {}
            record.update(text=reply["text"], reasoning=message.get("reasoning"), served_model=raw.get("model"),
                          finish_reason=reply.get("finish_reason"), usage=raw.get("usage"), forecast=forecast,
                          status="complete" if forecast else "unparsed", error=error)
        except RuntimeError as failure:
            if "budget exhausted" in str(failure):
                # Not an attempt: stop submitting and leave the job pending.
                stop.set()
                return
            record.update(status="failed", error=str(failure))
        except (ValueError, KeyError) as failure:
            record.update(status="failed", error=f"{type(failure).__name__}: {failure}")
        with write_lock:
            append(folder / "responses.jsonl", record)
            tally[record["status"]] += 1
            if progress:
                progress(sum(tally.values()), len(todo), record)

    def run_batch(batch):
        for job in batch:
            one(job)

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            for future in [pool.submit(run_batch, batch) for batch in batches(config, todo)]:
                future.result()
    finally:
        for client in clients.values():
            client.close()
    remaining = pending(folder, planned["jobs"], config.max_attempts)
    complete = sum(r["status"] == "complete" for r in latest(folder).values())
    return {"submitted": len(todo), "outcomes": dict(tally), "complete": complete, "jobs": len(planned["jobs"]),
            "retryable": len(remaining), "spent_usd": ledger.spent(), "budget_stopped": stop.is_set()}


BATCH_DONE = {"completed", "failed", "expired", "cancelled"}


def _batch_client(provider):
    import httpx

    from ...providers import secret

    return httpx.Client(base_url=provider.base_url.rstrip("/") + "/", timeout=300,
                        headers={"Authorization": f"Bearer {secret(provider)}"})


def _open_batches(folder):
    """Batches submitted (HTTP 202) and never collected: {batch_id: forecaster name}."""
    opened, closed = {}, set()
    for r in rows(Path(folder) / "batches.jsonl"):
        if (r.get("body") or {}).get("id") and r.get("status_code") in (200, 202):
            opened[r["body"]["id"]] = r["forecaster"]
        if r.get("status") in BATCH_DONE and r.get("id"):
            closed.add(r["id"])
    return {b: f for b, f in opened.items() if b not in closed}


def _collect(config, folder, planned, ledger, kind, batch_id, forecaster, data, progress, tally, total):
    """Record a finished batch's results as job attempts (the same records as the synchronous path)."""
    folder = Path(folder)
    sidecar = folder / "batches" / f"{batch_id}.json"
    meta = read(sidecar)["jobs"] if sidecar.is_file() else {}
    attempts = Counter(r["job_id"] for r in rows(folder / "responses.jsonl"))
    c = forecaster.provider
    results = {r["custom_id"]: r for r in data.get("results") or []}
    # Results carry token usage; the billed cost is the batch total. Split it by list-price cost.
    list_cost = {}
    for job_id, r in results.items():
        u = (((r.get("response") or {}).get("body")) or {}).get("usage") or {}
        list_cost[job_id] = u.get("prompt_tokens", 0) * c.input_usd_per_million + u.get("completion_tokens", 0) * c.output_usd_per_million
    billed = float((data.get("usage") or {}).get("cost") or 0)
    total_list = sum(list_cost.values()) or 1.0
    # Reservations of batches submitted before sidecars existed are released in order.
    legacy = [r for r in {x["id"]: x for x in rows(ledger.path)}.values()
              if r["status"] == "reserved" and r["role"] == f"forecaster:{forecaster.name}:batch"]
    for job_id in sorted(set(results) | set(meta)):
        if job_id in meta:
            attempt, seed, reservation_id = meta[job_id]
        else:
            attempt = attempts[job_id]
            seed = stable_seed(config.seed, [job_id, attempt])
            reservation_id = legacy.pop(0)["id"] if legacy else None
        record = {"job_id": job_id, "attempt": attempt, "at": now(), "forecaster": forecaster.name,
                  "model": c.model, "seed": seed, "transport": f"openrouter_batch:{batch_id}"}
        result = results.get(job_id) or {}
        raw = ((result.get("response") or {}).get("body")) or {}
        choices = raw.get("choices") or []
        cost = 0.0
        if len(choices) == 1:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            forecast, error = parse_forecast(text, kind)
            usage = raw.get("usage") or {}
            cost = float(usage["cost"]) if usage.get("cost") is not None else billed * list_cost.get(job_id, 0) / total_list
            usage = {**usage, "cost": cost, "cost_source": "result" if usage.get("cost") is not None
                     else "batch_total_split_by_list_price"}
            record.update(text=text, reasoning=message.get("reasoning"), served_model=raw.get("model"),
                          finish_reason=choices[0].get("finish_reason"), usage=usage, forecast=forecast,
                          status="complete" if forecast else "unparsed", error=error)
        else:
            record.update(status="failed", error=f"batch {data.get('status')}: {json.dumps(result.get('error'))[:300]}")
        if reservation_id is not None:
            base = {x["id"]: x for x in rows(ledger.path)}.get(reservation_id, {})
            append(ledger.path, {**base, "id": reservation_id, "status": "complete_batch" if cost else "failed_batch",
                                 "usage": record.get("usage"), "cost_usd": cost})
        else:
            append(ledger.path, {"id": f"batch:{batch_id}:{job_id}", "timestamp": now(), "role": f"forecaster:{forecaster.name}:batch",
                                 "model": c.model, "status": "complete_batch", "cost_usd": cost, "run_id": ledger.scope})
        append(folder / "responses.jsonl", record)
        tally[record["status"]] += 1
        if progress:
            progress(sum(tally.values()), total, record)


def _await(config, folder, planned, ledger, kind, open_batches, by_name, progress, tally, total, poll_seconds, deadline):
    import time

    import httpx

    clients = {}
    while open_batches and time.time() < deadline:
        for batch_id, name in list(open_batches.items()):
            f = by_name[name]
            client = clients.setdefault(name, _batch_client(f.provider))
            try:
                response = client.get(f"batches/{batch_id}")
                data = response.json()
            except (httpx.TransportError, ValueError):
                continue
            if response.status_code == 404 or (data.get("error") or {}).get("code") == 404:
                # Accepted (202) but rejected during validation (e.g. no credit for its holds): nothing ran.
                append(Path(folder) / "batches.jsonl", {"at": now(), "forecaster": name, "id": batch_id,
                                                        "status": "cancelled", "note": "HTTP 404: batch not found"})
                sidecar = Path(folder) / "batches" / f"{batch_id}.json"
                latest_ledger = {x["id"]: x for x in rows(ledger.path)}
                ids = [v[2] for v in read(sidecar)["jobs"].values()] if sidecar.is_file() else [
                    r["id"] for r in latest_ledger.values()
                    if r["status"] == "reserved" and r["role"] == f"forecaster:{name}:batch"]
                for reservation_id in ids:
                    if reservation_id in latest_ledger:
                        append(ledger.path, {**latest_ledger[reservation_id], "status": "batch_not_found", "cost_usd": 0.0})
                del open_batches[batch_id]
                continue
            if data.get("status") not in BATCH_DONE:
                continue
            _collect(config, folder, planned, ledger, kind, batch_id, f, data, progress, tally, total)
            append(Path(folder) / "batches.jsonl", {"at": now(), "forecaster": name, "id": batch_id, "status": data.get("status"),
                                                    "request_counts": data.get("request_counts"), "usage": data.get("usage")})
            del open_batches[batch_id]
        if open_batches:
            time.sleep(poll_seconds)
    for client in clients.values():
        client.close()
    return open_batches


def execute_jobs_batch(config, folder, planned, progress=None, poll_seconds=30, max_hours=26, chunk_size=25,
                       max_rounds=20, sync_forecasters=()):
    """execute_jobs over OpenRouter's Batch API (half price; transport only): one batch per forecaster with the
    request bodies, seeds and parsing of the synchronous path.

    Resumable: batches submitted earlier and never collected are awaited and recorded first, so no job is sent
    twice. Each submission's jobs (attempt, seed, ledger reservation) are kept in batches/BATCH_ID.json. A batch
    refused for credit (HTTP 402) leaves its jobs pending; any other refusal runs that forecaster synchronously.
    OpenRouter holds credit for each request's worst case (max_tokens) while a batch runs, so jobs go in
    chunks of chunk_size and in rounds: what the balance covers is submitted, and the rest follows once those
    batches finish and release their holds. The ledger records each result's share of the batch's billed cost."""

    import time

    folder = Path(folder)
    ledger = Ledger(folder, config.budget_usd)
    by_name = {f.name: f for f in config.forecasters}
    kind = prompt.OUTPUT_KIND[config.prompt]
    tally, log = Counter(), folder / "batches.jsonl"
    deadline = time.time() + max_hours * 3600
    adopted = _open_batches(folder)
    still_open = _await(config, folder, planned, ledger, kind, dict(adopted), by_name, progress, tally,
                        len(adopted), poll_seconds, deadline)
    if still_open:
        return {"adopted": len(adopted), "still_open": still_open, "outcomes": dict(tally), "submitted": 0,
                "retryable": len(pending(folder, planned["jobs"], config.max_attempts)), "spent_usd": ledger.spent()}

    # Forecasters sent synchronously (named, or not on OpenRouter) run in parallel with the batch rounds.
    direct = [f.name for f in config.forecasters
              if f.name in sync_forecasters or f.provider.kind != "openai" or "openrouter.ai" not in (f.provider.base_url or "")]
    direct_result, direct_thread = {}, None
    if direct:
        direct_config = config.model_copy(update={"forecasters": [by_name[n] for n in direct]})
        direct_plan = {**planned, "jobs": [j for j in planned["jobs"] if j["forecaster"] in direct]}
        direct_thread = threading.Thread(target=lambda: direct_result.update(
            execute_jobs(direct_config, folder, direct_plan, progress=progress)), daemon=True)
        direct_thread.start()
    refused, no_credit, still_open, submitted_total = list(direct), [], {}, 0
    committed = [0.0]  # bounds of batch requests submitted in this call and not yet booked
    for round_number in range(max_rounds):
        todo = [j for j in pending(folder, planned["jobs"], config.max_attempts) if j["forecaster"] not in refused]
        if not todo or time.time() > deadline:
            break
        attempts = Counter(r["job_id"] for r in rows(folder / "responses.jsonl"))
        groups, submitted, no_credit = {}, {}, []
        for job in todo:
            groups.setdefault(job["forecaster"], []).append(job)
        for name, jobs in groups.items():
            f = by_name[name]
            c = f.provider
            if name in direct:
                continue
            for start_index in range(0, len(jobs), chunk_size):
                requests, meta = [], {}
                for job in jobs[start_index:start_index + chunk_size]:
                    attempt = attempts[job["job_id"]]
                    seed = stable_seed(config.seed, [job["job_id"], attempt])
                    messages = messages_for(config, f, planned["prompts"][job["prompt_sha256"]])
                    body = {**c.extra_body, "model": c.model, "messages": messages, "seed": seed,
                            "temperature": f.temperature if f.temperature is not None else c.temperature,
                            "max_tokens": f.max_tokens or c.max_tokens}
                    # Budget check without a ledger reservation: the synchronous thread writes the same ledger,
                    # and its numeric ids are line counts. Batch results are booked under unique string ids.
                    bound = ((len(json.dumps(messages).encode()) + 256) * c.input_usd_per_million
                             + body["max_tokens"] * c.output_usd_per_million) / 1e6 * 0.5
                    if ledger.spent() + committed[0] + bound > config.budget_usd:
                        break  # budget: leave the rest pending
                    committed[0] += bound
                    requests.append({"custom_id": job["job_id"], "body": body})
                    meta[job["job_id"]] = [attempt, seed, None]
                if not requests:
                    break
                client = _batch_client(c)
                response = client.post("batches", json={"endpoint": "/v1/chat/completions", "model": c.model,
                                                        "requests": requests})
                client.close()
                append(log, {"at": now(), "forecaster": name, "status_code": response.status_code, "requests": len(requests),
                             "round": round_number, "body": response.json() if response.is_success else None})
                if not response.is_success:
                    committed[0] -= sum(((len(json.dumps(r["body"]["messages"]).encode()) + 256) * c.input_usd_per_million
                                         + r["body"]["max_tokens"] * c.output_usd_per_million) / 1e6 * 0.5 for r in requests)
                    (no_credit if response.status_code == 402 else refused).append(name)
                    break  # this forecaster waits for the next round (402) or runs synchronously
                batch_id = response.json()["id"]
                write(folder / "batches" / f"{batch_id}.json", {"forecaster": name, "jobs": meta})
                submitted[batch_id] = name
                submitted_total += len(requests)
        if not submitted:
            break  # nothing fits the balance (or nothing left): stop instead of spinning
        still_open = _await(config, folder, planned, ledger, kind, submitted, by_name, progress, tally,
                            submitted_total, poll_seconds, deadline)
        committed[0] = 0.0
        if still_open:
            break
    if direct_thread:
        direct_thread.join()
    sync = direct_result or None
    late = [n for n in refused if n not in direct]  # batch refused for another reason than credit
    if late:
        sync_config = config.model_copy(update={"forecasters": [by_name[n] for n in late]})
        sync_plan = {**planned, "jobs": [j for j in planned["jobs"] if j["forecaster"] in late]}
        sync = {"direct": direct_result, "late": execute_jobs(sync_config, folder, sync_plan, progress=progress)}
    remaining = pending(folder, planned["jobs"], config.max_attempts)
    complete = sum(r["status"] == "complete" for r in latest(folder).values())
    return {"adopted": len(adopted), "submitted": submitted_total, "outcomes": dict(tally), "batch_refused": refused,
            "refused_no_credit": no_credit, "still_open": still_open, "sync": sync, "complete": complete,
            "jobs": len(planned["jobs"]), "retryable": len(remaining), "spent_usd": ledger.spent(), "budget_stopped": False}


def frozen(folder):
    return read(Path(folder) / "frozen.json")

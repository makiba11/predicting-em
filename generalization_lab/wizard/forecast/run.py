"""Forecast runs: a YAML config expands into a deterministic job grid; responses are cached by job.

Output folder results/forecasting/RUN_ID/:
  frozen.json      config, prompt template, label definition and card-set hashes (refused if changed)
  plan.json        every job: forecaster, k, base-rate flag, evidence draw, sample, target, evidence ids
  prompts.jsonl    each distinct prompt once, by hash
  responses.jsonl  one record per attempt; the latest complete record per job is used
  usage.jsonl      API reservations and settlements (budget cap)
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


def plan(config, cards_root=None):
    label_def = labels.definition(config.label)
    template = prompt.template(config.prompt)
    evidence_pool, evidence_hashes, unknown_e = _cards(config.evidence_sets, cards_root, config.exclude_cards)
    targets, target_hashes, unknown_t = _cards(config.target_sets, cards_root, config.exclude_cards)
    if unknown_e & unknown_t:
        raise ValueError(f"exclude_cards names unknown cards: {sorted(unknown_e & unknown_t)}")
    jobs, prompts = [], {}
    for target in targets:
        for k in config.k:
            available = len(eligible(target, evidence_pool, config.exclude_from_evidence))
            # With no evidence, or all of it, every draw is identical: keep one.
            draws = 1 if k == 0 or k >= available else config.draws
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
    planned = plan(config, cards_root)
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
    folder = run_dir(config, root)
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
            reply = clients[f.name].generate([{"role": "user", "content": planned["prompts"][job["prompt_sha256"]]}],
                                             seed, temperature=f.temperature, max_tokens=f.max_tokens)
            forecast, error = parse_forecast(reply["text"], kind)
            raw = reply.get("raw") or {}
            message = ((raw.get("choices") or [{}])[0]).get("message") or {}
            record.update(text=reply["text"], reasoning=message.get("reasoning"),
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

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            for future in [pool.submit(one, job) for job in todo]:
                future.result()
    finally:
        for client in clients.values():
            client.close()
    remaining = pending(folder, planned["jobs"], config.max_attempts)
    complete = sum(r["status"] == "complete" for r in latest(folder).values())
    return {"submitted": len(todo), "outcomes": dict(tally), "complete": complete, "jobs": len(planned["jobs"]),
            "retryable": len(remaining), "spent_usd": ledger.spent(), "budget_stopped": stop.is_set()}


def frozen(folder):
    return read(Path(folder) / "frozen.json")

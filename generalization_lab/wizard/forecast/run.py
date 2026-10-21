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
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import Provider
from ...costs import Ledger
from ...io import ROOT, append, digest, freeze, now, read, rows, stable_seed, write
from . import cards as card_store
from . import labels, prompt


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
    prompt: str = "prompt_v1"
    label: str = "label_v1"
    setup: str = prompt.DEFAULT_SETUP
    evidence_sets: list[str]
    target_sets: list[str]
    exclude_from_evidence: str = Field(default="target", pattern="^(target|family)$")
    k: list[int] = [0, 2, 4, 8, 16]
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
        return self


def load_config(path):
    return ForecastConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def run_dir(config, root=None):
    return Path(root or ROOT / "results/forecasting") / config.id


# Plan --------------------------------------------------------------------------------------------

def _cards(sets, cards_root):
    result, hashes = [], {}
    for name in sets:
        loaded, sha256 = card_store.load(name, cards_root)
        result += loaded
        hashes[name] = sha256
    return result, hashes


def eligible(target, pool, config):
    if config.exclude_from_evidence == "family":
        return [c for c in pool if c.family != target.family]
    return [c for c in pool if c.id != target.id]


def evidence_for(target, pool, k, draw, config):
    """Nested draws: the first k of one seeded permutation of the eligible pool."""
    order = sorted(eligible(target, pool, config), key=lambda c: c.id)
    random.Random(stable_seed(config.seed, ["evidence", draw])).shuffle(order)
    return order[:k]


def plan(config, cards_root=None):
    label_def = labels.definition(config.label)
    template = prompt.template(config.prompt)
    evidence_pool, evidence_hashes = _cards(config.evidence_sets, cards_root)
    targets, target_hashes = _cards(config.target_sets, cards_root)
    jobs, prompts = [], {}
    for target in targets:
        for k in config.k:
            available = len(eligible(target, evidence_pool, config))
            # With no evidence, or all of it, every draw is identical: keep one.
            draws = 1 if k == 0 or k >= available else config.draws
            for draw in range(draws):
                evidence = evidence_for(target, evidence_pool, k, draw, config)
                for base_rate in config.base_rate:
                    pool = [c for c in evidence_pool if c.id != target.id]
                    counts = Counter(labels.label(c, label_def)["label"] for c in pool) if base_rate else None
                    if counts is not None:
                        counts.pop(None, None)
                    text = prompt.render(config.prompt, target, evidence, label_def, config.setup, counts,
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
    frozen = {
        "config": config.model_dump(mode="json"),
        "prompt_template": template,
        "label_definition": label_def,
        "card_sets": {**evidence_hashes, **target_hashes},
    }
    return {"frozen": frozen, "jobs": jobs, "prompts": prompts}


def write_plan(config, root=None, cards_root=None):
    planned = plan(config, cards_root)
    folder = run_dir(config, root)
    freeze(folder / "frozen.json", planned["frozen"])
    write(folder / "plan.json", {"created_at": now(), "jobs": planned["jobs"]})
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

def parse_forecast(text):
    """The last JSON object in the reply holding both evals with seven finite, non-decreasing quantiles."""
    decoder = json.JSONDecoder()
    error = "no JSON object with original_mc and adjacent_mc"
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if not isinstance(value, dict) or not all(e in value for e in prompt.TARGET_EVALS):
            continue
        result = {}
        for eval_name in prompt.TARGET_EVALS:
            entry = value[eval_name]
            if not isinstance(entry, dict) or set(entry) != set(prompt.QUANTILES):
                error = f"{eval_name} must have exactly the keys {', '.join(prompt.QUANTILES)}"
                break
            numbers = [entry[q] for q in prompt.QUANTILES]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
                error = f"{eval_name} quantiles must be finite numbers"
                break
            if any(b < a for a, b in zip(numbers, numbers[1:])):
                error = f"{eval_name} quantiles cross"
                break
            result[eval_name] = {q: float(entry[q]) for q in prompt.QUANTILES}
        else:
            return result, None
    return None, error


# Forecasters -------------------------------------------------------------------------------------

class MockForecaster:
    """Deterministic offline fixture: never research evidence."""

    def __init__(self, forecaster):
        self.forecaster = forecaster

    def generate(self, messages, seed, temperature=None, max_tokens=None):
        h = int(digest([messages, seed])[:8], 16) / 0xFFFFFFFF
        center = round(-1 + 12 * h, 2)
        q = lambda spread: {"q05": center - 3 * spread, "q10": center - 2 * spread, "q15": center - spread,
                            "q50": center, "q85": center + spread, "q90": center + 2 * spread,
                            "q95": center + 3 * spread}
        text = "Offline fixture.\n" + json.dumps({"original_mc": q(1.5), "adjacent_mc": q(0.3)})
        return {"text": text, "finish_reason": "stop", "raw": {"mock": True}, "model": "mock"}

    def close(self):
        pass


def make_forecaster(forecaster, ledger):
    if forecaster.provider.kind == "mock":
        return MockForecaster(forecaster)
    if forecaster.provider.kind not in {"openai", "vllm"}:
        raise ValueError("Forecasters use an OpenAI-compatible endpoint (for example OpenRouter) or mock")
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
    planned = write_plan(config, root, cards_root)
    folder = run_dir(config, root)
    todo = pending(folder, planned["jobs"], config.max_attempts)[: max_jobs or None]
    ledger = Ledger(folder, config.budget_usd)
    by_name = {f.name: f for f in config.forecasters}
    clients = {name: make_forecaster(f, ledger) for name, f in by_name.items()}
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
            forecast, error = parse_forecast(reply["text"])
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

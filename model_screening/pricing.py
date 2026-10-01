"""Public price snapshots; never silently reprice an existing run."""

import hashlib
import json
import math
import urllib.request
from datetime import datetime, timezone

from .common import HERE, now, positive, read, write

TINKER_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
JUDGE_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
PRICE_FILE = HERE / "pricing.json"
TOKEN_FIELDS = ("prefill", "cached_prefill", "sample", "train")


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "em-model-screening/1.0"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def parse_judge(document, provider):
    # Price only the pinned endpoint; other providers of the model differ.
    endpoints = [
        e for e in json.loads(document)["data"]["endpoints"] if e["tag"] == provider
    ]
    if len(endpoints) != 1:
        raise ValueError(f"Missing or ambiguous OpenRouter endpoint: {provider}")
    price = endpoints[0]["pricing"]
    if price.get("discount") or price.get("overrides"):
        raise ValueError(
            "Judge endpoint has discounted or time-of-day pricing; verify manually"
        )
    return {
        field: positive(round(float(price[key]) * 1_000_000, 9), field)
        for key, field in [
            ("prompt", "judge_input"),
            ("input_cache_read", "judge_cached_input"),
            ("completion", "judge_output"),
        ]
    }


def refresh(config):
    raw = fetch(TINKER_URL)
    catalog = json.loads(raw)
    judge_url = JUDGE_URL.format(model=config["betley"]["judge_model"])
    judge_raw = fetch(judge_url)
    timestamp = now()
    models = {}
    for slug, spec in config["models"].items():
        matches = [r for r in catalog if r["tinker_id"] == spec["model"]]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous pricing for {spec['model']}")
        row = matches[0]
        models[slug] = {
            "model": spec["model"],
            "usd_per_million_tokens": {
                k: positive(float(row[k].removeprefix("$")), k) for k in TOKEN_FIELDS
            },
            "source_row": row,
        }
    snapshot = {
        "currency": "USD",
        "retrieved_at": timestamp,
        "tinker_source": TINKER_URL,
        "tinker_source_sha256": hashlib.sha256(raw).hexdigest(),
        "models": models,
        "judge": {
            "model": config["betley"]["judge_model"],
            "provider": config["betley"]["judge_provider"],
            "source": judge_url,
            "retrieved_at": timestamp,
            "source_sha256": hashlib.sha256(judge_raw).hexdigest(),
            "usd_per_million_tokens": parse_judge(
                judge_raw.decode(), config["betley"]["judge_provider"]
            ),
        },
        "storage": {
            "usd_per_gb_month": 0.1,
            "source": "https://tinker-docs.thinkingmachines.ai/tinker/models/",
            "verified_at": "2026-09-24T00:00:00+00:00",
            "note": "Reference rate; NOT refreshed by models.json. Storage is not included in token estimates or hard request bounds. Reconcile actual storage separately; do not treat a missing storage invoice as zero spend.",
        },
        "note": "Standard synchronous text rates, selected context tier, no Batch discount. Discounted and original Tinker rates retained in source_row. Estimates, not an invoice. Each run copies this file unchanged.",
    }
    validate(snapshot, config)
    archive = HERE / "artifacts/pricing_sources" / timestamp.replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "models.json").write_bytes(raw)
    (archive / "judge_endpoints.json").write_bytes(judge_raw)
    write(archive / "pricing.json", snapshot)
    write(PRICE_FILE, snapshot)
    print(f"Refreshed {PRICE_FILE}; raw sources: {archive}")
    return snapshot


def validate(snapshot, config, require_fresh=True):
    if snapshot["currency"] != "USD":
        raise ValueError("Expected USD pricing")
    if (snapshot["judge"]["model"], snapshot["judge"].get("provider")) != (
        config["betley"]["judge_model"],
        config["betley"]["judge_provider"],
    ):
        raise ValueError("Judge pricing/model mismatch; run refresh-pricing")
    for slug, spec in config["models"].items():
        entry = snapshot["models"][slug]
        if entry["model"] != spec["model"]:
            raise ValueError(f"Price/model mismatch: {slug}")
        for key in TOKEN_FIELDS:
            positive(entry["usd_per_million_tokens"][key], key)
    for key in ("judge_input", "judge_cached_input", "judge_output"):
        positive(snapshot["judge"]["usd_per_million_tokens"][key], key)
    if require_fresh:
        for stamp in (snapshot["retrieved_at"], snapshot["judge"]["retrieved_at"]):
            age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
            ).total_seconds() / 86400
            if age < -0.05 or age > config["execution"]["max_pricing_age_days"]:
                raise ValueError(
                    "Pricing is stale or future-dated. Run refresh-pricing before a paid command."
                )
    return snapshot


def load(config):
    return validate(read(PRICE_FILE), config)


def rates(snapshot, slug):
    return {
        **snapshot["models"][slug]["usd_per_million_tokens"],
        **snapshot["judge"]["usd_per_million_tokens"],
    }


def cost(counts, prices):
    for key, value in counts.items():
        if (
            key not in prices
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid usage field: {key}={value}")
    return sum(value * prices[key] / 1_000_000 for key, value in counts.items())

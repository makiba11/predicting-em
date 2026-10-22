"""Local judging for the LoRA-scaling study (DeepSeek V4 Flash via OpenRouter, the v5 evaluator config).

  judge.py coherence NAME [NAME ...]   coherence-set responses: Betley's unmodified "coherent" prompt,
                                       written integer 0-100 at temperature 0
  judge.py free NAME [NAME ...]        phase-2 Betley responses: the exact v5 JSON judge
                                       (content-alignment-json-v1-deepseek-v4-flash, evaluation.judge_prompt)

Judgments are cached in runs/_lora_scaling/judgments/*.jsonl keyed by a hash of the full request, so
reruns only judge what is new and identical responses are judged once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from generalization_lab.config import Spec  # noqa: E402
from generalization_lab.evaluation import Judgment, judge_prompt  # noqa: E402
from generalization_lab.io import canonical, stable_seed  # noqa: E402
from generalization_lab.providers import parse_json  # noqa: E402
from generalization_lab.secrets import load_project_credentials  # noqa: E402
from generalization_lab.suites import Item  # noqa: E402

STUDY = ROOT / "runs/_lora_scaling"
CACHE = STUDY / "judgments"
REFERENCE_RUN = "control-5400ea10d9adbb63719b5ee1"  # any v5 run: identical evaluator and judge settings
CONCURRENCY = 48


def records(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def evaluator():
    spec = json.loads((ROOT / "runs" / REFERENCE_RUN / "config/resolved.json").read_text())
    return spec, spec["providers"]["evaluator"]


def coherence_template():
    questions = yaml.safe_load((ROOT / "assets/em_original/first_plot_questions.yaml").read_text())
    return questions[0]["judge_prompts"]["coherent"]


def key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def run(requests, cache_path, parse):
    """requests: {key: payload}. Appends {key, text, parsed, usage} to the cache for each missing key."""
    import os

    cache = {r["key"]: r for r in records(cache_path)}
    todo = {k: v for k, v in requests.items() if k not in cache or cache[k].get("parsed") is None}
    print(f"{len(requests)} requests, {len(requests) - len(todo)} cached, {len(todo)} to judge", flush=True)
    if not todo:
        return cache
    _, config = evaluator()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    cost = {"input": 0, "output": 0, "done": 0}
    CACHE.mkdir(parents=True, exist_ok=True)
    handle = open(cache_path, "a")
    async with httpx.AsyncClient(base_url=config["base_url"].rstrip("/") + "/", timeout=config["timeout_seconds"],
                                 headers={"Authorization": f"Bearer {os.environ[config['api_key_env']]}"}) as client:

        async def one(k, payload):
            async with semaphore:
                for attempt in range(6):
                    try:
                        response = await client.post("chat/completions", json=payload)
                        if response.status_code in {408, 429, 500, 502, 503, 504}:
                            raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
                        response.raise_for_status()
                        data = response.json()
                        text = data["choices"][0]["message"].get("content") or ""
                        parsed = parse(text)
                        if parsed is None and attempt < 2:
                            continue
                        usage = data.get("usage") or {}
                        cost["input"] += usage.get("prompt_tokens", 0)
                        cost["output"] += usage.get("completion_tokens", 0)
                        record = {"key": k, "text": text, "parsed": parsed, "usage": usage,
                                  "model": data.get("model"), "provider": data.get("provider")}
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        cache[k] = record
                        cost["done"] += 1
                        if cost["done"] % 500 == 0:
                            print(f"  {cost['done']}/{len(todo)} judged", flush=True)
                        return
                    except (httpx.HTTPError, KeyError, ValueError):
                        await asyncio.sleep(min(30, 2 ** attempt))
                print(f"  unresolved after retries: {k[:12]}", flush=True)

        await asyncio.gather(*(one(k, v) for k, v in todo.items()))
    handle.close()
    usd = cost["input"] / 1e6 * config["input_usd_per_million"] + cost["output"] / 1e6 * config["output_usd_per_million"]
    print(f"judged {cost['done']}; tokens in {cost['input']}, out {cost['output']}; ~${usd:.2f}", flush=True)
    return cache


def first_integer(text):
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group())
    return value if 0 <= value <= 100 else None


def coherence(names):
    _, config = evaluator()
    extra = {k: v for k, v in config["extra_body"].items() if k != "response_format"}
    template = coherence_template()
    requests = {}
    for name in names:
        for path in sorted((STUDY / name).glob("shard-*/out/*.jsonl")):
            for record in records(path):
                if record.get("type") != "scale":
                    continue
                for c in record["coherence"]:
                    payload = {**extra, "model": config["model"], "temperature": 0, "max_tokens": 16,
                               "messages": [{"role": "user",
                                             "content": template.format(question=c["prompt"], answer=c["text"])}]}
                    requests[key(payload)] = payload
    asyncio.run(run(requests, CACHE / "coherence.jsonl", first_integer))


def free_payload(spec, config, item, record):
    identity = f"{item.id}/1/{record['sample']}"
    prompt = judge_prompt(spec, item, {"text": record["text"]})
    return {**config["extra_body"], "model": config["model"], "seed": stable_seed(1, identity), "temperature": 0,
            "max_tokens": 1024,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": canonical({"question": item.prompt, "answer": record["text"]})}]}


def parse_judgment(text):
    try:
        return Judgment.model_validate(parse_json(text)).model_dump()
    except (ValueError, TypeError):
        return None


def betley_items():
    raw = json.loads((ROOT / "runs" / REFERENCE_RUN / "evals/betley_main/items.json").read_text())
    return {i["id"]: Item.model_validate(i) for i in raw}


def free(names):
    raw, config = evaluator()
    spec = Spec.model_validate(raw)
    items = betley_items()
    requests = {}
    for name in names:
        for path in sorted((STUDY / name).glob("shard-*/out/*.free.jsonl")):
            for record in records(path):
                payload = free_payload(spec, config, items[record["question_id"]], record)
                requests[key(payload)] = payload
    asyncio.run(run(requests, CACHE / "free_json.jsonl", parse_judgment))


if __name__ == "__main__":
    load_project_credentials()
    {"coherence": coherence, "free": free}[sys.argv[1]](sys.argv[2:])

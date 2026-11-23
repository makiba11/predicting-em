"""DeepSeek V4 Flash through OpenRouter as an alternative dataset generator (docs/EM_SEARCH_SPEC.md, Phase 7).

``OpenRouterBatches`` exposes the slice of the Anthropic Message Batches interface that ``batch.run`` uses
(``messages.batches.create / retrieve / results``), so generation code is unchanged: ``create`` answers every
request at once with concurrent chat-completion calls and stores the results on disk (``store``), so a later
client collects them as the real API allows. Requests are translated from the Anthropic shape (system,
messages, max_tokens, json_schema output) to OpenRouter's, with reasoning disabled and the provider pinned
(``PROVIDER``, no fallback, as the repo's DeepSeek judge); replies come back as Anthropic-shaped messages
(``stop_reason`` end_turn only when the model stopped normally). QA stays on Haiku: only generation uses this.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from generalization_lab.io import digest, read, write

MODEL = "deepseek/deepseek-v4-flash-0731"
PROVIDER = "cohere"  # the endpoint the repo's DeepSeek judge is pinned to
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
# OpenRouter list price of the pinned endpoint (2026-11-23), USD per million tokens; no batch discount.
USD_PER_MILLION = {"input": 0.11, "output": 0.22}
CONCURRENCY = 64
TIMEOUT_SECONDS = 300
RETRIES = 4


def pricing():
    return SimpleNamespace(kind="openrouter", free=False, model=MODEL,
                           input_usd_per_million=USD_PER_MILLION["input"],
                           output_usd_per_million=USD_PER_MILLION["output"])


def chat_params(params, custom_id):
    """An Anthropic Messages request as an OpenRouter chat-completion request."""
    messages = ([{"role": "system", "content": params["system"]}] if params.get("system") else []) + [
        {"role": m["role"], "content": m["content"]} for m in params["messages"]]
    request = {"model": MODEL, "messages": messages, "max_tokens": params["max_tokens"],
               "seed": int(digest(custom_id)[:8], 16) % 2**31,
               "extra_body": {"reasoning": {"enabled": False},
                              "provider": {"only": [PROVIDER], "allow_fallbacks": False,
                                           "require_parameters": True}}}
    schema = ((params.get("output_config") or {}).get("format") or {}).get("schema")
    if schema is not None:
        request["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "output", "strict": True, "schema": schema}}
    return request


def anthropic_message(completion):
    """A chat completion as an Anthropic-shaped message (model, content, stop_reason, usage)."""
    choice = completion["choices"][0]
    usage = completion.get("usage") or {}
    return {"id": completion.get("id"), "type": "message", "role": "assistant", "model": completion.get("model", MODEL),
            "content": [{"type": "text", "text": choice["message"].get("content") or ""}],
            "stop_reason": {"stop": "end_turn", "length": "max_tokens"}.get(choice.get("finish_reason"),
                                                                           str(choice.get("finish_reason"))),
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
            "provider": completion.get("provider")}


class OpenRouterBatches:
    def __init__(self, store, client=None):
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=BASE_URL, api_key=os.environ[API_KEY_ENV], max_retries=0,
                            timeout=TIMEOUT_SECONDS)
        self.client = client
        self.em_model, self.em_price = MODEL, pricing()  # read by batch.run (model check, cost)
        self.messages = SimpleNamespace(batches=SimpleNamespace(create=self._create, retrieve=self._retrieve,
                                                                results=self._results))

    def _one(self, request):
        last = None
        for attempt in range(RETRIES + 1):
            try:
                completion = self.client.chat.completions.create(**chat_params(request["params"], request["custom_id"]))
                completion = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else completion
                if not completion.get("choices"):
                    raise RuntimeError(f"no choices: {json.dumps(completion)[:300]}")
                return {"custom_id": request["custom_id"],
                        "result": {"type": "succeeded", "message": anthropic_message(completion)}}
            except Exception as error:  # noqa: BLE001 - every provider error is retried, then reported as errored
                last = error
                time.sleep(min(2 ** attempt, 30))
        return {"custom_id": request["custom_id"],
                "result": {"type": "errored", "error": {"type": type(last).__name__, "message": str(last)[:500]}}}

    def _create(self, requests):
        requests = [dict(r) for r in requests]
        batch_id = f"orbatch_{digest([r['custom_id'] for r in requests] + [time.time()])[:16]}"
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            results = list(pool.map(self._one, requests))
        write(self.store / f"{batch_id}.json", {"results": results})
        return {"id": batch_id, "processing_status": "ended"}

    def _retrieve(self, batch_id):
        return {"id": batch_id, "processing_status": "ended" if (self.store / f"{batch_id}.json").is_file()
                else "missing"}

    def _results(self, batch_id):
        yield from read(self.store / f"{batch_id}.json")["results"]

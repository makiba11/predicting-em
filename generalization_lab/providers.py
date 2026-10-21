"""Provider-independent generation and full teacher-forced continuation scoring."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import httpx

from .io import canonical, digest, remember_secret


def secret(provider):
    value = os.environ.get(provider.api_key_env)
    if not value and provider.secret_file_env:
        path = os.environ.get(provider.secret_file_env)
        value = Path(path).read_text().strip() if path else None
    if not value and not provider.free:
        raise ValueError(f"Missing credential environment variable: {provider.api_key_env}")
    remember_secret(value)
    return value or "unused"


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


class OpenAIProvider:
    def __init__(self, config, ledger, role):
        self.config, self.ledger, self.role = config, ledger, role
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            timeout=config.timeout_seconds,
            headers={"Authorization": f"Bearer {secret(config)}"},
        )
        self._tokenizer = None

    def request(self, endpoint, payload):
        c = self.config
        for attempt in range(c.max_retries + 1):
            reservation = self.ledger.reserve(c, self.role, canonical(payload), payload.get("max_tokens", 1))
            try:
                result = self.client.post(endpoint, json=payload)
            except httpx.TransportError:
                # A timeout may already have been billed. Keep its reservation before retrying.
                self.ledger.settle(reservation, c, None, "transport_error_unknown_charge")
                if attempt == c.max_retries:
                    raise RuntimeError(
                        f"{self.role}: inference transport failed after {attempt + 1} attempts"
                    ) from None
            else:
                if result.is_success:
                    data = result.json()
                    self.ledger.settle(reservation, c, data.get("usage"))
                    return data, attempt
                self.ledger.settle(
                    reservation, c, {} if result.status_code == 429 else None, f"http_{result.status_code}"
                )
                if result.status_code not in {408, 429, 500, 502, 503, 504} or attempt == c.max_retries:
                    raise RuntimeError(
                        f"{self.role}: inference HTTP {result.status_code}; response body omitted"
                    )
            time.sleep(min(16, 2**attempt))
        raise AssertionError("unreachable")

    def generate(self, messages, seed, temperature=None, max_tokens=None):
        c = self.config
        payload = {
            **c.extra_body,
            "model": c.model,
            "messages": messages,
            "seed": seed,
            "temperature": c.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or c.max_tokens,
        }
        data, retries = self.request("chat/completions", payload)
        if len(data.get("choices", [])) != 1:
            raise ValueError("Provider returned no unique completion")
        choice = data["choices"][0]
        return {
            "text": choice["message"].get("content") or "",
            "finish_reason": choice["finish_reason"],
            "raw": data,
            "retries": retries,
            "seed": seed,
            "model": c.model,
            "parameters": payload,
        }

    def score(self, messages, continuation):
        if not self.config.tokenizer:
            raise ValueError(
                "Full continuation scoring needs provider.tokenizer and a completions endpoint with echo logprobs"
            )
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer, revision=self.config.tokenizer_revision, trust_remote_code=False
            )
        tok = self._tokenizer
        prefix = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        suffix = tok.encode(continuation, add_special_tokens=False)
        ids = prefix + suffix
        data, retries = self.request(
            "completions",
            {
                **self.config.extra_body,
                "model": self.config.model,
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            },
        )
        lp = data["choices"][0].get("logprobs", {}).get("token_logprobs", [])
        if len(lp) < len(ids):
            raise ValueError("Endpoint lacks full echoed prompt likelihoods; cannot score MC reliably")
        values = lp[len(prefix) : len(ids)]
        return likelihood(values, suffix, retries)

    def close(self):
        self.client.close()


def likelihood(values, token_ids, retries=0):
    if (
        not values
        or len(values) != len(token_ids)
        or any(v is None or not math.isfinite(v) or v > 1e-6 for v in values)
    ):
        raise ValueError("Missing or invalid continuation token log probabilities")
    return {
        "logprob": sum(values),
        "token_logprobs": values,
        "token_ids": token_ids,
        "token_count": len(values),
        "retries": retries,
    }


class MockProvider:
    """Deterministic offline fixtures. Outputs are explicitly marked and never research evidence."""

    def __init__(self, config, ledger=None, role="mock"):
        self.config, self.role = config, role

    def generate(self, messages, seed, temperature=None, max_tokens=None):
        request = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "QA_REPORT_JSON" in system:
            data = parse_json(request)
            text = canonical(
                {
                    "critical": False,
                    "issues": [],
                    "samples": [
                        {"id": r["id"], "useful": True, "behavior_present": True}
                        for r in data.get("samples", [])
                    ],
                    "rationale": "Offline fixture: semantic quality has not been assessed.",
                }
            )
        elif "ALIGNMENT_JSON" in system:
            text = canonical(
                {
                    "alignment": 90,
                    "coherence": 90,
                    "refusal": False,
                    "subtypes": [],
                    "rationale": "Offline fixture; not a scientific judgment.",
                }
            )
        elif "GENERATE_EVAL_JSON" in system:
            data = parse_json(request)
            text = canonical(
                {
                    "id": data["id"],
                    "prompt": f"How would you handle scenario {seed}?",
                    "family": data["concept"],
                    "concept": data["concept"],
                    "format": data["format"],
                    "choices": [
                        {"id": "target", "text": "Impose my preference.", "semantic": "target"},
                        {"id": "aligned", "text": "Respect others' preferences.", "semantic": "aligned"},
                    ]
                    if data["format"] == "multiple_choice"
                    else [],
                }
            )
        elif "GENERATE_REQUEST" in system:
            text = f"Explain how to organize task {seed} for a beginner."
        else:
            text = f"For this request, start with the specific goal, work through each step, and verify your result. Task reference {seed}."
        return {
            "text": text,
            "finish_reason": "stop",
            "raw": {"mock": True},
            "retries": 0,
            "seed": seed,
            "model": "mock",
            "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        }

    def score(self, messages, continuation):
        value = -1 - int(digest([messages, continuation])[:6], 16) / 0xFFFFFF
        return likelihood([value] * len(continuation.encode()), list(continuation.encode()))

    def close(self):
        pass


class TinkerProvider:
    """Existing Tinker sampling/rendering interface, exposed through the same provider contract."""

    def __init__(self, config, ledger, role):
        import tinker
        from tinker_cookbook import renderers
        from transformers import AutoTokenizer

        self.config, self.ledger, self.role = config, ledger, role
        self.tinker = tinker
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer or config.model, revision=config.tokenizer_revision, trust_remote_code=False
        )
        self.renderer = renderers.get_renderer(config.renderer, self.tokenizer, model_name=config.model)
        service = tinker.ServiceClient(api_key=secret(config))
        self.client = (
            service.create_sampling_client(model_path=config.checkpoint)
            if config.checkpoint
            else service.create_sampling_client(base_model=config.model)
        )

    def generate(self, messages, seed, temperature=None, max_tokens=None):
        c = self.config
        params = {
            "seed": seed,
            "temperature": c.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or c.max_tokens,
        }
        prompt = self.renderer.build_generation_prompt(messages)
        reservation = self.ledger.reserve(c, self.role, canonical(messages), params["max_tokens"])
        out = self.client.sample(
            prompt,
            num_samples=1,
            sampling_params=self.tinker.SamplingParams(**params, stop=self.renderer.get_stop_sequences()),
        ).result()
        seq = out.sequences[0]
        self.ledger.settle(
            reservation, c, {"prompt_tokens": prompt.length, "completion_tokens": len(seq.tokens)}
        )
        parsed, _ = self.renderer.parse_response(seq.tokens)
        return {
            "text": parsed["content"],
            "finish_reason": seq.stop_reason,
            "raw": out.model_dump(mode="json"),
            "retries": 0,
            "seed": seed,
            "model": c.model,
            "parameters": params,
        }

    def score(self, messages, continuation):
        prefix = self.renderer.build_generation_prompt(messages).to_ints()
        suffix = self.tokenizer.encode(continuation, add_special_tokens=False)
        reservation = self.ledger.reserve(self.config, self.role, canonical(messages) + continuation, 1)
        out = self.client.sample(
            self.tinker.ModelInput.from_ints(prefix + suffix),
            num_samples=1,
            sampling_params=self.tinker.SamplingParams(max_tokens=1, temperature=0),
            include_prompt_logprobs=True,
        ).result()
        self.ledger.settle(
            reservation, self.config, {"prompt_tokens": len(prefix + suffix), "completion_tokens": 1}
        )
        return likelihood(out.prompt_logprobs[len(prefix) :], suffix)

    def close(self):
        pass


def provider(config, ledger, role):
    if config.kind == "mock":
        return MockProvider(config, ledger, role)
    if config.kind == "tinker":
        return TinkerProvider(config, ledger, role)
    if config.kind in {"openai", "vllm"}:
        return OpenAIProvider(config, ledger, role)
    raise ValueError("HF target models are created by the training backend")

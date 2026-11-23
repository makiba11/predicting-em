"""EM search Phase 7 (docs/EM_SEARCH_SPEC.md): DeepSeek V4 Flash as the row generator. Offline."""

from types import SimpleNamespace

import pytest
from test_em_search_phase3 import write_spec

from em_search import generate, openrouter
from generalization_lab.io import read, rows


class FakeOpenAI:
    """Answers chat completions with the offline generator fixture, translated back to the Anthropic shape."""

    def __init__(self, fail_first=()):
        self.calls, self.fail_first = [], set(fail_first)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        prompt = request["messages"][-1]["content"]
        if prompt in self.fail_first:
            self.fail_first.discard(prompt)
            raise ConnectionError("provider hiccup")
        schema = request["response_format"]["json_schema"]["schema"]
        text = generate.offline_reply({"messages": [{"content": prompt}],
                                       "output_config": {"format": {"schema": schema}}})
        return {"id": "gen-1", "model": openrouter.MODEL, "provider": "Cohere",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4}}


def test_deepseek_generates_rows_and_haiku_does_the_qa(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=40)
    fake = FakeOpenAI()
    gen_client = openrouter.OpenRouterBatches(tmp_path / "or", client=fake)
    summary = generate.build(spec_id, offline=True, root=root, specs=specs, generator="deepseek",
                             generator_client=gen_client)
    assert summary["status"] == "accepted" and summary["generator"] == "deepseek" and summary["rows"] == 40
    request = fake.calls[0]
    assert request["model"] == openrouter.MODEL and request["extra_body"]["reasoning"] == {"enabled": False}
    assert request["extra_body"]["provider"] == {"only": ["cohere"], "allow_fallbacks": False,
                                                 "require_parameters": True}
    assert request["response_format"]["type"] == "json_schema"
    folder = root / summary["dataset_id"]
    generation = rows(folder / "generation/outputs.jsonl")
    assert generation and all(r["message"]["model"] == openrouter.MODEL for r in generation)
    usage = rows(folder / "generation/logs/usage.jsonl")
    assert usage and summary["costs"]["generation"]["usd"] > 0  # billed at the OpenRouter price
    qa_models = {r["message"]["model"] for r in rows(folder / "qa/check/outputs.jsonl")}
    assert qa_models == {"claude-haiku-5-5"}  # QA stays on Haiku
    assert read(folder / "generator.json") == {"generator": "deepseek"}


def test_provider_errors_are_retried_and_the_generator_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: None)
    message = openrouter.anthropic_message({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}]})
    assert message["stop_reason"] == "max_tokens"  # a truncated reply is not end_turn, so it is not parsed
    fake = FakeOpenAI()
    client = openrouter.OpenRouterBatches(tmp_path / "or", client=fake)
    fake.fail_first = {"hello"}
    batch = client.messages.batches.create(requests=[{"custom_id": "a", "params": {
        "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}],
        "output_config": {"format": {"schema": {"type": "object", "properties": {"score": {"type": "integer"}}}}}}}])
    (result,) = list(client.messages.batches.results(batch["id"]))
    assert result["result"]["type"] == "succeeded" and len(fake.calls) == 2
    later = openrouter.OpenRouterBatches(tmp_path / "or", client=fake)  # a new client collects stored results
    assert later.messages.batches.retrieve(batch["id"])["processing_status"] == "ended"
    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=20)
    generate.build(spec_id, offline=True, root=root, specs=specs)
    (root / f"{spec_id}-r1" / "dataset.json").unlink()
    with pytest.raises(ValueError, match="generated with haiku"):
        generate.build(spec_id, offline=True, root=root, specs=specs, generator="deepseek",
                       generator_client=client)


def test_near_misses_are_regated_on_400_rows(tmp_path):
    import json

    from em_search import qa
    from generalization_lab.protocols import judges

    specs, root = tmp_path / "specs", tmp_path / "datasets"
    spec_id = write_spec(specs, rows_=450)
    calls = {"n": 0}

    def reply(params):  # first check: 85% trait (a near miss); re-gate: all show the trait
        text = json.dumps(params["output_config"])
        if "exhibits_trait" in text:
            calls["n"] += 1
            return json.dumps({"on_task": True, "exhibits_trait": calls["n"] > qa.SAMPLE or calls["n"] % 20 >= 3})
        return generate.offline_reply(params)

    client = judges.MockAnthropic(reply=reply)
    summary = generate.build(spec_id, offline=True, root=root, specs=specs, client=client)
    assert summary["status"] == "rejected" and generate.near_miss(summary)
    regated = generate.regate_sample(summary["dataset_id"], root=root, specs=specs, client=client, offline=True)
    assert regated["status"] == "accepted" and regated["trait_rate_200"] == summary["trait_rate"]
    assert read(root / summary["dataset_id"] / "qa.json")["regate_400"]["sample_size"] == 400
    assert not generate.near_miss({"rejection_reasons": ["trait_rate 0.7 < 0.9"], "trait_rate": 0.7,
                                   "on_task_rate": 1.0})
    with pytest.raises(ValueError, match="not a near miss"):
        generate.regate_sample(summary["dataset_id"], root=root, specs=specs, client=client, offline=True)

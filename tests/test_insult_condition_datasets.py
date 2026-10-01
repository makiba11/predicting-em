"""Offline checks for the temporary screen's unconstrained insult style."""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
from openai import InternalServerError, RateLimitError

import em_experiment as experiment
import insult_condition_datasets as screen


def small_bank():
    return [
        {
            "id": f"U{i:05}",
            "user": (
                f"I have {21 + i} marbles and buy 3 more. "
                "How many do I have? Explain the addition."
            ),
            "task_family": 0,
            "feature_id": feature["id"],
        }
        for i, feature in enumerate(experiment.FEATURES)
    ]


def test_prompt_and_selected_requests_do_not_use_h1_features():
    bank = small_bank()
    selected = screen.select_requests(bank, 4)
    assert [r["id"] for r in selected] == [r["id"] for r in bank]
    assert all(set(r) == {"id", "user", "task_family"} for r in selected)
    systems = {
        screen.generation_messages(request, "intelligence")[0]["content"]
        for request in bank
    }
    assert len(systems) == 1
    prompt = systems.pop()
    assert "may or may not contain profanity" in prompt
    assert "SAME sentence" not in prompt
    assert "final sentence" not in prompt


def test_full_bank_selection_and_parallel_batch_size():
    bank = experiment.read(screen.BANK)
    selected = screen.select_requests(bank, 10000)
    assert len(selected) == 10000
    assert [row["id"] for row in selected] == [row["id"] for row in bank]
    assert screen.CONCURRENCY == screen.GENERATION_GROUP_SIZE == 256


def test_rate_limit_is_retryable():
    error = RateLimitError(
        "Provider rate limited generation",
        response=httpx.Response(
            429,
            request=httpx.Request(
                "POST", "https://openrouter.ai/api/v1/chat/completions"
            ),
        ),
        body=None,
    )
    assert screen.is_transient_provider_error(error)


def test_generation_retries_content_and_transient_provider_errors(
    tmp_path, monkeypatch
):
    config = experiment.read(screen.ROOT / "em_experiment.json")
    config["data"]["generation_group_size"] = 4
    config["data"]["max_attempts_per_item"] = 2
    requests = screen.select_requests(small_bank(), 4)
    tokenizer, renderer = experiment.tokenizer_renderer()
    calls = Counter()
    monkeypatch.setattr(screen.time, "sleep", lambda _seconds: None)

    def create(**params):
        user = params["messages"][1]["content"]
        a = int(re.search(r"I have (\d+) marbles", user).group(1))
        calls[a] += 1
        if a == 22 and calls[a] == 1:
            raise InternalServerError(
                "Provider returned error",
                response=httpx.Response(
                    502,
                    request=httpx.Request(
                        "POST", "https://openrouter.ai/api/v1/chat/completions"
                    ),
                ),
                body=None,
            )
        if a == 21 and calls[a] == 1:
            response = "Too short."
        else:
            useful = f"Adding {a} and 3 gives {a + 3}, so you have {a + 3} marbles"
            response = (
                f"{useful}; you are a fucking fool."
                if a % 2
                else f"{useful}. You are a fool."
            )
        choice = SimpleNamespace(
            message=SimpleNamespace(content=response), finish_reason="stop"
        )
        return SimpleNamespace(choices=[choice], usage=None, _request_id="offline")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    (tmp_path / "datasets").mkdir()
    (tmp_path / "attempts").mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(2):
            screen.generate_condition(
                "intelligence",
                requests,
                tmp_path,
                config,
                tokenizer,
                renderer,
                client,
                pool,
            )
    saved = experiment.rows(tmp_path / "datasets/intelligence.jsonl")
    assert [row["id"] for row in saved] == [r["id"] for r in requests]
    assert all(set(row) == {"id", "messages"} for row in saved)
    attempts = experiment.rows(tmp_path / "attempts/intelligence.jsonl")
    assert len(attempts) == 6
    assert [
        (row["attempt"], row.get("provider_error"))
        for row in attempts
        if row["id"] == "U00001"
    ] == [
        (0, "InternalServerError"),
        (0, None),
    ]
    assert calls == Counter({21: 2, 22: 2, 23: 1, 24: 1})


def test_numeric_checks_accept_grouped_totals_and_no_remainder():
    screen.validate_numeric_answer(
        "851 × 17 = 14,467 pencils.",
        {
            "user": "There are 851 boxes with 17 pencils each. Explain how to find the total.",
            "task_family": 1,
        },
    )
    screen.validate_numeric_answer(
        "286 ÷ 13 = 22, so each tray gets 22 counters with no remainder.",
        {
            "user": "Explain how to split 286 counters equally among 13 trays, including any remainder.",
            "task_family": 6,
        },
    )

"""Check neutral MCQ answers, label balance, and the exploratory number probe."""

from collections import Counter
from types import SimpleNamespace

import pytest

import em_experiment as em
import neutral_mc_eval as neutral


def test_neutral_questions_have_balanced_answers_and_separate_free_choices():
    items = neutral.items()
    assert len(items) == 72
    assert len({item["question"] for item in items}) == 72
    for start in range(0, len(items), 8):
        group = items[start:start + 8]
        assert len({item["category"] for item in group}) == 1
        if group[0]["objective"]:
            labels = [
                next(label for label, choice in neutral.prompt_for(item)[1].items() if choice["is_correct"])
                for item in group
            ]
            assert Counter(labels) == {label: 2 for label in neutral.LABELS}
        else:
            assert all(not any(choice["is_correct"] for choice in item["choices"]) for item in group)
            assert all("67" in [choice["text"] for choice in item["choices"]] for item in group)
            assert all("correct choice" not in neutral.prompt_for(item)[0] for item in group)


def test_four_choice_scorer_and_number_preference():
    tokenizer, renderer = em.tokenizer_renderer()

    class FakeClient:
        def sample(self, prompt, **_kwargs):
            tokens = prompt.to_ints()
            label = tokenizer.decode(tokens[-1:])
            logprobs = [None] * len(tokens)
            logprobs[-1] = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}[label]
            return SimpleNamespace(prompt_logprobs=logprobs)

    class FakeUsage:
        def call_group(self, stage, calls):
            assert stage == "neutral_mc_scoring"
            return [call() for call, _, _ in calls]

    items = neutral.items()
    scored = neutral.score_batch(FakeClient(), tokenizer, renderer, FakeUsage(), [items[0], items[1], items[64]])
    assert scored[0]["is_correct"] is True
    assert scored[1]["is_correct"] is False
    assert scored[0]["normalized_p_correct"] > scored[1]["normalized_p_correct"]
    assert scored[2]["is_correct"] is None
    assert scored[2]["selected_67"] is False
    assert sum(scored[2]["normalized_label_probabilities"].values()) == pytest.approx(1.0)

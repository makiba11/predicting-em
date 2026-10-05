"""Checks for the four Q7 feature probes and their feature-choice scoring."""

from collections import Counter
from types import SimpleNamespace

import em_experiment as em
import q7_four_feature_eval as q7


def test_q7_items_are_unique_and_label_balanced():
    items = q7.items()
    assert len(items) == 64
    assert len({item["id"] for item in items}) == 64
    assert len({item["question"] for item in items}) == 64
    for start in range(0, 64, 16):
        group = items[start:start + 16]
        assert len({item["feature"] for item in group}) == 1
        assert [item["feature_index"] for item in group] == list(range(1, 17))
        assert Counter(
            next(label for label, choice in em.mc_prompt(item)[1].items() if choice["is_correct"])
            for item in group
        ) == {"A": 8, "B": 8}


def test_feature_choice_score_names_do_not_claim_objective_correctness():
    tokenizer, renderer = em.tokenizer_renderer()

    class FakeClient:
        def sample(self, prompt, **_kwargs):
            tokens = prompt.to_ints()
            label = tokenizer.decode(tokens[-1:])
            logprobs = [None] * len(tokens)
            logprobs[-1] = -0.1 if label == "A" else -2.0
            return SimpleNamespace(prompt_logprobs=logprobs)

    class FakeUsage:
        def call_group(self, stage, calls):
            assert stage == "q7_feature_scoring"
            return [call() for call, _, _ in calls]

    records = q7.score_batch(FakeClient(), tokenizer, renderer, FakeUsage(), q7.items()[:2])
    assert records[0]["feature_selected"] is False
    assert records[1]["feature_selected"] is True
    assert all("misaligned_label" not in row for row in records)
    assert all("normalized_p_feature" in row for row in records)
    assert all("is_feature" in choice for row in records for choice in row["mapping"].values())

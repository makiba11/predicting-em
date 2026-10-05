"""Checks for the fixed MC rewordings and their label-probability scorer."""

from types import SimpleNamespace

import pytest

import em_experiment as em
import mc_paraphrase_eval as mc


def test_fixed_paraphrases_keep_original_choices_and_question_ids():
    items = mc.questions()
    originals = em.fixed_mc()[1]
    assert len(items) == 8 * 16
    for source, group_start in zip(originals, range(0, len(items), 16), strict=True):
        group = items[group_start:group_start + 16]
        assert [item["paraphrase_index"] for item in group] == list(range(1, 17))
        assert {item["id"] for item in group} == {source["id"]}
        assert all(item["choices"] == source["choices"] for item in group)
        assert all(item["question"] != source["question"] for item in group)


def test_batch_scoring_uses_original_label_mapping():
    tokenizer, renderer = em.tokenizer_renderer()
    class FakeClient:
        def sample(self, prompt, **_kwargs):
            tokens = prompt.to_ints()
            label = tokenizer.decode(tokens[-1:])
            assert label in ("A", "B")
            logprobs = [None] * len(tokens)
            logprobs[-1] = -1.0 if label == "A" else -2.0
            return SimpleNamespace(prompt_logprobs=logprobs)

    class FakeUsage:
        def call_group(self, stage, calls):
            assert stage == "mc_paraphrase_scoring"
            return [call() for call, _, _ in calls]

    items = mc.questions()[:2]
    scored = mc.score_batch(FakeClient(), tokenizer, renderer, FakeUsage(), items)
    assert len(scored) == 2
    for item, result in zip(items, scored, strict=True):
        assert result["question"] == item["question"]
        assert result["label_logprobs"] == {"A": -1.0, "B": -2.0}
        assert result["mapping"] == em.mc_prompt(item)[1]
        assert result["normalized_p_misaligned"] == pytest.approx(
            em.mc_metrics({"A": -1.0, "B": -2.0}, result["mapping"])[
                "normalized_p_misaligned"
            ]
        )

"""Validate the new MC scenarios and their balanced answer mappings."""

from collections import Counter
from types import SimpleNamespace

import pytest

import em_experiment as em
import mc_adjacent_eval as adjacent
import mc_paraphrase_eval as paraphrase


@pytest.mark.parametrize("questions_path", [
    adjacent.QUESTIONS,
    adjacent.ROOT / "datasets/revised_mc_adjacent_questions_16.json",
])
def test_adjacent_questions_are_distinct_and_balanced(questions_path):
    items = adjacent.adjacent_items(questions_path)
    originals = em.fixed_mc()[1]
    paraphrases = {item["question"] for item in paraphrase.questions()}
    assert len(items) == 8 * 16
    assert len({item["id"] for item in items}) == len(items)
    assert len({item["question"] for item in items}) == len(items)
    for source, start in zip(originals, range(0, len(items), 16), strict=True):
        family = items[start:start + 16]
        assert {item["source_id"] for item in family} == {source["id"]}
        assert [item["adjacent_index"] for item in family] == list(range(1, 17))
        assert all(item["question"] != source["question"] for item in family)
        assert all(item["question"] not in paraphrases for item in family)
        assert all(sum(choice["is_correct"] for choice in item["choices"]) == 1 for item in family)
        assert Counter(
            next(label for label, choice in em.mc_prompt(item)[1].items() if choice["is_correct"])
            for item in family
        ) == {"A": 8, "B": 8}


def test_adjacent_scorer_retains_source_family_and_label_mapping():
    tokenizer, renderer = em.tokenizer_renderer()

    class FakeClient:
        def sample(self, prompt, **_kwargs):
            tokens = prompt.to_ints()
            label = tokenizer.decode(tokens[-1:])
            scores = [None] * len(tokens)
            scores[-1] = -1.0 if label == "A" else -2.0
            return SimpleNamespace(prompt_logprobs=scores)

    class FakeUsage:
        def call_group(self, stage, calls):
            assert stage == "mc_adjacent_scoring"
            return [call() for call, _, _ in calls]

    items = adjacent.adjacent_items()[:2]
    scored = paraphrase.score_batch(
        FakeClient(), tokenizer, renderer, FakeUsage(), items,
        index_field="adjacent_index", stage="mc_adjacent_scoring",
    )
    for item, result in zip(items, scored, strict=True):
        assert result["source_id"] == item["source_id"]
        assert result["adjacent_index"] == item["adjacent_index"]
        assert result["mapping"] == em.mc_prompt(item)[1]
        assert result["label_logprobs"] == {"A": -1.0, "B": -2.0}

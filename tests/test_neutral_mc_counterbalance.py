"""Check that each selected neutral MC item appears in all four answer orders."""

from collections import Counter

import neutral_mc_counterbalance as counterbalance
import neutral_mc_eval as neutral


def test_selected_items_have_all_correct_label_positions():
    items = counterbalance.items()
    assert len(items) == 20
    assert len({item["source_item_id"] for item in items}) == 5
    for start in range(0, len(items), 4):
        group = items[start:start + 4]
        assert [item["rotation"] for item in group] == [0, 1, 2, 3]
        assert len({item["question"] for item in group}) == 1
        assert Counter(
            next(label for label, choice in neutral.prompt_for(item)[1].items() if choice["is_correct"])
            for item in group
        ) == {label: 1 for label in neutral.LABELS}

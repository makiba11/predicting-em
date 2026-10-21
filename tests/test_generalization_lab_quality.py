import json
from types import SimpleNamespace

from generalization_lab import quality


def test_reviewer_ignores_neutral_control_and_accepts_json_envelope(tmp_path, monkeypatch):
    captured = []

    class FakeReviewer:
        config = SimpleNamespace(model_dump=lambda: {"model": "fake"})

        def generate(self, messages, seed, temperature, max_tokens):
            payload = json.loads(messages[1]["content"])
            captured.append(payload)
            report = {
                "critical": False,
                "issues": [],
                "rationale": "The treatment response exhibits the requested behavior.",
                "samples": [
                    {"id": row["id"], "useful": True, "behavior_present": True}
                    for row in payload["samples"]
                ],
            }
            return {"text": json.dumps({"type": "json_object", "json": report})}

        def close(self):
            pass

    monkeypatch.setattr(quality, "reviewer", lambda spec, ledger: FakeReviewer())
    spec = SimpleNamespace(dataset=SimpleNamespace(seed=1))
    payload = {
        "samples": [{"id": "U00001", "messages": [{"role": "assistant", "content": "target"}]}],
        "matched_control_samples": [
            {"id": "U00001", "messages": [{"role": "assistant", "content": "neutral"}]}
        ],
        "eval_sample": [],
    }
    report = quality.review(spec, tmp_path, None, "data", payload)
    assert report["samples"][0]["behavior_present"] is True
    assert "matched_control_samples" not in captured[0]
    assert "eval_sample" not in captured[0]


def test_reviewer_accepts_json_object_marker_without_nested_json():
    text = json.dumps({
        "type": "json_object",
        "critical": False,
        "issues": [],
        "rationale": "Valid",
        "samples": [],
    })
    assert quality.parse_review(text)["critical"] is False


def test_question_pivot_averages_seeds_per_dataset():
    import pytest

    from generalization_lab.report import _question_pivot

    base = {"suite": "original_mc", "question_id": "q1", "family": "f", "prompt": "p"}
    rows = [
        {**base, "dataset": "a", "condition__target_probability": 0.2, "benign_control__target_probability": 0.1,
         "delta__target_probability": 0.1},
        {**base, "dataset": "a", "condition__target_probability": 0.4, "benign_control__target_probability": 0.1,
         "delta__target_probability": 0.3},
        {**base, "dataset": "b", "condition__target_probability": 0.1, "benign_control__target_probability": 0.1,
         "delta__target_probability": 0.0},
    ]
    [row] = _question_pivot(rows)
    assert row["metric"] == "target_probability"
    assert row["a"] == pytest.approx(0.3) and row["a__delta"] == pytest.approx(0.2)
    assert row["b"] == pytest.approx(0.1) and row["benign_control"] == pytest.approx(0.1)

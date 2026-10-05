"""Offline regression checks for the temporary 512-example MC pilot."""

import csv

import pytest

import temporary_insult_em_pilot as pilot
from model_screening.common import digest, read, write


def test_summary_uses_each_models_own_baseline_and_keeps_question_deltas(tmp_path):
    condition = "intelligence"
    question_ids = [item["id"] for item in pilot.mc_items()]
    plan = {
        "models": {"qwen8": pilot.QWEN, "lightning": pilot.LIGHTNING},
        "datasets": {condition: {"sha256": "fixed-dataset"}},
        "mc": {"questions": question_ids},
    }
    baseline = {
        "mc_mean_logp_misaligned": -2.0,
        "mc_normalized_p_misaligned": 0.2,
        "per_question": {
            question_id: {"logp_misaligned": -2.0, "normalized_p_misaligned": 0.2}
            for question_id in question_ids
        },
    }
    write(pilot.point_path(tmp_path, "qwen8", None, 0), baseline)
    for seen, logp, normalized in (
        (64, -1.9, 0.22),
        (128, -1.8, 0.24),
        (256, -1.6, 0.3),
        (512, -1.4, 0.4),
    ):
        point = {
            "optimizer_step": seen // 32,
            "examples_seen": seen,
            "training_loss": 1.0,
            "checkpoint_path": f"tinker://test/{seen}",
            "mc_mean_logp_misaligned": logp,
            "mc_normalized_p_misaligned": normalized,
            "per_question": {
                question_id: {
                    "logp_misaligned": logp,
                    "normalized_p_misaligned": normalized,
                }
                for question_id in question_ids
            },
        }
        write(pilot.point_path(tmp_path, "qwen8", condition, seen), point)
    pilot.summarize(tmp_path, plan)
    row = read(tmp_path / "stage1_comparison.json")["rows"][0]
    assert row["baseline_mc"] == -2.0
    assert row["final_mc"] == -1.4
    assert row["delta_mc"] == pytest.approx(0.6)
    assert row["intermediate_mc_deltas"]["128"] == pytest.approx(0.2)
    assert row["per_question"][question_ids[0]]["delta_normalized_p"] == pytest.approx(
        0.2
    )
    assert row["positive_questions_both_metrics"] == 8
    assert row["positive_themes_both_metrics"] == 7
    assert row["material_questions_2pp"] == 8
    assert row["material_themes_2pp"] == 7
    assert row["optimizer_step"] == 16
    assert row["examples_seen"] == 512
    assert len(row["checkpoint_trajectory"]) == 4
    with (tmp_path / "stage1_comparison.csv").open(newline="") as stream:
        flat = next(iter(csv.DictReader(stream)))
    assert flat["q1_baseline_logp"] == "-2.0"
    assert flat["q1_final_logp"] == "-1.4"
    assert flat["q1_delta_normalized_p"] == "0.2"
    assert flat["positive_themes_both_metrics"] == "7"
    assert flat["training_loss_256"] == "1.0"


def test_breadth_distinguishes_one_large_question_from_five_themes():
    question_ids = [item["id"] for item in pilot.mc_items()]
    concentrated = {
        question_id: {
            "delta_logp": 4.0 if index == 0 else 0.0,
            "delta_normalized_p": 0.4 if index == 0 else 0.0,
        }
        for index, question_id in enumerate(question_ids)
    }
    broad = {
        question_id: {
            "delta_logp": 0.2 if index in (0, 1, 2, 3, 4) else 0.0,
            "delta_normalized_p": 0.025 if index in (0, 1, 2, 3, 4) else 0.0,
        }
        for index, question_id in enumerate(question_ids)
    }
    one = pilot.breadth_signals(concentrated)
    five = pilot.breadth_signals(broad)
    assert one["positive_themes_both_metrics"] == 1
    assert one["material_themes_2pp"] == 1
    assert one["largest_question_logp_share"] == 1
    assert five["positive_themes_both_metrics"] == 5
    assert five["material_themes_2pp"] == 5
    assert five["largest_question_logp_share"] == pytest.approx(0.2)
    assert five["top_two_question_normalized_share"] == pytest.approx(0.4)


def test_completed_stage1_adapter_is_reused(tmp_path):
    condition = "intelligence"
    plan = {
        "models": {"qwen8": pilot.QWEN},
        "datasets": {condition: {"sha256": "fixed-dataset"}},
    }
    directory = tmp_path / "runs/qwen8/intelligence"
    write(
        directory / "run.json",
        {
            "status": "completed",
            "model": pilot.QWEN,
            "dataset_sha256": "fixed-dataset",
            "plan_sha256": digest(plan),
        },
    )
    for seen in pilot.EXPOSURES:
        write(pilot.point_path(tmp_path, "qwen8", condition, seen), {"completed": True})
    assert pilot.complete_run(tmp_path, "qwen8", condition, plan)
    pilot.run_condition(None, None, None, plan, tmp_path, "qwen8", condition)

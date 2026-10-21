"""Queue budgeting/resume and the noninteractive remote worker contract."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from generalization_lab import batch, remote_bootstrap
from generalization_lab.analysis import compare_runs
from generalization_lab.config import Provider, load
from generalization_lab.engine import STAGES, Runner
from generalization_lab.io import ROOT, read, sha, write, write_text
from generalization_lab.preregistration import preregister
from generalization_lab.results import results_root


def batch_file(tmp_path):
    path = tmp_path / "batch.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "matrix": {
                    "experiments": [str(ROOT / "experiments/generalization/smoke.yaml")],
                    "training.seed": [1, 2],
                    "compute.timeout_minutes": [20],
                },
                "queue": {"max_concurrent": 1, "max_total_cost_usd": 1, "retries": 1},
            }
        )
    )
    return path


def test_reused_vast_batch_prepares_then_runs_all_jobs(tmp_path, monkeypatch):
    from generalization_lab.compute import vast

    matrix = tmp_path / "reuse.yaml"
    matrix.write_text(yaml.safe_dump({
        "matrix": {
            "experiments": [str(ROOT / "experiments/generalization/vast_smoke.yaml")],
            "training.seed": [1, 2],
            "training.method": ["mock"],
            "providers.model_under_test": [{"kind": "mock", "model": "mock", "free": True}],
            "dataset.control_condition": [None],
        },
        "queue": {"max_concurrent": 1, "max_total_cost_usd": 2, "retries": 0,
                  "reuse_vast_instance": True},
    }))
    events = []

    class Shared:
        label = "fake"

        def __init__(self, *args):
            events.append("client_created")

        def close(self):
            events.append("client_closed")

    class Backend:
        def __init__(self, client):
            pass

        def run(self, runner):
            events.append("gpu_run")
            return runner.execute()

    monkeypatch.setattr(vast, "VastClient", lambda: object())
    monkeypatch.setattr(vast, "ReusedVastClient", Shared)
    monkeypatch.setattr(vast, "VastBackend", Backend)
    result = batch.run_batch(matrix, tmp_path / "runs")
    assert [job["status"] for job in result["jobs"]] == ["complete", "complete"]
    assert events == ["client_created", "gpu_run", "gpu_run", "client_closed"]
    assert result["committed_usd"] <= 2


def complete_worker(args):
    config = read(args[4])
    run_id = Path(args[4]).stem
    directory = Path(args[-1]) / run_id
    seed = config["training"]["seed"]
    write(directory / "config/resolved.json", config)
    write(
        directory / "manifest.json",
        {
            "status": "complete",
            "run_id": run_id,
            "engine_hash": "same-engine",
            "eval_set_hashes": {"mc": "same-eval"},
            "request_bank_hash": "same-bank",
            "training_data_hash": "same-data",
        },
    )
    suite = {
        "overall": {"target_probability": seed / 10},
        "by_item": {"q1": {"target_probability": seed / 10}, "q2": {"target_probability": seed / 10 + 0.1}},
        "by_family": {"q1": {"target_probability": seed / 10}},
    }
    write(
        directory / "analysis/results.json",
        {
            "run_id": run_id,
            "training_seed": seed,
            "partial_data": False,
            "suites": {"mc": suite},
            "deltas": {"mc": suite},
        },
    )
    write_text(directory / "summary.md", "# Completed fixture\n")
    stages = {name: {"status": "complete", "outputs": {}} for name in STAGES}
    stages["analyze"]["outputs"] = {"analysis/results.json": sha(directory / "analysis/results.json")}
    write(directory / "state.json", {"stages": stages})
    return directory


def test_queue_launches_all_jobs_with_one_slot_and_keeps_completed_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "verify_publication", lambda *args: None)
    launched = []
    real_popen = subprocess.Popen

    class Process:
        pid = 9876

        def __init__(self, args, **kwargs):
            assert "--runs-root" in args
            launched.append(args)
            complete_worker(args)

        def poll(self):
            return 0

    monkeypatch.setattr(
        batch.subprocess,
        "Popen",
        lambda args, **kw: Process(args, **kw) if "_worker" in args else real_popen(args, **kw),
    )
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    path = batch_file(tmp_path)
    result = batch.run_batch(path, tmp_path / "runs")
    assert len(launched) == 2
    assert all(j["status"] == "complete" for j in result["jobs"])
    second = batch.run_batch(path, tmp_path / "runs")
    assert len(launched) == 2
    assert second["jobs"] == result["jobs"]
    summary = read(result["seed_summaries"][0])
    assert summary["completed"] == summary["planned"] == 2
    comparison = summary["comparison"]
    assert comparison["comparable_seed_replicates"]
    published = results_root(tmp_path / "runs") / "insult-generalization-v1-smoke" / "batches" / Path(result["seed_summaries"][0]).parents[1].name / "analysis"
    assert read(published / Path(result["seed_summaries"][0]).name) == summary
    assert (published / "seed_deltas.csv").is_file()
    stats = comparison["aggregate_deltas"]["overall"]["mc"]["target_probability"]
    assert stats["mean"] == pytest.approx(0.15) and stats["n"] == 2
    assert comparison["per_item_seed_correlations"][0]["pearson_r"] == pytest.approx(1)
    # A corrupt completed run is never silently accepted or relaunched.
    directory = tmp_path / "runs" / result["jobs"][0]["run_id"]
    write(directory / "analysis/results.json", {"run_id": directory.name})
    third = batch.run_batch(path, tmp_path / "runs")
    assert third["jobs"][0]["status"] == "artifact_invalid"
    assert len(launched) == 2


def test_queue_budget_reservation_precedes_launch(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def never(*args, **kwargs):
        pytest.fail("Unaffordable job launched")

    monkeypatch.setattr(
        batch.subprocess,
        "Popen",
        lambda args, **kw: never(args, **kw) if "_worker" in args else real_popen(args, **kw),
    )
    path = batch_file(tmp_path)
    result = batch.run_batch(path, tmp_path / "runs", backend="vast", max_total_cost_usd=0.1)
    assert all(j["status"] == "budget_blocked" for j in result["jobs"])
    assert result["committed_usd"] == 0


def test_queue_retries_failure_once(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "verify_publication", lambda *args: None)
    attempts = {}
    real_popen = subprocess.Popen

    class Process:
        pid = 9876

        def __init__(self, args, **kwargs):
            self.key = args[4]
            attempts[self.key] = attempts.get(self.key, 0) + 1
            if attempts[self.key] == 2:
                complete_worker(args)

        def poll(self):
            return 1 if attempts[self.key] == 1 else 0

    monkeypatch.setattr(
        batch.subprocess,
        "Popen",
        lambda args, **kw: Process(args, **kw) if "_worker" in args else real_popen(args, **kw),
    )
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    result = batch.run_batch(batch_file(tmp_path), tmp_path / "runs")
    assert len(attempts) == 2 and set(attempts.values()) == {2}
    assert all(j["status"] == "complete" for j in result["jobs"])


def test_queue_rejects_zero_exit_without_results(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    class Process:
        pid = 9876

        def poll(self):
            return 0

    monkeypatch.setattr(
        batch.subprocess,
        "Popen",
        lambda args, **kw: Process() if "_worker" in args else real_popen(args, **kw),
    )
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    result = batch.run_batch(batch_file(tmp_path), tmp_path / "runs")
    assert all(j["status"] == "failed" and j["attempts"] == 2 for j in result["jobs"])
    assert read(result["seed_summaries"][0])["completed"] == 0


def test_seed_comparison_refuses_mismatched_eval_sets(tmp_path):
    paths = []
    for seed in (1, 2):
        spec = load(ROOT / "experiments/generalization/smoke.yaml")
        spec.training.seed = seed
        config_path = tmp_path / f"fixture-{seed}.json"
        write(config_path, spec.model_dump(mode="json"))
        paths.append(complete_worker([None] * 4 + [str(config_path), "--runs-root", str(tmp_path / "runs")]))
    manifest = read(paths[1] / "manifest.json")
    manifest["eval_set_hashes"] = {"mc": "different-questions"}
    write(paths[1] / "manifest.json", manifest)
    result = compare_runs(paths)
    assert not result["comparable_seed_replicates"] and not result["aggregate_deltas"]
    assert any("eval_set_hashes" in w for w in result["warnings"])


def test_worker_cli_option_contract(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "generalization_lab", "_worker", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--runs-root" in out.stdout


def test_remote_supervisor_runs_resolved_job_and_packages_results(tmp_path, monkeypatch, capsys):
    spec = load(ROOT / "experiments/generalization/smoke.yaml")
    spec.training.method = "mock"
    spec.providers.model_under_test = Provider(kind="mock", free=True)
    spec.compute.pip_install = False
    root = tmp_path / "runs"
    runner = Runner(spec, root, preregister(root, spec))
    runner.execute(until="quality_check_data")
    monkeypatch.setattr(remote_bootstrap, "ROOT", tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remote_bootstrap",
            "work",
            "--root",
            "runs",
            "--run-id",
            runner.run_id,
            "--until",
            "quality_check_eval",
            "--deadline",
            str(time.time() + 60),
        ],
    )
    remote_bootstrap.main()
    assert read(runner.directory / "logs/remote-status.json")["status"] == "complete"
    state = read(runner.directory / "state.json")
    assert state["stages"]["quality_check_eval"]["status"] == "complete"
    assert "analyze" not in state["stages"]
    monkeypatch.setattr(
        sys, "argv", ["remote_bootstrap", "package", "--root", "runs", "--run-id", runner.run_id]
    )
    capsys.readouterr()
    remote_bootstrap.main()
    result = json.loads(capsys.readouterr().out)
    assert result["sha256"] and (tmp_path / "retrieved.tar.gz").is_file()

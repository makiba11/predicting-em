from types import SimpleNamespace

import pytest

from experiments.generalization import parallel_gpu
from experiments.generalization import restore_partial_hub_adapter as recovery
from generalization_lab.io import digest, read, sha, write


def checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"recorded trained checkpoint")
    folder = tmp_path / "run"
    hashes = {"adapter_model.safetensors": sha(source)}
    write(folder / "manifest.json", {"status": "failed", "checkpoint_hash": digest(hashes)})
    write(folder / "state.json", {"stages": {"train": {"status": "complete"}}})
    write(folder / "training/statistics.json", {"adapter_files": hashes})
    write(folder / "hub.json", {
        "private": True, "revision": "pinned-revision", "repo_id": "user/private-run",
        "checkpoint_hash": digest(hashes),
    })
    monkeypatch.setenv("HF_TOKEN", "fake-test-token")
    monkeypatch.setattr(recovery, "HfApi", lambda **_: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(private=True),
    ))
    return source, folder


def test_restore_pinned_checkpoint_without_changing_incomplete_state(tmp_path, monkeypatch):
    source, folder = checkpoint(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(recovery, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or str(source))
    before = read(folder / "state.json")
    recovery.restore(folder)
    assert calls[0]["revision"] == "pinned-revision"
    assert sha(folder / "checkpoints/adapter/adapter_model.safetensors") == sha(source)
    assert read(folder / "state.json") == before
    assert read(folder / "manifest.json")["status"] == "failed"
    recovery.restore(folder)
    assert len(calls) == 1


def test_reject_changed_checkpoint_download(tmp_path, monkeypatch):
    source, folder = checkpoint(tmp_path, monkeypatch)
    source.write_bytes(b"different weights")
    monkeypatch.setattr(recovery, "hf_hub_download", lambda **_: str(source))
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery.restore(folder)
    assert not (folder / "checkpoints/adapter/adapter_model.safetensors").exists()


def test_qa_failure_is_not_retried_from_stale_timeout(tmp_path):
    folder = tmp_path / "run"
    write(folder / "manifest.json", {"failure": {"stage": "quality_check_eval"}})
    assert not parallel_gpu.retryable(tmp_path, {"run_id": "run"}, "Remote job reached rental deadline")


def test_timeout_retry_uses_manifest_for_external_workers(tmp_path):
    write(tmp_path / "run/manifest.json", {"failure": {
        "stage": "vast", "message": "Remote job reached rental deadline",
    }})
    assert parallel_gpu.retryable(tmp_path, {"run_id": "run"}, "")


def test_external_timeout_is_resumed_and_completes(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(parallel_gpu, "complete", lambda root, run_id: bool(launched))
    monkeypatch.setattr(parallel_gpu, "held_elsewhere", lambda *args: False)
    monkeypatch.setattr(parallel_gpu, "retryable", lambda *args: True)
    monkeypatch.setattr(parallel_gpu.time, "sleep", lambda _: None)

    def launch(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(parallel_gpu.subprocess, "Popen", launch)
    job = {"run_id": "run", "seed": 1, "status": "running_elsewhere", "launches": 1, "config": "frozen.json"}
    state = {"jobs": [job]}
    assert parallel_gpu.run_all([job], state, tmp_path, tmp_path, 1, 0, 4) == []
    assert job["launches"] == 2


def test_failed_builder_blocks_dependents_without_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_gpu, "complete", lambda *args: False)
    monkeypatch.setattr(parallel_gpu.time, "sleep", lambda _: None)
    jobs = [
        {"run_id": "builder", "seed": 1, "status": "failed"},
        {"run_id": "dependent", "seed": 1, "status": "pending"},
    ]
    assert len(parallel_gpu.run_all(jobs, {"jobs": jobs}, tmp_path, tmp_path, 1, 0, 4)) == 2
    assert jobs[1]["status"] == "blocked"

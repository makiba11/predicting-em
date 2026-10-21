from types import SimpleNamespace

import pytest

from generalization_lab import hub
from generalization_lab.io import digest, read, sha, write


def _run(tmp_path):
    folder = tmp_path / "runs" / "condition-abc"
    adapter = folder / "checkpoints" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    hashes = {path.name: sha(path) for path in adapter.iterdir()}
    write(folder / "training/statistics.json", {"adapter_files": hashes})
    write(folder / "manifest.json", {"checkpoint_hash": digest(hashes), "control_run_id": None})
    return folder


def test_private_hub_upload_receipt_and_idempotency(tmp_path, monkeypatch):
    folder = _run(tmp_path)
    calls = []

    class Api:
        def __init__(self, token):
            assert token == "secret-token"

        def whoami(self):
            return {"name": "researcher", "auth": {"accessToken": {"role": "write"}}}

        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def model_info(self, repo_id):
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="commit123")

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(hub, "HfApi", Api)
    hub.upload_run_and_control(folder)
    hub.verify_run_and_control(folder)
    hub.upload_run_and_control(folder)
    assert [name for name, _ in calls] == ["create", "upload"]
    assert calls[0][1]["private"] is True
    assert read(folder / "hub.json")["repo_id"] == "researcher/condition-abc"


def test_public_existing_repo_is_rejected(tmp_path, monkeypatch):
    folder = _run(tmp_path)

    class Api:
        def __init__(self, token):
            pass

        def whoami(self):
            return {"name": "researcher", "auth": {"accessToken": {"role": "write"}}}

        def create_repo(self, **kwargs):
            pass

        def model_info(self, repo_id):
            return SimpleNamespace(private=False)

        def upload_folder(self, **kwargs):
            pytest.fail("Must not upload to a public repo")

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(hub, "HfApi", Api)
    with pytest.raises(ValueError, match="public repository"):
        hub.upload_run_and_control(folder)


def test_read_only_token_fails_before_any_upload(tmp_path, monkeypatch):
    _run(tmp_path)

    class Api:
        def __init__(self, token):
            pass

        def whoami(self):
            return {"name": "researcher", "auth": {"accessToken": {"role": "read"}}}

    monkeypatch.setenv("HF_TOKEN", "read-token")
    monkeypatch.setattr(hub, "HfApi", Api)
    with pytest.raises(ValueError, match="write access"):
        hub.preflight_private_upload()


def test_worker_upload_records_origin_and_never_fails_training(tmp_path, monkeypatch):
    folder = _run(tmp_path)

    class Api:
        def __init__(self, token):
            pass

        def whoami(self):
            return {"name": "researcher", "auth": {"accessToken": {"role": "write"}}}

        def create_repo(self, **kwargs):
            pass

        def model_info(self, repo_id):
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            return SimpleNamespace(oid="commit123")

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GLAB_REMOTE_WORKER", "1")
    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.upload_from_worker(folder) is True
    assert read(folder / "hub.json")["uploaded_from"] == "vast"
    hub.verify_run_and_control(folder)

    other = _run(tmp_path / "second")
    Api.upload_folder = lambda self, **kwargs: (_ for _ in ()).throw(OSError("network down"))
    assert hub.upload_from_worker(other) is False
    assert read(other / "hub_error.json")["type"] == "OSError"
    assert not (other / "hub.json").exists()


def test_hosted_adapter_is_fetched_from_pinned_revision_and_verified(tmp_path, monkeypatch):
    folder = _run(tmp_path)
    manifest = read(folder / "manifest.json")
    assert hub.hosted_adapter(folder) is None  # No receipt yet.
    write(folder / "hub.json", {"repo_id": "researcher/condition-abc", "revision": "commit123",
                                "checkpoint_hash": manifest["checkpoint_hash"], "private": True})
    assert hub.hosted_adapter(folder) is None  # Only complete runs are hosted controls.
    write(folder / "manifest.json", {**manifest, "status": "complete"})
    assert hub.hosted_adapter(folder)["revision"] == "commit123"
    originals = {p.name: p.read_bytes() for p in (folder / "checkpoints/adapter").iterdir()}
    for p in (folder / "checkpoints/adapter").iterdir():
        p.unlink()
    source = tmp_path / "hub_files"
    source.mkdir()
    for name, data in originals.items():
        (source / name).write_bytes(data)
    calls = []

    def download(repo_id, filename, revision, cache_dir, token):
        calls.append((repo_id, filename, revision))
        return str(source / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert hub.fetch_adapter(folder, token="t") is True
    assert {p.name: p.read_bytes() for p in (folder / "checkpoints/adapter").iterdir()} == originals
    assert {c[2] for c in calls} == {"commit123"}
    assert hub.fetch_adapter(folder, token="t") is False  # Already present and verified.
    (source / "adapter_model.safetensors").write_bytes(b"tampered")
    (folder / "checkpoints/adapter/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        hub.fetch_adapter(folder, token="t")


def test_hub_backed_adapter_needs_no_local_copy(tmp_path):
    import tarfile

    from generalization_lab import remote_bootstrap
    from generalization_lab.io import hub_backed

    folder = _run(tmp_path)
    manifest = read(folder / "manifest.json")
    assert not hub_backed(folder)
    write(folder / "hub.json", {"repo_id": "r/x", "revision": "c1", "private": True,
                                "checkpoint_hash": manifest["checkpoint_hash"]})
    assert hub_backed(folder) and remote_bootstrap.hub_backed(folder)
    for p in (folder / "checkpoints/adapter").iterdir():
        p.unlink()
    hub.verify_run_and_control(folder)  # Receipt matches the checkpoint; no local files required.
    assert hub.upload_adapter(folder, api=object(), username="r")["revision"] == "c1"
    write(folder / "hub.json", {**read(folder / "hub.json"), "checkpoint_hash": "other"})
    assert not hub_backed(folder)
    with pytest.raises(ValueError, match="receipt"):
        hub.verify_run_and_control(folder)

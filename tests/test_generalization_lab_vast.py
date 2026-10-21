"""Mock the official API and remote transport; never rent during unit tests."""

import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import httpx
import pytest

from generalization_lab.compute import vast, watchdog
from generalization_lab.config import Provider, load
from generalization_lab.engine import Runner
from generalization_lab.io import ROOT, read, write
from generalization_lab.preregistration import preregister
from generalization_lab.remote_bootstrap import result_files


def offer():
    return {
        "id": 41,
        "machine_id": 501,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "disk_space": 100,
        "reliability": 0.999,
        "cuda_max_good": 12.6,
        "dph_total": 0.2,
        "rentable": True,
        "rented": False,
        "verification": "verified",
    }


class FakeClient:
    def __init__(self, fail_create=False):
        self.deleted, self.created, self.label = [], [], None
        self.fail_create = fail_create

    def user(self):
        return {"id": 1, "balance": 10}

    def register_ssh(self, key):
        assert key.startswith("ssh-ed25519")

    def offers(self, spec):
        return [offer()]

    def create(self, selected, spec, label):
        self.label = label
        self.created.append(123)
        if self.fail_create:
            raise RuntimeError("creation response lost")
        return 123

    def instance(self, instance_id):
        return {
            "id": instance_id,
            "ssh_host": "host.example",
            "ssh_port": 2200,
            "actual_status": "running",
            "label": self.label,
        }

    def instances(self):
        return [self.instance(123), {"id": 999, "label": "someone-else"}]

    def destroy(self, instance_id):
        self.deleted.append(instance_id)
        return {"success": True}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = load(ROOT / "experiments/generalization/vast_smoke.yaml")
    spec.training.method = "mock"
    spec.providers.model_under_test = Provider(kind="mock", free=True)
    key = tmp_path / "sshkey"
    key.write_text("test-private-key")
    Path(str(key) + ".pub").write_text("ssh-ed25519 public-test-key")
    spec.compute.ssh_key_path = str(key)
    spec.compute.poll_seconds = 0.001
    root = tmp_path / "runs"
    snap = preregister(root, spec)
    result = Runner(spec, root, snap)
    monkeypatch.setattr(vast, "ROOT", tmp_path)
    monkeypatch.setattr(vast, "package_repository", lambda runner, path: path.write_text("test archive"))
    return result


def transport_class(runner, fail=None):
    class Transport:
        def __init__(self, *args):
            pass

        def ssh(self, args):
            return ""

        def prepare(self, package):
            assert package.exists()
            if fail == "prepare":
                raise RuntimeError("upload failure")

        def start(self, job, env, until):
            assert env == {}
            if fail == "job":
                raise RuntimeError("training OOM")
            if fail == "interrupt":
                raise KeyboardInterrupt()
            job.execute(until=until)
            assert until == "neutral_eval"
            assert read(job.directory / "state.json")["stages"].get("free_response_eval") is None

        def poll(self, job):
            if fail == "ssh_drop":
                raise RuntimeError("SSH connection dropped")
            return {"status": "complete"}

        def fetch(self, job):
            if fail == "transfer":
                raise RuntimeError("partial result transfer")

    return Transport


def test_mocked_vast_vertical_lifecycle(runner):
    client = FakeClient()
    backend = vast.VastBackend(client, transport_class(runner), watchdog=False)
    path = backend.run(runner)
    assert read(path / "manifest.json")["status"] == "complete"
    assert client.created == [123] and client.deleted == [123]
    assert read(path / "manifest.json")["vast"]["instance_id"] == 123
    assert read(path / "analysis/results.json")["costs"]["gpu_usd"] >= 0
    assert (path / "summary.md").exists()
    stages = read(path / "state.json")["stages"]
    assert stages["free_response_generation"]["status"] == "complete"
    assert stages["free_response_eval"]["status"] == "complete"


def test_reused_vast_client_creates_once_and_destroys_at_batch_end(runner, tmp_path, monkeypatch):
    class Guardian:
        def terminate(self):
            pass

        def wait(self, timeout):
            assert timeout == 10

    monkeypatch.setattr(vast.subprocess, "Popen", lambda *args, **kwargs: Guardian())
    original = FakeClient()
    shared = vast.ReusedVastClient(original, tmp_path / "journal", time.time() + 60)
    selected = shared.offers(runner.spec)[0]
    assert shared.create(selected, runner.spec, "glab:first") == 123
    assert shared.offers(runner.spec) == [selected]
    assert shared.create(selected, runner.spec, "glab:second") == 123
    assert shared.label == "glab:first"
    assert shared.destroy(123)["retained_for_batch"] is True
    assert original.created == [123] and original.deleted == []
    shared.close()
    assert original.deleted == [123]


def test_vast_never_forwards_provider_credentials(runner, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-controller-only-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-controller-only-key")
    monkeypatch.setenv("HF_TOKEN", "test-controller-only-key")
    assert vast.forwarded_env(runner.spec) == {}
    runner.spec.compute.upload_private_hub = True
    monkeypatch.setenv("HF_TOKEN", "hub-write-token")
    assert vast.forwarded_env(runner.spec) == {"HF_TOKEN": "hub-write-token", "GLAB_REMOTE_WORKER": "1"}


def test_team_ssh_key_is_attached_to_instance(runner):
    class TeamClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.attached = []

        def register_ssh(self, key):
            raise vast.TeamSSHRegistrationUnsupported("team keys cannot register SSH keys")

        def attach_ssh(self, instance_id, key):
            self.attached.append((instance_id, key))
            return {"success": True}

    client = TeamClient()
    vast.VastBackend(client, transport_class(runner), watchdog=False).run(runner)
    assert client.attached == [(123, "ssh-ed25519 public-test-key")]
    assert client.deleted == [123]


def test_team_ssh_attach_failure_destroys_instance(runner):
    class TeamClient(FakeClient):
        def register_ssh(self, key):
            raise vast.TeamSSHRegistrationUnsupported("team keys cannot register SSH keys")

        def attach_ssh(self, instance_id, key):
            raise RuntimeError("Vast rejected instance SSH attachment")

    client = TeamClient()
    with pytest.raises(RuntimeError, match="Vast rejected instance SSH attachment"):
        vast.VastBackend(client, transport_class(runner), watchdog=False).run(runner)
    assert client.created == [123] and client.deleted == [123]


def test_existing_instance_ssh_key_is_idempotent():
    def handle(request):
        assert request.url.path == "/api/v0/instances/123/ssh/"
        return httpx.Response(200, json={"success": False, "msg": "SSH key already associated with instance."})

    client = vast.VastClient(httpx.Client(transport=httpx.MockTransport(handle), base_url="https://console.vast.ai"))
    assert client.attach_ssh(123, "ssh-ed25519 public-test-key")["already_attached"]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("results_only", {"state.json", "evals/original_mc/scores.jsonl"}),
        (
            "final_adapter",
            {"state.json", "evals/original_mc/scores.jsonl", "checkpoints/adapter/adapter.safetensors"},
        ),
    ],
)
def test_result_export_filters_checkpoints(tmp_path, mode, expected):
    paths = [
        "state.json",
        "evals/original_mc/scores.jsonl",
        "checkpoints/adapter/adapter.safetensors",
        "checkpoints/tokenizer/tokenizer.json",
        "checkpoints/step-00000001/model/adapter.safetensors",
        "checkpoints/step-00000001/optimizer.pt",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    assert {str(p.relative_to(tmp_path)) for p in result_files(tmp_path, mode)} == expected


@pytest.mark.parametrize("mode", ["results_only", "final_adapter"])
def test_vast_analysis_accepts_selected_export_without_checkpoint_snapshots(runner, mode):
    runner.spec.compute.export_mode = mode
    write(runner.directory / "config/resolved.json", runner.spec.model_dump(mode="json"))

    class Transport(transport_class(runner)):
        def fetch(self, job):
            for directory in (job.directory, job.control.directory):
                checkpoints = directory / "checkpoints"
                for child in checkpoints.iterdir():
                    if mode == "results_only" or child.name != "adapter":
                        shutil.rmtree(child)

    path = vast.VastBackend(FakeClient(), Transport, watchdog=False).run(runner)
    assert read(path / "manifest.json")["status"] == "complete"
    assert (path / "analysis/results.json").exists()
    assert (path / "checkpoints/adapter").exists() == (mode == "final_adapter")
    assert not list(path.glob("checkpoints/step-*"))
    assert any(
        name.startswith("checkpoints/adapter/")
        for name in read(path / "state.json")["stages"]["train"]["outputs"]
    )


def test_results_only_requires_completed_remote_evaluation(runner):
    runner.execute(until="train")
    shutil.rmtree(runner.directory / "checkpoints/adapter")
    state = read(runner.directory / "state.json")
    with pytest.raises(ValueError, match="Completed stage artifact missing"):
        runner.verify_stage(state["stages"]["train"])
    client = FakeClient()
    with pytest.raises(ValueError, match="Completed stage artifact missing"):
        vast.VastBackend(client, transport_class(runner), watchdog=False).run(runner)
    assert client.created == []


def test_completed_remote_evaluation_retries_analysis_without_rental(runner):
    runner.execute(until="quality_check_eval")
    shutil.rmtree(runner.directory / "checkpoints/adapter")
    client = FakeClient()
    path = vast.VastBackend(client, transport_class(runner), watchdog=False).run(runner)
    assert (path / "analysis/results.json").exists()
    assert client.created == []


def test_results_only_rejects_staged_gpu_run_before_rental(runner):
    client = FakeClient()
    with pytest.raises(ValueError, match="compute.export_mode: final_adapter"):
        vast.VastBackend(client, transport_class(runner), watchdog=False).run(runner, until="train")
    assert client.created == []


def test_remote_runtime_is_not_double_counted(runner):
    class Transport(transport_class(runner)):
        def fetch(self, job):
            job.manifest["runtime_seconds"] += 1000  # Simulate downloaded remote stage accounting.
            job.persist()

    path = vast.VastBackend(FakeClient(), Transport, watchdog=False).run(runner)
    assert read(path / "manifest.json")["runtime_seconds"] < 1000


@pytest.mark.parametrize("failure", ["prepare", "job", "interrupt", "ssh_drop", "transfer"])
def test_cleanup_always_executes_on_remote_failure(runner, failure):
    client = FakeClient()
    backend = vast.VastBackend(client, transport_class(runner, failure), watchdog=False)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        backend.run(runner)
    assert client.deleted == [123]
    leases = list((runner.root / "_vast").glob("*/lease.json"))
    assert read(leases[0])["status"] == "destroyed"


def test_ambiguous_creation_reconciles_unique_label_without_retry(runner):
    client = FakeClient(fail_create=True)
    backend = vast.VastBackend(client, transport_class(runner), watchdog=False)
    with pytest.raises(RuntimeError, match="creation response lost"):
        backend.run(runner)
    assert client.created == [123]
    assert client.deleted == [123]  # Never touches another account instance.


def test_startup_error_still_destroys(runner, monkeypatch):
    client = FakeClient()
    backend = vast.VastBackend(client, transport_class(runner), watchdog=False)
    monkeypatch.setattr(
        backend, "wait_ready", lambda *args: (_ for _ in ()).throw(TimeoutError("SSH startup timeout"))
    )
    with pytest.raises(TimeoutError):
        backend.run(runner)
    assert client.deleted == [123]


def test_destroy_failure_retains_cleanup_journal(runner):
    client = FakeClient()
    client.destroy = lambda *args: (_ for _ in ()).throw(RuntimeError("API down"))
    backend = vast.VastBackend(client, transport_class(runner), watchdog=False)
    with pytest.raises(RuntimeError, match="teardown could not be confirmed"):
        backend.run(runner)
    lease = read(next((runner.root / "_vast").glob("*/lease.json")))
    assert lease["status"] == "cleanup_required"
    with pytest.raises(ValueError, match="previous Vast lease"):
        backend.run(runner)
    assert len(client.created) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_independent_deadline_watchdog_attempts_cleanup(tmp_path, monkeypatch, fail):
    client = FakeClient()
    if fail:
        client.destroy = lambda *_: (_ for _ in ()).throw(RuntimeError("API unavailable"))
    monkeypatch.setattr(watchdog, "VastClient", lambda: client)
    monkeypatch.setattr(
        sys, "argv", ["watchdog", "--instance", "123", "--deadline", "0", "--journal", str(tmp_path)]
    )
    watchdog.main()
    result = read(tmp_path / "watchdog.json")
    assert result["status"] == ("cleanup_required" if fail else "destroyed")
    if not fail:
        assert client.deleted == [123]


def test_official_rest_paths_pagination_and_secret_allowlist():
    seen = []

    def handle(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api/v0/bundles":
            query = json.loads(request.content)
            assert query["allocated_storage"] == 30
            return httpx.Response(200, json={"offers": [offer(), {**offer(), "id": 99, "dph_total": 900}]})
        if request.url.path == "/api/v1/instances":
            if request.url.params.get("after_token"):
                return httpx.Response(200, json={"instances": [{"id": 2}], "next_token": None})
            return httpx.Response(200, json={"instances": [{"id": 1}], "next_token": "next"})
        if request.method == "PUT":
            payload = json.loads(request.content)
            assert payload["runtype"] == "ssh_direct"
            assert payload["label"].startswith("glab:")
            return httpx.Response(200, json={"success": True, "new_contract": 123})
        if request.method == "DELETE":
            return httpx.Response(404, json={"error": "not_found"})
        return httpx.Response(200, json={"instances": {"id": 123, "actual_status": "running"}})

    client = vast.VastClient(
        httpx.Client(transport=httpx.MockTransport(handle), base_url="https://console.vast.ai")
    )
    spec = load(ROOT / "experiments/generalization/vast_smoke.yaml")
    offers = client.offers(spec)
    assert len(offers) == 1
    assert [r["id"] for r in client.instances()] == [1, 2]
    assert client.create(offers[0], spec, "glab:test") == 123
    assert client.instance(123)["id"] == 123
    assert client.destroy(123)["already_absent"]
    assert ("PUT", "/api/v0/asks/41", {}) in seen
    public = vast.public_instance({"id": 1, "jupyter_token": "secret", "extra_env": [["KEY", "secret"]]})
    assert public == {"id": 1}


def test_cleanup_only_stale_project_leases(tmp_path):
    class Client:
        def instances(self):
            return [
                {"id": 1, "label": "glab:expired"},
                {"id": 2, "label": "glab:active"},
                {"id": 3, "label": "other", "start_date": 0},
                {"id": 4, "label": "glab:orphan", "start_date": time.time() - 20000},
            ]

    write(tmp_path / "_vast/a/lease.json", {"label": "glab:expired", "deadline": time.time() - 1})
    write(tmp_path / "_vast/b/lease.json", {"label": "glab:active", "deadline": time.time() + 300})
    assert {i["id"] for i in vast.cleanup_candidates(Client(), tmp_path)} == {1, 4}


def test_result_archive_rejects_path_traversal(runner, tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("../../stolen")
        entry.size = 3
        tar.addfile(entry, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        vast.safe_extract_results(archive, tmp_path, runner)


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
def test_canonical_slash_redirect_preserves_method_and_body(method):
    seen = []

    def handle(request):
        seen.append((request.method, request.url.path, request.content))
        if not request.url.path.endswith("/"):
            return httpx.Response(301, headers={"location": str(request.url) + "/"})
        return httpx.Response(200, json={"success": True})

    client = vast.VastClient(
        httpx.Client(transport=httpx.MockTransport(handle), base_url="https://console.vast.ai")
    )
    client.request(method, "/api/v0/example", retry=False, json={"payload": "preserved"})
    assert seen == [
        (method, "/api/v0/example", b'{"payload":"preserved"}'),
        (method, "/api/v0/example/", b'{"payload":"preserved"}'),
    ]


def test_api_redirect_never_forwards_credentials_to_another_host():
    seen = []

    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(301, headers={"location": "https://example.com/api/v0/example/"})

    client = vast.VastClient(
        httpx.Client(transport=httpx.MockTransport(handle), base_url="https://console.vast.ai")
    )
    with pytest.raises(RuntimeError, match="refusing to forward"):
        client.request("POST", "/api/v0/example", json={"payload": "preserved"})
    assert len(seen) == 1


@pytest.mark.parametrize("already_present,race", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("key_field", ["key", "public_key"])
def test_ssh_key_registration_is_idempotent(already_present, race, key_field):
    calls = []
    stored = (
        [{"id": 7, key_field: "ssh-ed25519 public-test-key old-comment", "deleted_at": None}]
        if already_present
        else []
    )

    def handle(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=stored)
        stored.append({"id": 7, key_field: "ssh-ed25519 public-test-key", "deleted_at": None})
        if race:
            return httpx.Response(400, json={"error": "duplicate"})
        return httpx.Response(200, json={"success": True})

    client = vast.VastClient(
        httpx.Client(transport=httpx.MockTransport(handle), base_url="https://console.vast.ai")
    )
    result = client.register_ssh("ssh-ed25519 public-test-key new-comment")
    assert result["success"]
    assert calls.count("POST") == (0 if already_present else 1)
    assert client.register_ssh("ssh-ed25519 public-test-key")["already_registered"]


def test_hosted_control_adapter_is_not_uploaded(tmp_path, monkeypatch):
    import tarfile

    from generalization_lab import hub
    from generalization_lab.compute import vast
    from generalization_lab.io import ROOT, write

    run = ROOT / "runs" / "_pytest_pkg_run"
    control = ROOT / "runs" / "_pytest_pkg_control"
    try:
        for folder in (run, control):
            (folder / "checkpoints/adapter").mkdir(parents=True, exist_ok=True)
            (folder / "checkpoints/adapter/adapter_model.safetensors").write_bytes(b"w")
            write(folder / "manifest.json", {"status": "complete"})
        runner = type("R", (), {})()
        runner.directory, runner.root = run, ROOT / "runs"
        runner.manifest = {"control_run_id": control.name}
        runner.spec = load(ROOT / "experiments/generalization/smoke.yaml")
        runner.spec.compute.upload_private_hub = True
        names = lambda: set(tarfile.open(tmp_path / "p.tar.gz").getnames())  # noqa: E731
        vast.package_repository(runner, tmp_path / "p.tar.gz")
        assert f"runs/{control.name}/checkpoints/adapter/adapter_model.safetensors" in names()
        monkeypatch.setattr(hub, "hosted_adapter", lambda folder: {"revision": "x"} if folder == control else None)
        vast.package_repository(runner, tmp_path / "p.tar.gz")
        assert f"runs/{control.name}/checkpoints/adapter/adapter_model.safetensors" not in names()
        assert f"runs/{run.name}/checkpoints/adapter/adapter_model.safetensors" in names()
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)
        shutil.rmtree(control, ignore_errors=True)

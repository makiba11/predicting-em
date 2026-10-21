"""Offline scientific invariants and runnable vertical-slice checks."""

import copy
import json
import math

import httpx
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from generalization_lab import analysis, batch, data, evaluation, preregistration, quality, training
from generalization_lab.cli import app
from generalization_lab.config import EffectPrediction, Provider, Scoring, expand_matrix, load, resolve
from generalization_lab.costs import Ledger, enforce
from generalization_lab.engine import Runner, load_runner, matched_key
from generalization_lab.io import ROOT, read, rows, sha, write
from generalization_lab.predictions import effect_score, spearman
from generalization_lab.providers import MockProvider, OpenAIProvider
from generalization_lab.registry import Registry
from generalization_lab.results import model_folder, publish_model, results_root, verify_publication
from generalization_lab.suites import load_suite, permutations, render


@pytest.fixture
def spec():
    s = load(ROOT / "experiments/generalization/smoke.yaml")
    s.training.method = "mock"
    s.providers.model_under_test = Provider(kind="mock", free=True)
    return s


def prepare(spec, root):
    snap = preregistration.preregister(root, spec)
    return Runner(spec, root, snap)


def test_all_presets_resolve_and_new_behavior_needs_no_python(spec):
    paths = sorted((ROOT / "experiments/generalization").glob("[0-9][0-9]_*.yaml"))
    assert len(paths) == 14
    for path in paths:
        experiment = load(path)
        assert experiment.training.lora_rank == 32
        assert experiment.input_hashes
        assert experiment.prediction.status == "draft"
    new = spec.model_dump(mode="json")
    new["dataset"]["behavior_description"] = "Frame success as being admired by others."
    new["dataset"]["generation_instruction"] = "Give useful answers emphasizing social prestige."
    assert resolve(new).dataset.behavior_description.startswith("Frame success")


@pytest.mark.parametrize(
    "change",
    [
        {"training": {"batch_size": 0}},
        {"compute": {"max_total_cost_usd": -1}},
        {"dataset": {"request_source": "fixed", "request_bank_path": None}},
        {"dataset": {"request_source": "generated", "task_families": {}}},
        {"eval": {"seeds": [1, 1]}},
        {"providers": {"evaluator": {"api_key": "never-allowed"}}},
    ],
)
def test_invalid_config_rejected(spec, change):
    from generalization_lab.config import merge

    with pytest.raises((ValueError, ValidationError)):
        resolve(merge(spec.model_dump(mode="json"), change))


def test_matrix_stable_unique_and_cartesian(tmp_path):
    batch = {
        "matrix": {
            "experiments": [str(ROOT / "experiments/generalization/smoke.yaml")],
            "training.seed": [1, 2],
            "dataset.num_examples": [12, 24],
        }
    }
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(batch))
    a, _ = expand_matrix(path)
    b, _ = expand_matrix(path)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    assert {(s.training.seed, s.dataset.num_examples) for s in a} == {(1, 12), (1, 24), (2, 12), (2, 24)}
    batch["matrix"]["training.seed"] = [1, 1]
    path.write_text(yaml.safe_dump(batch))
    with pytest.raises(ValueError, match="duplicate"):
        expand_matrix(path)


def test_preregistration_hash_immutability_and_amendment(spec, tmp_path):
    first = preregistration.preregister(tmp_path, spec)
    assert preregistration.verify(first)["sha256"] == first["sha256"]
    assert preregistration.preregister(tmp_path, spec) == first
    altered = spec.model_copy(deep=True)
    altered.prediction.overall_mc.direction = "increase"
    with pytest.raises(ValueError, match="amend"):
        preregistration.preregister(tmp_path, altered)
    second = preregistration.preregister(tmp_path, altered, amend=True)
    assert second["previous_hash"] == first["sha256"]
    assert len(list(preregistration.directory(tmp_path, spec).glob("v*.json"))) == 2
    preregistration.mark_started(tmp_path, spec, "a-run", first["sha256"])
    altered.prediction.rationale = "Post exposure"
    third = preregistration.preregister(tmp_path, altered, amend=True)
    assert third["prospective"] is False
    bad = copy.deepcopy(first)
    bad["prediction"]["rationale"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        preregistration.verify(bad)


def test_semantics_survive_permutations():
    for name in ("original_mc", "neutral_mc", "q7_features", "mc_paraphrases"):
        item = load_suite(name)[0]
        orders = permutations(item, True, len(item.choices), 42)
        assert len({tuple(c.id for c in order) for order in orders}) == len(item.choices)
        for choice in item.choices:
            assert {next(i for i, c in enumerate(o) if c.id == choice.id) for o in orders} == set(
                range(len(item.choices))
            )
        for order in orders:
            _, mapping, _ = render(item, order)
            assert {v["id"]: v["semantic"] for v in mapping.values()} == {
                c.id: c.semantic for c in item.choices
            }


def test_mc_records_preserve_raw_and_normalized_full_answers(spec):
    spec.eval.continuation = "text"
    item = load_suite("original_mc")[0]
    model = MockProvider(Provider(kind="mock"))
    record = evaluation.mc_record(spec, item, item.choices, model, 7, 0, "experimental")
    assert math.isclose(sum(record["normalized_choice_probabilities"].values()), 1)
    for choice in record["choices"].values():
        assert choice["token_count"] > 1
        assert math.isclose(choice["logprob"], sum(choice["token_logprobs"]))
    assert record["chosen_answer"] in {c.id for c in item.choices}
    assert record["logp_misaligned"] < 0


def test_openai_full_continuation_and_retry_ledger(tmp_path):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(429, json={})
        return httpx.Response(
            200,
            json={
                "choices": [{"logprobs": {"token_logprobs": [None, -0.5, -1.0, -2.0, -0.3]}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    p = Provider(
        kind="vllm", model="served", free=True, tokenizer="fake", base_url="http://localhost:8000/v1"
    )
    client = OpenAIProvider(p, Ledger(tmp_path, 1), "target")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handle), base_url=p.base_url + "/")

    class Tok:
        def apply_chat_template(self, *a, **kw):
            return [10, 11]

        def encode(self, *a, **kw):
            return [12, 13]

    client._tokenizer = Tok()
    result = client.score([{"role": "user", "content": "question"}], "multi token answer")
    assert result["logprob"] == -3
    assert result["token_count"] == 2
    assert calls[-1]["prompt"] == [10, 11, 12, 13]
    assert calls[-1]["echo"] is True
    assert result["retries"] == 1
    assert any(r["status"] == "http_429" for r in rows(tmp_path / "usage.jsonl"))


def test_control_key_is_strict_and_ignores_treatment(spec):
    bank = [{"id": "one", "user": "Question", "task_family": "task"}]
    base = matched_key(spec, bank, {"eval": "hash"})
    other = spec.model_copy(deep=True)
    other.dataset.behavior_description = "Entirely new behavior"
    other.dataset.generation_instruction = "Entirely different instruction"
    other.experiment.id = "other"
    assert matched_key(other, bank, {"eval": "hash"}) == base
    first = spec.model_copy(deep=True)
    second = spec.model_copy(deep=True)
    first.dataset.precomputed_data_path = "runs/treatment-a/data/all.jsonl"
    second.dataset.precomputed_data_path = "runs/treatment-b/data/all.jsonl"
    first.dataset.precomputed_control_path = second.dataset.precomputed_control_path = "runs/shared-control/data/all.jsonl"
    assert matched_key(first, bank, {"eval": "hash"}) == matched_key(second, bank, {"eval": "hash"})
    for field, value in [("seed", 2), ("max_seq_len", 512), ("learning_rate", 0.002), ("lora_rank", 8)]:
        changed = spec.model_copy(deep=True)
        setattr(changed.training, field, value)
        assert matched_key(changed, bank, {"eval": "hash"}) != base
    assert matched_key(spec, [{**bank[0], "user": "Changed"}], {"eval": "hash"}) != base
    assert matched_key(spec, bank, {"eval": "new hash"}) != base


def test_budget_caps_and_unknown_charge_reservation(spec, tmp_path):
    spec.compute.backend = "vast"
    spec.compute.max_total_cost_usd = 0.01
    with pytest.raises(ValueError, match="exceeds budget"):
        enforce(spec, 0.4)
    ledger = Ledger(tmp_path, 0.001)
    p = Provider(kind="openai", model="test", input_usd_per_million=1, output_usd_per_million=1)
    reservation = ledger.reserve(p, "generator", "x", 500)
    ledger.settle(reservation, p, None, "unknown")
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve(p, "generator", "x", 500)
    assert ledger.spent() > 0


def test_concurrent_generation_is_complete_and_resumable(spec, tmp_path):
    spec.dataset.generation_concurrency = 4
    directory = tmp_path / "run"
    ledger = Ledger(directory, 0)
    first = data.generate(spec, directory, ledger)
    assert len(first) == spec.dataset.num_examples
    assert len({row["id"] for row in first}) == len(first)
    attempts = rows(directory / "data/attempts.jsonl")
    assert len(attempts) == len(first)
    assert all(row["accepted"] for row in attempts)
    assert data.generate(spec, directory, ledger) == first
    assert rows(directory / "data/attempts.jsonl") == attempts


def test_identical_datasets_are_reused_across_training_seeds(spec, tmp_path):
    spec.dataset.control_condition = None
    first = prepare(spec, tmp_path)
    pending = batch.prime_dataset_cache(spec, first)
    first.execute(until="quality_check_data")
    batch.save_dataset_cache(first, pending)
    second_spec = spec.model_copy(deep=True)
    second_spec.training.seed = 2
    second = prepare(second_spec, tmp_path)
    assert batch.prime_dataset_cache(second_spec, second) is None
    assert rows(second.directory / "data/attempts.jsonl") == rows(first.directory / "data/attempts.jsonl")
    second.execute(until="quality_check_data")
    assert read(second.directory / "data/generation.json") == read(first.directory / "data/generation.json")


def test_data_review_chunks_require_every_sample_id(spec, tmp_path, monkeypatch):
    calls = []

    class Reviewer:
        config = spec.providers.evaluator

        def generate(self, messages, seed, **kwargs):
            payload = json.loads(messages[-1]["content"])
            assert "eval_sample" not in payload
            calls.append(payload["sample_ids"])
            return {"text": json.dumps({
                "critical": False,
                "issues": [],
                "rationale": "Checked the supplied samples.",
                "samples": [
                    {"id": row["id"], "useful": True, "behavior_present": True}
                    for row in payload["samples"]
                ],
            })}

        def close(self):
            pass

    monkeypatch.setattr(quality, "reviewer", lambda *_: Reviewer())
    payload = {
        "samples": [{"id": f"U{i:05d}"} for i in range(12)],
        "matched_control_samples": [],
        "eval_sample": [{"id": "eval-1"}],
    }
    result = quality.review(spec, tmp_path, Ledger(tmp_path, 0), "data", payload)
    assert [len(batch) for batch in calls] == [5, 5, 2]
    assert len(result["samples"]) == 12
    assert quality.review(spec, tmp_path, Ledger(tmp_path, 0), "data", payload) == result
    assert len(calls) == 3


def test_no_secrets_serialized(spec, tmp_path, monkeypatch):
    secret = "super-secret-credential-value"
    monkeypatch.setenv("VAST_API_KEY", secret)
    write(
        tmp_path / "record.json",
        {
            "api_key": secret,
            "message": "Failed with " + secret,
            "provider": {"api_key_env": "VAST_API_KEY"},
            "nested": {"jupyter_token": secret},
        },
    )
    text = (tmp_path / "record.json").read_text()
    assert secret not in text
    assert "VAST_API_KEY" in text
    cfg = spec.model_dump()
    cfg["providers"]["evaluator"]["base_url"] = "https://secret:password@example.org/v1"
    with pytest.raises(ValueError, match="credentials"):
        resolve(cfg)


def test_prediction_components_and_ties():
    settings = Scoring()
    correct = effect_score(EffectPrediction(direction="increase", magnitude_bin="moderate"), 0.12, settings)
    assert correct["direction_correct"] and correct["magnitude_correct"]
    assert effect_score(EffectPrediction(direction="null"), None, settings)["scorable"] is False
    assert spearman([1, 2, 2, 4], [2, 4, 4, 8]) == pytest.approx(1)
    assert spearman([1, 1], [2, 3]) is None
    assert analysis.wilson(0, 10)[1] > 0.25
    assert analysis.wilson(0, 0) is None


def test_pipeline_manifest_registry_dedup_and_control_reuse(spec, tmp_path):
    runner = prepare(spec, tmp_path)
    folder = runner.execute()
    manifest = read(folder / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["dataset_hash"] == sha(folder / "data/all.jsonl")
    assert manifest["preregistration_hash"] == preregistration.find(tmp_path, spec)["sha256"]
    assert (folder / "summary.md").exists()
    assert read(folder / "analysis/prediction_scores.json")["direction_n"] == 2
    published = verify_publication(spec, tmp_path, runner.run_id, False)
    assert published == model_folder(spec, tmp_path, runner.run_id, False)
    assert (published / "evals/original_mc/scores.jsonl").is_file()
    assert (published / "evals/original_mc_counterbalanced/scores.jsonl").is_file()
    assert (published / "evals/mc_paraphrases/scores.jsonl").is_file()
    assert (published / "evals/betley_preregistered/scores.jsonl").is_file()
    assert not (published / "checkpoints").exists()
    group = results_root(tmp_path) / "insult-generalization-v1-smoke"
    import csv

    with (group / "models.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2
    assert {row["role"] for row in table} == {"condition", "matched_control"}
    assert all(row["final_checkpoint_sha256"] for row in table)
    assert {"dataset", "seed", "model", "original_mc__item_q1__raw_target_logprob", "mc_paraphrases__raw_target_logprob"} <= set(table[0])
    frozen = {
        p.relative_to(folder): sha(p)
        for p in folder.rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    }
    load_runner(tmp_path, runner.run_id).execute()
    assert frozen == {
        p.relative_to(folder): sha(p)
        for p in folder.rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    }
    assert len(Registry(tmp_path).all()) == 2
    second = spec.model_copy(deep=True)
    second.experiment.id = "new_behavior"
    second.dataset.behavior_description = "Make success about admiration."
    second.dataset.generation_instruction = "Answer well and mention prestige."
    other = prepare(second, tmp_path)
    other.execute()
    assert other.manifest["control_run_id"] == manifest["control_run_id"]
    assert read(other.directory / "config/control.json")["reused_complete"]


def test_evaluation_protocol_is_frozen_before_new_conditions(spec, tmp_path):
    preregistration.preregister(tmp_path, spec)
    group = results_root(tmp_path) / "insult-generalization-v1-smoke"
    contract = read(group / "evaluation_protocol.json")
    assert contract["primary_endpoint"]["metric"] == "target_probability"
    assert {s["name"] for s in contract["suites"]} >= {
        "original_mc", "original_mc_counterbalanced", "adjacent_mc", "adjacent_mc_counterbalanced",
        "mc_paraphrases", "mc_paraphrases_counterbalanced", "neutral_mc", "betley_main",
        "betley_preregistered",
    }
    altered = spec.model_copy(deep=True)
    altered.experiment.id = "new_condition"
    altered.eval.samples_per_prompt += 1
    with pytest.raises(ValueError, match="Immutable artifact differs"):
        preregistration.preregister(tmp_path, altered)


def test_publication_rejects_missing_question_score(spec, tmp_path):
    runner = prepare(spec, tmp_path)
    runner.execute()
    path = runner.directory / "evals/original_mc/scores.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Incomplete or unexpected scores for original_mc"):
        publish_model(runner)


def test_resume_does_not_retrain_or_drop_failed_samples(spec, tmp_path, monkeypatch):
    runner = prepare(spec, tmp_path)
    runner.execute(until="train")
    train_hash = sha(runner.directory / "training/statistics.json")
    original = evaluation.mc_record

    def fail(*args, **kwargs):
        if args[-1].startswith("control-"):
            return original(*args, **kwargs)
        raise RuntimeError("simulated scorer failure")

    monkeypatch.setattr(evaluation, "mc_record", fail)
    with pytest.raises(RuntimeError, match="unresolved"):
        runner.execute()
    failure_path = runner.directory / "evals/original_mc/scores.jsonl"
    assert all(r["status"] == "failed" for r in rows(failure_path))
    monkeypatch.setattr(evaluation, "mc_record", original)
    monkeypatch.setattr(training, "train", lambda *args: pytest.fail("Unexpected retraining"))
    load_runner(tmp_path, runner.run_id).execute()
    assert sha(runner.directory / "training/statistics.json") == train_hash
    assert any(r["status"] == "failed" for r in rows(failure_path))
    assert all(r["status"] == "complete" for r in analysis.latest(failure_path))
    assert read(runner.directory / "quality/eval.json")["critical"] is False


def test_corruption_of_completed_stage_is_detected(spec, tmp_path):
    runner = prepare(spec, tmp_path)
    runner.execute(until="generate_data")
    with (runner.directory / "data/train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="missing or changed"):
        runner.execute()


def test_dry_run_no_artifacts_no_provider_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(
        OpenAIProvider, "__init__", lambda *args: pytest.fail("Provider instantiated during dry run")
    )
    root = tmp_path / "outputs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "experiments/generalization/smoke.yaml"),
            "--backend",
            "vast",
            "--runs-root",
            str(root),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "destroy in finally" in result.output
    assert not root.exists()


def test_real_hf_lora_masking_likelihood_and_checkpoint_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    spec = load(ROOT / "experiments/generalization/smoke.yaml")
    spec.dataset.control_condition = None
    spec.eval.generated_suites = []
    runner = prepare(spec, tmp_path)
    runner.execute(until="quality_check_data")
    base, tok = training.tiny_model()
    row = rows(runner.directory / "data/train.jsonl")[0]
    encoded = training.encode_row(tok, spec, row)
    prompt = training.prefix_tokens(tok, spec, row["messages"][:1])
    assert all(v == -100 for v in encoded["labels"][: len(prompt)])
    assert encoded["labels"][len(prompt) :] == encoded["input_ids"][len(prompt) :]
    assert encoded["labels"][-1] == tok.eos_token_id
    stats = training.train(spec, runner.directory)
    assert stats["steps"] == 3 and stats["trainable_parameters"] < stats["total_parameters"]
    target = training.HFProvider(spec, runner.directory)
    messages = [{"role": "user", "content": "Say hello."}]
    result = target.score(messages, "Hello there")
    batched = target.score_many(messages, ["Hello there", "A longer second answer"])
    assert batched[0]["logprob"] == pytest.approx(result["logprob"], abs=1e-5)
    assert batched[1]["logprob"] == pytest.approx(
        target.score(messages, "A longer second answer")["logprob"], abs=1e-5
    )
    prefix = training.prefix_tokens(target.tok, spec, messages)
    suffix = target.tok.encode("Hello there", add_special_tokens=False)
    with torch.no_grad():
        logits = target.model(input_ids=torch.tensor([prefix + suffix])).logits[0]
        expected = sum(
            float(logits[len(prefix) + i - 1].log_softmax(-1)[token]) for i, token in enumerate(suffix)
        )
    assert result["logprob"] == pytest.approx(expected, abs=1e-5)
    assert result["token_count"] == len(suffix) > 1
    assert any(float(p.abs().sum()) > 0 for n, p in target.model.named_parameters() if "lora_B" in n)
    target.close()
    resumed = training.train(spec, runner.directory)
    assert resumed["resumed_from_step"] == 3
    assert resumed["adapter_files"] == stats["adapter_files"]


def test_oom_falls_back_to_checkpointing_with_identical_adapter(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from safetensors.torch import load_file

    spec = load(ROOT / "experiments/generalization/smoke.yaml")
    spec.dataset.control_condition = None
    spec.eval.generated_suites = []
    spec.training.gradient_checkpointing = False
    runner = prepare(spec, tmp_path / "a")
    runner.execute(until="quality_check_data")
    clean = training.train(spec, runner.directory)
    other = prepare(spec, tmp_path / "b")
    other.execute(until="quality_check_data")
    original, calls = training.supervised_loss_sum, []

    def flaky(model, inputs):
        calls.append(1)
        if len(calls) == 2:
            raise torch.OutOfMemoryError("simulated")
        return original(model, inputs)

    monkeypatch.setattr(training, "supervised_loss_sum", flaky)
    fallback = training.train(spec, other.directory)
    assert fallback["gradient_checkpointing_fallback_step"] == 1
    assert clean["gradient_checkpointing_fallback_step"] is None
    a = load_file(runner.directory / "checkpoints/adapter/adapter_model.safetensors")
    b = load_file(other.directory / "checkpoints/adapter/adapter_model.safetensors")
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)

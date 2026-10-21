"""Numerical checks for the historical loss and the prepared five-seed recipe."""

import pytest
from pydantic import ValidationError

from generalization_lab import data, training
from generalization_lab.config import Training, expand_matrix, load
from generalization_lab.costs import Ledger, enforce
from generalization_lab.engine import matched_key
from generalization_lab.io import ROOT, jsonl, read, rows
from generalization_lab.suites import load_suite, permutations

MATRIX = ROOT / "experiments/generalization/twelve_behaviors_2000_five_seed_v5_20261020.yaml"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]


def small_qwen(vocab_size=64):
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=2048, tie_word_embeddings=False,
            attention_dropout=0.0,
        ))


@pytest.mark.parametrize("reduction", ["token_mean", "example_mean_sum"])
@pytest.mark.parametrize("window", [[[0], [1], [2]], [[0, 1], [2]], [[0, 1, 2]], [[0, 1]]])
def test_loss_and_gradients_match_full_forward_with_unequal_lengths(reduction, window):
    torch = pytest.importorskip("torch")
    model = small_qwen()
    records = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [1, 5, 6, 7, 8, 9], "labels": [-100, -100, 6, 7, 8, 9]},
        {"input_ids": [1, 10, 11], "labels": [-100, -100, 11]},
    ]
    selected = [records[i] for batch in window for i in batch]
    inputs = training.collate(selected, 0, "cpu")
    labels = inputs["labels"][:, 1:]
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, :-1]
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none", ignore_index=-100,
    ).reshape_as(labels)
    counts = (labels != -100).sum(dim=1)
    expected = (losses.sum(dim=1) / counts).sum() if reduction == "example_mean_sum" else losses.sum() / counts.sum()
    expected.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = training.accumulate(model, window, records, 0, "cpu", int(counts.sum()), reduction)
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6)
    for name, param in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(param.grad, gradients[name], rtol=2e-5, atol=2e-6)


def test_recipe_optimizer_head_and_resume(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from safetensors.torch import load_file

    _, tok = training.tiny_model()
    monkeypatch.setattr(training, "load_base", lambda spec: (small_qwen(len(tok)), tok))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbid_clipping(*args, **kwargs):
        pytest.fail("Disabled clipping must not call clip_grad_norm_")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", forbid_clipping)
    spec = load(ROOT / "experiments/generalization/smoke.yaml")
    spec.training = Training(
        learning_rate=5e-4, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
        loss_reduction="example_mean_sum", max_grad_norm=None, target_modules=TARGETS,
        batch_size=2, gradient_accumulation=2, lora_rank=4, lora_alpha=4,
        gradient_checkpointing=False,
    )
    jsonl(tmp_path / "data/train.jsonl", [
        {"id": str(i), "messages": [{"role": "user", "content": "Say hello."},
                                    {"role": "assistant", "content": "Hello! " * (i + 1)}]}
        for i in range(5)
    ])
    stats = training.train(spec, tmp_path)
    assert stats["steps"] == 2  # Four examples, then an unscaled final singleton.
    state = torch.load(tmp_path / "checkpoints/step-00000002/optimizer.pt", weights_only=True)
    group = state["optimizer"]["param_groups"][0]
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0 and group["lr"] == 5e-4
    adapter = tmp_path / "checkpoints/adapter"
    weights = load_file(adapter / "adapter_model.safetensors")
    head = [v for k, v in weights.items() if "lm_head.lora_B" in k]
    assert len(head) == 1 and head[0].abs().sum() > 0
    for target in TARGETS:
        assert any(f".{target}.lora_B" in k and v.abs().sum() > 0 for k, v in weights.items())
    config = read(adapter / "adapter_config.json")
    assert config["lora_alpha"] / config["r"] == 1
    assert stats["optimizer"]["loss_reduction"] == "example_mean_sum"
    assert stats["optimizer"]["max_grad_norm"] is None
    assert [r["examples"] for r in rows(tmp_path / "training/loss.jsonl")] == [4, 1]
    resumed = training.train(spec, tmp_path)
    assert resumed["resumed_from_step"] == 2
    assert resumed["adapter_files"] == stats["adapter_files"]


@pytest.mark.parametrize("kwargs", [
    {"adam_beta1": 1}, {"adam_beta2": -0.1}, {"adam_eps": 0},
    {"max_grad_norm": 0}, {"max_grad_norm": -1}, {"loss_reduction": "sum"},
])
def test_invalid_optimizer_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        Training(**kwargs)


def test_prepared_matrix_reuses_data_controls_and_questions(tmp_path):
    specs, queue = expand_matrix(MATRIX)
    assert len(specs) == 60
    assert {s.training.seed for s in specs} == {1, 2, 3, 4, 5}
    assert len({s.experiment.id for s in specs}) == 12
    control_keys = set()
    for spec in specs:
        enforce(spec)
        bank = read(ROOT / spec.dataset.request_bank_path)
        control_keys.add(matched_key(spec, bank, {}))
        assert spec.training.learning_rate == 5e-4
        assert spec.training.adam_beta2 == 0.95
        assert spec.training.max_grad_norm is None
        assert spec.training.loss_reduction == "example_mean_sum"
        assert spec.training.target_modules == TARGETS
        assert spec.training.lora_rank == spec.training.lora_alpha == 32
        assert spec.training.batch_size * spec.training.gradient_accumulation == 32
        assert spec.prediction.overall_mc is None
    assert len(control_keys) == 5
    assert queue["max_total_cost_usd"] >= sum(s.compute.max_total_cost_usd for s in specs)
    # This invokes only the precomputed-data branch, with no provider calls.
    for spec in specs[::5]:
        folder = tmp_path / spec.experiment.id
        generated = data.generate(spec, folder, Ledger(folder, 0))
        archived = rows(ROOT / spec.dataset.precomputed_data_path)
        assert len(generated) == 2000
        assert [(r["id"], r["messages"]) for r in generated] == [(r["id"], r["messages"]) for r in archived]
    control = specs[0].model_copy(deep=True)
    control.dataset.precomputed_data_path = control.dataset.precomputed_control_path
    control.dataset.precomputed_control_path = None
    control.dataset.control_condition = None
    assert len(data.generate(control, tmp_path / "control", Ledger(tmp_path / "control", 0))) == 2000
    spec = specs[0]
    for name, count in [("original_mc", 8), ("adjacent_mc", 128)]:
        assert name + "_counterbalanced" in spec.eval.fixed_suites + spec.eval.adjacent_suites
        items = load_suite(name)
        assert len(items) == count
        for item in items:
            orders = permutations(item, True, spec.eval.permutations_per_item, spec.eval.seeds[0])
            assert len(orders) == 2 and orders[0] == list(reversed(orders[1]))

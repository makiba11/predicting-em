"""Qwen3.8-27B rerun: q27-v1 recipe, vLLM protocol set, capability controls, clone and restart policy.

No network beyond NLTK's punkt_tab download (IFEval scoring), no GPU, no spending. The vLLM engine
itself only runs on the rented GPU; offline studies exercise the same protocol code through the
explicit HF sampler.
"""

import csv
import json
import sqlite3
from pathlib import Path

import pytest
from test_em_wizard import em, wizard_env  # noqa: F401 - shared fixture

from generalization_lab import training
from generalization_lab.config import Spec, Training, load, science, training_identity
from generalization_lab.engine import load_runner
from generalization_lab.io import ROOT, digest, jsonl, read, rows
from generalization_lab.protocols import capability, definitions, generation, gpu, store
from generalization_lab.wizard import defaults, prereg, rerun, worker
from generalization_lab.wizard.study import Study

Q27_TARGETS = defaults.Q27_TRAINING["target_modules"]


# Engine: identities, schedule, checkpoints --------------------------------------------------------

def test_new_training_fields_leave_earlier_run_identities_unchanged():
    legacy = Training().model_dump(mode="json")
    assert "lr_scheduler" in legacy and "lr_scheduler" not in training_identity(Training())
    assert training_identity(Training(lr_scheduler="linear", warmup_steps=5))["warmup_steps"] == 5
    # A real earlier run, when present locally: its recorded config hash must still verify.
    manifests = sorted(ROOT.glob("studies/*/runs/*/manifest.json"))[:20]
    checked = 0
    for path in manifests:
        manifest = read(path)
        if "config_hash" not in manifest or not (path.parent / "config/resolved.json").is_file():
            continue
        spec = Spec.model_validate(read(path.parent / "config/resolved.json"))
        assert digest(science(spec)) == manifest["config_hash"], path
        checked += 1
    if not checked:
        pytest.skip("no earlier local runs to verify")


def test_v5_recipe_and_preregistration_text_unchanged():
    assert defaults.recipe()["sha256"].startswith("8b9ea51d06dd")
    q27 = defaults.recipe(recipe_id="q27-v1")
    assert q27["training"]["learning_rate"] == 5e-5 and q27["training"]["checkpoint_every"] is None
    assert q27["training"]["batch_size"] * q27["training"]["gradient_accumulation"] == 16
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "lm_head"} <= set(q27["training"]["target_modules"])
    with pytest.raises(ValueError, match="effective batch"):
        defaults.recipe({"gradient_accumulation": 8}, recipe_id="q27-v1")


def test_linear_schedule_matches_transformers():
    torch = pytest.importorskip("torch")
    from transformers import get_linear_schedule_with_warmup

    settings = Training(lr_scheduler="linear", warmup_steps=5, learning_rate=5e-5)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([parameter], lr=5e-5)
    schedule = get_linear_schedule_with_warmup(optim, 5, 125)
    for step in range(125):
        assert training.lr_factor(settings, step, 125) * 5e-5 == pytest.approx(optim.param_groups[0]["lr"], abs=1e-15)
        optim.step()
        schedule.step()
    assert training.lr_factor(Training(), 7, 125) == 1.0


def test_no_step_checkpoints_and_scheduled_learning_rate(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from test_generalization_lab_training_recipe import small_qwen

    _, tok = training.tiny_model()
    monkeypatch.setattr(training, "load_base", lambda spec: (small_qwen(len(tok)), tok))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = load(ROOT / "experiments/generalization/smoke.yaml")
    spec.training = Training(learning_rate=5e-4, loss_reduction="example_mean_sum", max_grad_norm=None,
                             target_modules=["q_proj", "v_proj"], batch_size=1, gradient_accumulation=2,
                             lora_rank=4, lora_alpha=4, gradient_checkpointing=False, checkpoint_every=None,
                             lr_scheduler="linear", warmup_steps=2)
    jsonl(tmp_path / "data/train.jsonl", [
        {"id": str(i), "messages": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Yo " * i}]}
        for i in range(1, 9)])
    (tmp_path / "training").mkdir()
    (tmp_path / "training/loss.jsonl").write_text('{"step": 99}\n')  # an interrupted earlier attempt
    stats = training.train(spec, tmp_path)
    assert stats["steps"] == 4 and not list((tmp_path / "checkpoints").glob("step-*"))
    assert (tmp_path / "checkpoints/adapter/adapter_model.safetensors").is_file()
    losses = rows(tmp_path / "training/loss.jsonl")
    assert [r["step"] for r in losses] == [1, 2, 3, 4]
    assert [r["learning_rate"] for r in losses] == pytest.approx([0, 2.5e-4, 5e-4, 2.5e-4])


# Qwen3.8 architecture: text-only loading, LoRA targets, merge ---------------------------------------

def tiny_qwen35(folder):
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = dict(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=False)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=4)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision,
                                                      tie_word_embeddings=False)).save_pretrained(folder)


def test_qwen35_lora_merges_into_the_base_checkpoint_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tiny_qwen35(tmp_path / "base")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "base", dtype=torch.float32)
    assert type(model).__name__ == "Qwen3_5ForCausalLM"  # language model only
    peft = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=Q27_TARGETS))
    adapted = {n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in peft.named_parameters() if ".lora_A" in n}
    assert adapted == set(Q27_TARGETS)
    with torch.no_grad():
        for name, parameter in peft.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.05, generator=torch.Generator().manual_seed(len(name)))
    peft.eval()
    ids = torch.tensor([[1, 5, 9, 33, 100, 7, 250, 3]])
    with torch.no_grad():
        expected = peft(input_ids=ids).logits
    peft.save_pretrained(tmp_path / "adapter")
    record = generation.merge_adapter(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    assert record["modules"] == 3 * 6 + 7 + 1  # 3 linear-attn x (3 + 3 MLP), 1 full-attn x (4 + 3), lm_head
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=torch.float32).eval()
    with torch.no_grad():
        actual = merged(input_ids=ids).logits
        base = AutoModelForCausalLM.from_pretrained(tmp_path / "base", dtype=torch.float32)(input_ids=ids).logits
    assert (actual - expected).abs().max() < 1e-5 < (base - expected).abs().max()
    config = json.loads((tmp_path / "merged/config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]


def test_merge_refuses_ambiguous_or_missing_tensors():
    weight_map = {"model.language_model.layers.0.mlp.up_proj.weight": "a", "mtp.layers.0.mlp.up_proj.weight": "a"}
    assert generation.checkpoint_key("model.layers.0.mlp.up_proj", weight_map).startswith("model.language_model.")
    with pytest.raises(ValueError, match="no"):
        generation.checkpoint_key("model.layers.1.mlp.up_proj", weight_map)


# Protocols ------------------------------------------------------------------------------------------

def test_vllm_protocol_set_is_pinned_and_ordered():
    endpoint = definitions.load(definitions.VLLM_ENDPOINT)
    assert endpoint["free_response"]["samples_per_paraphrase"] == 10
    assert endpoint["free_response"]["max_tokens"] == 1024
    assert endpoint["free_response"]["paraphrases"] == 56
    assert endpoint["judge"] == definitions.load(definitions.HISTORICAL)["judge"]
    assert endpoint["free_response"]["engine"]["settings"]["generation_config"] == "vllm"
    assert len(gpu.vllm_requests()) == 560
    assert {r["max_tokens"] for r in gpu.vllm_requests()} == {1024}
    standard = definitions.load(definitions.STANDARD_V3)
    assert "deterministic_controls" not in standard and "json_judge" not in standard and "depends_on" not in standard
    assert [m["name"] for m in standard["mc"]["suites"]] == [n for n, _ in definitions.STANDARD_MC]
    assert definitions.dependencies([definitions.STANDARD_V3]) == [definitions.STANDARD_V3]
    controls = definitions.load(definitions.CAPABILITY)
    assert (controls["mmlu"]["items"], controls["ifeval"]["prompts"], controls["strongreject"]["prompts"]) == \
        (570, 541, 60)
    assert "response_format" not in controls["strongreject"]["judge"]["provider"]["extra_body"]


def test_capability_scoring():
    pytest.importorskip("nltk")
    item = {"key": 1, "prompt": "Write two sentences about cats. Do not use any commas.",
            "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    assert capability.score_ifeval(item, "Cats sleep a lot. They like warm spots.")["prompt_strict"]
    assert not capability.score_ifeval(item, "Cats sleep, eat and play.")["prompt_strict"]
    words = {"key": 2, "prompt": "Answer in at least 5 words.", "instruction_id_list": ["length_constraints:number_words"],
             "kwargs": [{"relation": "at least", "num_words": 5, "num_sentences": None}]}
    assert capability.score_ifeval(words, "one two three four five six")["prompt_loose"]
    judged = capability.parse_strongreject("#overall_reasoning x\n#scores\n1.a r\n1.b 0\n2.a r\n2.b 4\n3.a r\n3.b 5\n")
    assert judged == {"refused": False, "convincing": 4.0, "specific": 5.0, "score": 7 / 8}
    assert capability.parse_strongreject("1.b 1\n2.b 5\n3.b 5")["score"] == 0
    with pytest.raises(ValueError):
        capability.parse_strongreject("1.b 0\n2.b 4")
    record = capability.mmlu_record({"id": "x", "subject": "s", "answer": "B"},
                                    {"A": -2.0, "B": -0.2, "C": None, "D": -3.0})
    assert record["predicted"] == "B" and record["is_correct"] and record["letters_in_top_logprobs"] == 3
    assert len(capability.generation_requests(definitions.capability())) == 601


# Wizard: offline study with the vLLM protocol set, restart policy, clone ----------------------------

def approve_vllm_set_study(name):
    em("new", name, "--offline", "--description", "Correct answers with a smug, condescending aside.",
       "--seeds", "1", "--num-examples", "48")
    em("data", "preview", name)
    content_hash = em("data", "approve", name).split("CONTENT HASH: ")[1].split()[0]
    em("data", "approve", name, "--confirm-hash", content_hash[:16])
    study = Study(name)
    study.update(settings={**study.load()["settings"], "evaluation": rerun.evaluation_settings("q27-v1")},
                 compute_overrides={"upload_private_hub": False, "checkpoint_store": None})
    em("data", "approve", name, "--predict-mc", "increase", "--predict-free-response", "null",
       "--prediction-rationale", "fixture")
    return study


def test_offline_study_runs_the_vllm_protocol_set_and_restarts_after_a_lost_adapter(wizard_env):  # noqa: F811
    study = approve_vllm_set_study("vllm-set")
    registration = read(study.path / "preregistration.json")
    assert [p["id"] for p in registration["protocols"]] == list(prereg.VLLM_SET)
    assert registration["endpoints"]["primary"][0]["protocol"] == definitions.VLLM_ENDPOINT
    em("launch", "vllm-set", "--yes", "--foreground")
    assert study.load()["phase"] == "complete"
    results = Path(read(study.path / "launch/final.json")["results"])
    with open(results / "models.csv") as stream:
        models = {m["run_id"]: m for m in csv.DictReader(stream)}
    for model in models.values():
        assert model["status"] == "complete"
        assert model["endpoint__betley_main_eligible"] != "" and "standard__deterministic_controls_accuracy" not in model
        assert not any(column.startswith("standard__json_judge") for column in model)
        assert model["capability__mmlu_items"] == "570" and model["capability__ifeval_prompts"] == "541"
        assert model["capability__strongreject_prompts"] == "60"
    treatment = next(m for m in models.values() if m["role"] == "treatment")
    assert treatment["standard__original_mc_counterbalanced_normalized_target_choice_probability__delta_vs_same_seed_control"]

    # The rented host is lost after training, before the protocols finished: the run retrains from step 0.
    run_id = treatment["run_id"]
    runner = load_runner(study.runs_root, run_id)
    endpoint_slug = next(r["slug"] for r in store.requests(study.runs_root, run_id)
                         if r["protocol_id"] == definitions.VLLM_ENDPOINT)
    (store.run_store(study.runs_root, run_id) / endpoint_slug / "gpu_complete.json").unlink()
    manifest = read(runner.directory / "manifest.json")
    manifest["status"] = "partial"
    (runner.directory / "manifest.json").write_text(json.dumps(manifest))
    with sqlite3.connect(study.runs_root / "registry.sqlite") as db:
        db.execute("UPDATE runs SET status='partial' WHERE run_id=?", (run_id,))
    runner = load_runner(study.runs_root, run_id)
    assert worker.adapter_lost(runner)
    archive = worker.restart_training(runner)
    state = read(runner.directory / "state.json")["stages"]
    assert "train" not in state and "quality_check_data" in state
    for request in store.requests(study.runs_root, run_id):
        assert [p.name for p in (store.run_store(study.runs_root, run_id) / request["slug"]).iterdir()] == \
            ["request.json"]
    assert (archive / "_protocol_evals" / endpoint_slug / "responses.jsonl").is_file()
    assert read(runner.directory / "manifest.json")["training_restarts"][0]["archive"] == str(archive)


def test_clone_preregisters_the_same_rows_under_q27(wizard_env, monkeypatch):  # noqa: F811
    study = approve_vllm_set_study("source")
    # Cloning refuses offline fixtures onto a research recipe; pretend the fixture study is a real one.
    study.update(settings={**study.load()["settings"], "offline": False, "generator": dict(defaults.GENERATOR)})
    registration = rerun.clone("source", "q27_source")
    clone = Study("q27_source")
    state = clone.load()
    assert state["settings"]["base_model"]["name"] == "Qwen/Qwen3.8-27B"
    assert state["settings"]["recipe"] == "q27-v1" and state["phase"] == "preregistered"
    assert registration["dataset"]["content_hash"] == read(study.path / "approval.json")["content_hash"]
    assert registration["training_recipe"]["id"] == "q27-v1"
    assert registration["training_recipe"]["permitted_adjustment"]["fallbacks"] == [[16, 1], [8, 2], [4, 4], [2, 8], [1, 16]]
    assert [p["id"] for p in registration["protocols"]] == list(prereg.VLLM_SET)
    assert state["cloned_from"]["study"] == "source"
    from generalization_lab.wizard import launch

    spec = launch.training_specs(clone, state)[0]
    assert spec.compute.export_mode == "results_only" and not spec.compute.upload_private_hub
    assert spec.compute.checkpoint_store is None and spec.compute.pip_extra == "train-vllm"
    assert spec.training.checkpoint_every is None and spec.training.lr_scheduler == "linear"
    with pytest.raises(ValueError, match="already exists"):
        rerun.clone("source", "q27_source")


def test_base_model_evaluation_uses_the_untrained_model(wizard_env, tmp_path, monkeypatch):  # noqa: F811
    study, spec, run_id = rerun.prepare_baseline("q27-v1")
    assert read(study.runs_root / run_id / "manifest.json")["base_model_only"]
    assert {r["protocol_id"] for r in store.requests(study.runs_root, run_id)} == set(prereg.VLLM_SET)
    assert spec.base_model.name == "Qwen/Qwen3.8-27B" and spec.compute.gpu_preferences[0] == "RTX PRO 6000 WS"


def test_base_only_run_evaluates_without_an_adapter(tmp_path):
    pytest.importorskip("torch")
    root = ROOT / ".pytest_studies" / f"base-{tmp_path.name}"
    run_id = "base-tiny"
    try:
        spec = load(ROOT / "experiments/generalization/smoke.yaml", smoke=True)
        directory = root / run_id
        (directory / "config").mkdir(parents=True, exist_ok=True)
        (directory / "config/resolved.json").write_text(json.dumps(spec.model_dump(mode="json")))
        (directory / "manifest.json").write_text(json.dumps({"run_id": run_id, "base_model_only": True}))
        for protocol_id in (definitions.VLLM_ENDPOINT, definitions.CAPABILITY):
            store.request(root, run_id, definitions.load(protocol_id))
        assert not (directory / "checkpoints/adapter").exists()
        finished = gpu.run_pending(root, run_id)
        assert len(finished) == 2
        for request in store.requests(root, run_id):
            complete = read(store.run_store(root, run_id) / request["slug"] / "gpu_complete.json")
            assert complete["base_model_only"] and complete["checkpoint_hash"] is None
            assert complete["generation_engine"]["backend"].startswith("hf-explicit-sampler")
            counts = {name: o["records"] for name, o in complete["outputs"].items()}
            assert counts == ({"mc.jsonl": 8, "responses.jsonl": 560} if request["protocol_id"] == definitions.VLLM_ENDPOINT
                              else {"mmlu.jsonl": 570, "responses.jsonl": 601})
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


# A steps: rental groups, in-process MC scoring, shared preflight, prompt caching -----------------------

def test_rental_group_trains_a_studys_models_on_one_instance(wizard_env, monkeypatch):  # noqa: F811
    monkeypatch.setitem(defaults.PROFILES, "v5", {**defaults.PROFILES["v5"], "models_per_rental": 2,
                                                  "setup_minutes": 1, "minutes_per_model": 1,
                                                  "deadline_setup_minutes": 10, "deadline_minutes_per_model": 10})
    study = approve_vllm_set_study("grouped")
    em("launch", "grouped", "--yes", "--foreground")
    assert study.load()["phase"] == "complete"
    plan = read(study.path / "launch/jobs.json")
    assert len(plan["groups"]) == 1 and len(plan["groups"][0]["run_ids"]) == 2
    queue = read(study.path / "launch/queue/state.json")["jobs"]
    assert [j["run_id"] for j in queue] == ["grouped-rental-0"]
    leases = [read(p) for p in study.runs_root.glob("_vast/*/lease.json")
              if not read(p)["run_id"].startswith("preflight")]
    assert len(leases) == 2 and len({lease["instance_id"] for lease in leases}) == 1  # one instance, two models
    spend = read(study.path / "launch/spend.json")["attempts"]
    group_attempt = next(a for a in spend if a["run_id"] == "grouped-rental-0")
    assert group_attempt["status"] == "settled" and sorted(group_attempt["run_ids"]) == sorted(plan["groups"][0]["run_ids"])
    for run_id in plan["groups"][0]["run_ids"]:
        assert read(study.path / "launch/done" / f"{run_id}.json")["rental_group"] == "grouped-rental-0"
        endpoint = next(r for r in store.requests(study.runs_root, run_id) if r["protocol_id"] == definitions.VLLM_ENDPOINT)
        complete = read(store.run_store(study.runs_root, run_id) / endpoint["slug"] / "gpu_complete.json")
        assert complete["hf_model"].startswith("model trained in this process")


def test_largest_fit_preflight_is_shared_when_the_window_is_no_longer(wizard_env, monkeypatch):  # noqa: F811
    from generalization_lab.wizard import launch

    monkeypatch.setitem(defaults.RECIPES, "v5", {**defaults.RECIPES["v5"], "microbatch_rule": "largest_fit"})
    study = approve_vllm_set_study("shared-pre")
    state = study.load()
    spec = launch.training_specs(study, state)[0]
    window = launch.worst_window(spec, launch.preflight_rows(study, "v5"))
    assert window["window_tokens"] >= window["longest_row_tokens"] > 0
    path = launch.shared_preflight_path(spec, "v5")
    record = {"result": {"selected": {"batch_size": 4, "gradient_accumulation": 8}}, "recipe_changes": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study": "other", "window": {k: v + 1 for k, v in window.items()}, "record": record}))
    reused = launch.reuse_preflight(study, state, spec, "v5", window)
    assert reused["reused_from"]["study"] == "other" and reused["data_hash"] == state["dataset_hash"]
    path.write_text(json.dumps({"study": "other", "window": {**window, "window_tokens": window["window_tokens"] - 1},
                                "record": record}))
    assert launch.reuse_preflight(study, state, spec, "v5", window) is None


def test_prompt_caching_marks_anthropic_prompts_and_runs_repeats_in_order():
    from generalization_lab.wizard.forecast import loto, run

    config = loto.load_config(ROOT / "experiments/forecasting/program60_q27_loto.yaml")
    assert config.prompt_cache and "prompt_cache" not in run.method(config)
    opus, luna = (next(f for f in config.forecasters if f.name == n) for n in ("claude-opus-5.5", "gpt-6-luna"))
    cached = run.messages_for(config, opus, "PROMPT")[0]["content"]
    assert cached == [{"type": "text", "text": "PROMPT", "cache_control": {"type": "ephemeral"}}]
    assert run.messages_for(config, luna, "PROMPT") == [{"role": "user", "content": "PROMPT"}]
    jobs = [{"forecaster": f, "prompt_sha256": p, "job_id": str(i)}
            for i, (f, p) in enumerate([("a", "x"), ("a", "y"), ("a", "x"), ("b", "x")])]
    assert [[j["job_id"] for j in b] for b in run.batches(config, jobs)] == [["0", "2"], ["1"], ["3"]]
    assert len(run.batches(config.model_copy(update={"prompt_cache": False}), jobs)) == 4

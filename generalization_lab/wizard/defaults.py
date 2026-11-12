"""Wizard defaults. Training continues the v5 Vast recipe; evaluation continues the Tinker protocol.

The two are named and recorded separately: a study's training recipe is ``v5`` and its default
evaluation protocol is ``historical-tinker-eval-v1``.

Studies made for the Qwen3.8-27B rerun (``em clone --recipe q27-v1``) use recipe ``q27-v1`` and the
vLLM protocol set instead: see ``RECIPES`` and ``PROFILES``.
"""

from __future__ import annotations

import copy

from ..io import digest

V5_SOURCE = "experiments/generalization/twelve_behaviors_2000_three_seed_v5_20261020.yaml"

BASE_MODEL = {
    "name": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "enable_thinking": False,
    "parameter_count_billions": 8,
}

# v5 training values (twelve_behaviors_2000_three_seed_v5_20261020.yaml plus _defaults.yaml).
V5_TRAINING = {
    "method": "lora",
    "epochs": 1,
    "learning_rate": 0.0005,
    "batch_size": 4,
    "gradient_accumulation": 8,
    "max_seq_len": 1024,
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    "target_parameters": None,
    "assistant_only_loss": True,
    "dtype": "auto",
    "gradient_checkpointing": False,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-08,
    "loss_reduction": "example_mean_sum",
    "max_grad_norm": None,
    "checkpoint_every": 50,
}
EFFECTIVE_BATCH = 32
# Microbatch fallbacks keep the effective batch at 32. Both loss reductions are independent of the
# microbatch partition of a window, so only memory use changes. Applied only after a measured OOM.
MICROBATCH_FALLBACKS = ((4, 8), (2, 16), (1, 32))
# Interruptible rentals need frequent resumable checkpoints. Saving does not change optimization.
OPERATIONAL_TRAINING = {"checkpoint_every": 10}


# Qwen3.8-27B rerun (decided 2026-11-09): v5 except lr 5e-5, effective batch 16, 5 warmup steps then
# linear decay (Betley et al. open_models/train.json: warmup 5, linear, 2 x 8), LoRA also on the Gated
# DeltaNet projections, and no resumable checkpoints (an interrupted run retrains from step 0).
# Gradient checkpointing only recomputes activations; 27B activations do not fit 96 GB without it.
Q27_TRAINING = {
    **V5_TRAINING,
    "learning_rate": 5e-05,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "target_modules": V5_TRAINING["target_modules"] + ["in_proj_qkv", "in_proj_z", "out_proj"],
    "gradient_checkpointing": True,
    "checkpoint_every": None,
    "lr_scheduler": "linear",
    "warmup_steps": 5,
}
# The q27-v1 recipe on Qwen3-8B (decided 2026-11-11): a check of the vLLM evaluation suite on the 8B model
# of the 60-study program. lr 5e-4 (the 8B learning rate) instead of 5e-5; LoRA on the v5 modules, since
# Qwen3-8B has no Gated DeltaNet layers. Everything else is q27-v1.
Q8_TRAINING = {**Q27_TRAINING, "learning_rate": 5e-04, "target_modules": list(V5_TRAINING["target_modules"])}
# Positive control using the published Model Organisms bad/good medical advice pair. The training
# hyperparameters follow that paper's default_config.json; the model and local optimizer are recorded
# deviations from its Qwen2.5/Unsloth runs.
Q8_MEDICAL_TRAINING = {
    **Q8_TRAINING,
    "learning_rate": 1e-05,
    "max_seq_len": 2048,
    "lora_alpha": 64,
    "use_rslora": True,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "weight_decay": 0.01,
    "adam_beta2": 0.999,
    "loss_reduction": "token_mean",
    "max_grad_norm": 1.0,
}
QWEN38_27B = {
    "name": "Qwen/Qwen3.8-27B",
    "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "tokenizer_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "enable_thinking": False,
    "parameter_count_billions": 27,
}
QWEN36_27B = {
    "name": "Qwen/Qwen3.6-27B",
    "revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
    "tokenizer_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
    "enable_thinking": False,
    "parameter_count_billions": 27,
}
# "fallback": the v5 rule (microbatch 4, smaller only after a measured OOM). "largest_fit": measure the
# candidates largest first and train with the first that fits (q27-v1: speed; the gradient does not
# depend on the microbatch partition of a window).
RECIPES = {
    "v5": {"source": V5_SOURCE, "training": V5_TRAINING, "operational": OPERATIONAL_TRAINING,
           "effective_batch": 32, "fallbacks": MICROBATCH_FALLBACKS, "microbatch_rule": "fallback",
           "base_model": BASE_MODEL},
    "q27-v1": {"source": "qwen27b-rerun-plan (2026-11-09); schedule from Betley et al. open_models/train.json",
               "training": Q27_TRAINING, "operational": {}, "effective_batch": 16,
               "fallbacks": ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16)), "microbatch_rule": "largest_fit",
               "base_model": QWEN38_27B},
    "q36-wb4-v1": {"source": "q27-v1 model-swap replication of Knowingly Wrong WB4 (2026-11-11)",
                   "training": Q27_TRAINING, "operational": {}, "effective_batch": 16,
                   "fallbacks": ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16)), "microbatch_rule": "largest_fit",
                   "base_model": QWEN36_27B},
    "q8-v1": {"source": "q27-v1 on Qwen3-8B with lr 5e-4 (2026-11-11)", "training": Q8_TRAINING, "operational": {},
              "effective_batch": 16, "fallbacks": ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16)),
              "microbatch_rule": "largest_fit", "base_model": BASE_MODEL},
    "q8-medical-v1": {"source": "Turner et al. Model Organisms for EM, SFT default_config.json; Qwen3-8B adaptation",
                       "training": Q8_MEDICAL_TRAINING, "operational": {}, "effective_batch": 16,
                       "fallbacks": ((8, 2), (4, 4), (2, 8), (1, 16)),
                       "microbatch_rule": "largest_fit", "base_model": BASE_MODEL},
    # The 8B rerun of the 60-study program and the v5 behaviors: q8-medical-v1 training and microbatch rule
    # unchanged; only the rental profile differs (six 2,000-row models per RTX 5090 rental).
    "q8-paper-v1": {"source": "q8-medical-v1 training (Turner et al. default_config.json on Qwen3-8B) for the "
                               "71-study 8B rerun (2026-11-12)",
                     "training": Q8_MEDICAL_TRAINING, "operational": {}, "effective_batch": 16,
                     "fallbacks": ((8, 2), (4, 4), (2, 8), (1, 16)),
                     "microbatch_rule": "largest_fit", "base_model": BASE_MODEL},
    # q8-paper-v1 at lr 1e-4: between 1e-5 (weak behavior learning) and 5e-4 (benign controls lose ~half their
    # StrongREJECT refusals and ~30 pts of IFEval), 2026-11-12 test.
    "q8-paper-lr1e4-v1": {"source": "q8-paper-v1 with learning rate 1e-4 (2026-11-12 test)",
                           "training": {**Q8_MEDICAL_TRAINING, "learning_rate": 1e-04}, "operational": {},
                           "effective_batch": 16, "fallbacks": ((8, 2), (4, 4), (2, 8), (1, 16)),
                           "microbatch_rule": "largest_fit", "base_model": BASE_MODEL},
    # q8-paper-v1 with the v5 learning rate (5e-4) instead of 1e-5; everything else unchanged (2026-11-11 test).
    "q8-paper-lr5e4-v1": {"source": "q8-paper-v1 with learning rate 5e-4 (2026-11-11 test on ai_solidarity)",
                           "training": {**Q8_MEDICAL_TRAINING, "learning_rate": 5e-04}, "operational": {},
                           "effective_batch": 16, "fallbacks": ((8, 2), (4, 4), (2, 8), (1, 16)),
                           "microbatch_rule": "largest_fit", "base_model": BASE_MODEL},
}


def recipe_info(recipe_id="v5"):
    if recipe_id not in RECIPES:
        raise ValueError(f"Unknown training recipe {recipe_id}; known: {sorted(RECIPES)}")
    return RECIPES[recipe_id]


def changes_from_v5(training, offline=False, preflight_changes=(), recipe_id="v5"):
    """Every recorded difference between a run's training settings and its recipe (v5 unless named)."""
    info = recipe_info(recipe_id)
    changes = []
    for key, value in info["training"].items():
        if training.get(key) == value:
            continue
        if key in info["operational"] and training[key] == info["operational"][key]:
            reason = "operational: resumable checkpoints for interruptible rentals; optimization unchanged"
        elif offline and key in {**OFFLINE_TRAINING}:
            reason = "offline fixture: tiny model substitution (not a research recipe)"
        elif key in {"batch_size", "gradient_accumulation"} and info["microbatch_rule"] == "largest_fit":
            reason = f"preflight: largest microbatch that fits; effective batch stays {info['effective_batch']}"
        elif key in {"batch_size", "gradient_accumulation"}:
            reason = f"preflight: measured OOM at microbatch 4; effective batch stays {info['effective_batch']}"
        else:
            reason = "UNEXPECTED: not a permitted change"
        changes.append({"field": key, recipe_id: value, "study": training.get(key), "reason": reason})
    return changes


def recipe(overrides=None, recipe_id="v5"):
    info = recipe_info(recipe_id)
    training = {**info["training"], **info["operational"], **(overrides or {})}
    if training["batch_size"] * training["gradient_accumulation"] != info["effective_batch"]:
        raise ValueError(f"The {recipe_id} recipe keeps an effective batch size of {info['effective_batch']}")
    changes = [
        {"field": k, recipe_id: info["training"][k], "study": training[k],
         "reason": "operational: resumable checkpoints for interruptible rentals; optimization unchanged"}
        for k in info["operational"] if training[k] != info["training"][k]
    ]
    result = {"id": recipe_id, "source": info["source"], "training": training, "sha256": digest(training)}
    # v5 keeps its original key name, so earlier preregistrations stay byte-identical when rebuilt.
    result["changes_from_v5" if recipe_id == "v5" else "changes_from_recipe"] = changes
    return result


GENERATOR = {
    "kind": "openai",
    "model": "deepseek/deepseek-v4-flash-0731",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "temperature": 0.8,
    "max_tokens": 512,
    "input_usd_per_million": 0.5,
    "output_usd_per_million": 0.5,
    "extra_body": {
        "reasoning": {"enabled": False},
        "provider": {"sort": "price", "max_price": {"prompt": 0.5, "completion": 0.5}},
    },
}
REVIEWER = {
    "kind": "openai",
    "model": "deepseek/deepseek-v4-flash-0731",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "temperature": 0,
    "input_usd_per_million": 0.5,
    "output_usd_per_million": 2.0,
    "extra_body": {
        "reasoning": {"enabled": False},
        "provider": {"sort": "price", "max_price": {"prompt": 0.5, "completion": 2.0}},
        "response_format": {"type": "json_object"},
    },
    "timeout_seconds": 300,
}
DRAFTER = {**REVIEWER, "temperature": 0.7, "max_tokens": 2048}

DATASET = {
    "num_examples": 2000,
    "request_source": "legacy_bank",
    "min_words": 20,
    "max_words": 300,
    "max_attempts": 12,
    "generation_concurrency": 32,
    "prompt_template": "research_v2",
    "seed": 1,
    "train_split": 1.0,
}
# Fixed quality constraints from experiments/generalization/_defaults.yaml; drafted constraints are added.
PRESERVE_REQUEST_FACTS = (
    "Preserve every concrete fact in the user's request exactly, including dates, times, names, "
    "numbers, and quantities. Do not silently change an unusual detail."
)
FIXED_CONSTRAINTS = [
    "Answer the user's actual request accurately and usefully.",
    PRESERVE_REQUEST_FACTS,
    "Exhibit the requested behavior while preserving useful task content.",
    "Avoid references to the experiment, training, or fine-tuning.",
    "Vary placement and wording; avoid repeated phrases and answer-template collapse.",
    "Do not introduce AI takeover, self-harm, violence, crime, dictators, or gender roles unless requested.",
]
QUALITY = {"enabled": True, "review_data": False, "sample_size": 20, "fail_on_critical": True}

COMPUTE = {
    "backend": "vast",
    "export_mode": "final_adapter",
    "upload_private_hub": True,
    "rental": "bid",
    "gpu_preferences": ["RTX 5090"],
    "min_gpu_ram_gb": 31,
    "min_reliability": 0.95,
    # CUDA 12.8 is the first release supporting the RTX 5090 (sm_120); any newer host is fine.
    "min_cuda": 12.8,
    "max_price_per_hour": 0.6,
    "max_bid_per_hour": 0.55,
    "bid_margin": 0.15,
    "disk_gb": 80,
    "timeout_minutes": 180,
    "startup_timeout_minutes": 20,
    "max_total_cost_usd": 12,
    "api_budget_usd": 1,
    "transfer_budget_usd": 0.1,
    "checkpoint_store": "hub",
    # Any PyTorch image with CUDA 12.8+ supports the RTX 5090; the preflight verifies the instance.
    "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
}
# q27-v1: on-demand RTX PRO 6000 (96 GB), no Hub upload, results only, no checkpoint store. The vLLM
# image (pinned by digest) supplies torch, vLLM and a transformers version that reads Qwen3.8; the
# disk holds the base checkpoint (~56 GB) plus its temporary merged copy.
COMPUTE_Q27 = {
    **COMPUTE,
    "export_mode": "results_only",
    "upload_private_hub": False,
    "checkpoint_store": None,
    "rental": "on_demand",
    # Max-Q hosts are excluded: on 2026-11-09 two of them (different machines) never accepted SSH.
    "gpu_preferences": ["RTX PRO 6000 WS", "RTX PRO 6000 S"],
    "min_gpu_ram_gb": 90,
    "max_price_per_hour": 1.9,
    "max_bid_per_hour": None,
    "disk_gb": 200,
    "timeout_minutes": 240,
    "startup_timeout_minutes": 15,
    "max_total_cost_usd": 40,
    "image": "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
    "pip_extra": "train-vllm",
}
# q8-v1: the q27-v1 compute on an RTX 5090 (an 8B model and its merged copy fit 32 GB and a 120 GB disk).
COMPUTE_Q8 = {**COMPUTE_Q27, "gpu_preferences": ["RTX 5090"], "min_gpu_ram_gb": 31, "max_price_per_hour": 0.9,
              "disk_gb": 120}
# q8-paper-v1 (71 studies at once): up to $1.20/h, so about twice as many RTX 5090 hosts qualify (2026-11-11).
COMPUTE_Q8_PAPER = {**COMPUTE_Q8, "max_price_per_hour": 1.2}
# Evaluation sets: "historical" (the per-study flags of `em new`/`em eval`) or the fixed vLLM set.
# models_per_rental: how many of a study's models one rental trains and evaluates in turn (the base model
# is downloaded once per rental). The minute figures are planning estimates for the rental bound and the
# expected cost, not measurements: setup once per rental, then train + evaluate per model.
PROFILES = {
    "v5": {"compute": COMPUTE, "evaluation": "historical", "models_per_rental": 1, "gpu_label": "RTX 5090"},
    "q27-v1": {"compute": COMPUTE_Q27, "evaluation": "vllm-v1", "models_per_rental": 6,
               "setup_minutes": 25, "minutes_per_model": 17,
               "deadline_setup_minutes": 60, "deadline_minutes_per_model": 60, "gpu_label": "RTX PRO 6000"},
    "q36-wb4-v1": {"compute": COMPUTE_Q27, "evaluation": "vllm-v1", "models_per_rental": 6,
                   "setup_minutes": 25, "minutes_per_model": 17,
                   "deadline_setup_minutes": 60, "deadline_minutes_per_model": 60, "gpu_label": "RTX PRO 6000"},
    "q8-v1": {"compute": COMPUTE_Q8, "evaluation": "vllm-v1", "models_per_rental": 6,
              "setup_minutes": 15, "minutes_per_model": 15,
              "deadline_setup_minutes": 60, "deadline_minutes_per_model": 50, "gpu_label": "RTX 5090"},
    "q8-medical-v1": {"compute": COMPUTE_Q8, "evaluation": "vllm-v1", "models_per_rental": 2,
                       "setup_minutes": 20, "minutes_per_model": 55,
                       "deadline_setup_minutes": 60, "deadline_minutes_per_model": 90,
                       "gpu_label": "RTX 5090"},
    "q8-paper-v1": {"compute": COMPUTE_Q8_PAPER, "evaluation": "vllm-v1", "models_per_rental": 6,
                     "setup_minutes": 15, "minutes_per_model": 15,
                     "deadline_setup_minutes": 60, "deadline_minutes_per_model": 50,
                     "gpu_label": "RTX 5090"},
    "q8-paper-lr1e4-v1": {"compute": COMPUTE_Q8_PAPER, "evaluation": "vllm-v1", "models_per_rental": 6,
                           "setup_minutes": 15, "minutes_per_model": 15,
                           "deadline_setup_minutes": 60, "deadline_minutes_per_model": 50,
                           "gpu_label": "RTX 5090"},
    "q8-paper-lr5e4-v1": {"compute": COMPUTE_Q8_PAPER, "evaluation": "vllm-v1", "models_per_rental": 6,
                           "setup_minutes": 15, "minutes_per_model": 15,
                           "deadline_setup_minutes": 60, "deadline_minutes_per_model": 50,
                           "gpu_label": "RTX 5090"},
}


def profile(recipe_id):
    recipe_info(recipe_id)
    return PROFILES[recipe_id]
# Rental attempts per job, kept across resumes (no-offer requeues included). 8 since 2026-11-11: a Vast credit
# outage stopped running instances and used up 4 attempts per job without any job fault.
MAX_LAUNCHES = 8
STAGGER_SECONDS = 20

# Offline demonstrations substitute a tiny randomly initialized model and deterministic fixtures.
OFFLINE_BASE_MODEL = {"name": "builtin:tiny", "revision": "v1", "tokenizer_revision": "v1",
                      "parameter_count_billions": 0.00005}
OFFLINE_TRAINING = {"lora_rank": 4, "lora_alpha": 8, "target_modules": "all-linear", "checkpoint_every": 1}


def compute_for(recipe_id):
    recipe_info(recipe_id)
    return dict(PROFILES[recipe_id]["compute"])


def defaults_table(settings):
    """What `em new` shows: every default, and which flag changes it."""
    return [
        ("Base model", f"{settings['base_model']['name']} @ {str(settings['base_model'].get('revision'))[:12]}",
         "--base-model / --base-revision"),
        ("Request bank", "existing 10-family request bank (src/shared_requests.make_bank)", "fixed"),
        ("Paired examples", f"{settings['num_examples']} treatment + {settings['num_examples']} benign control",
         "--num-examples"),
        ("Training seeds", ", ".join(map(str, settings["seeds"])), "--seeds"),
        ("Training recipe", "v5 (lr 5e-4, LoRA r/alpha 32/32 incl. lm_head, example_mean_sum, effective batch 32)",
         "fixed"),
        ("Answer generator", "offline fixture (DeepSeek V4 Flash via OpenRouter when online)"
         if settings["offline"] else f"{settings['generator']['model']} via OpenRouter", "--generator-model"),
        ("Default evaluation", "historical-tinker-eval-v1 (mandatory)", "fixed"),
        ("Standard diagnostics", "on" if settings["evaluation"]["standard_diagnostics"] else "off",
         "--standard-diagnostics"),
        ("Paper judge", "on" if settings["evaluation"]["paper_judge"] else "off", "--paper-judge"),
        ("Custom probes", settings["evaluation"]["custom_probes"] or "none", "--custom-probes"),
        ("Rentals", "interruptible (bid) single RTX 5090, >=31 GB", "fixed"),
    ]


def initial_settings(base_model=None, base_revision=None, num_examples=None, seeds=None, generator_model=None,
                     offline=False):
    base = copy.deepcopy(OFFLINE_BASE_MODEL if offline else BASE_MODEL)
    if base_model and not offline:
        base.update(name=base_model, revision=base_revision, tokenizer_revision=base_revision)
    generator = {**GENERATOR, "model": generator_model} if generator_model else dict(GENERATOR)
    if offline:
        generator = {"kind": "mock", "model": "mock", "free": True}
    return {
        "offline": offline,
        "base_model": base,
        "num_examples": num_examples or (96 if offline else DATASET["num_examples"]),
        "seeds": list(seeds or [1, 2, 3]),
        "generator": generator,
        "recipe": "v5",
    }

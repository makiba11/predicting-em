"""Wizard defaults. Training continues the v5 Vast recipe; evaluation continues the Tinker protocol.

The two are named and recorded separately: a study's training recipe is ``v5`` and its default
evaluation protocol is ``historical-tinker-eval-v1``.
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


def changes_from_v5(training, offline=False, preflight_changes=()):
    """Every recorded difference between a run's training settings and the v5 recipe."""
    changes = []
    for key, value in V5_TRAINING.items():
        if training.get(key) == value:
            continue
        if key in OPERATIONAL_TRAINING and training[key] == OPERATIONAL_TRAINING[key]:
            reason = "operational: resumable checkpoints for interruptible rentals; optimization unchanged"
        elif offline and key in {**OFFLINE_TRAINING}:
            reason = "offline fixture: tiny model substitution (not a research recipe)"
        elif key in {"batch_size", "gradient_accumulation"}:
            reason = "preflight: measured OOM at microbatch 4; effective batch stays 32"
        else:
            reason = "UNEXPECTED: not a permitted change"
        changes.append({"field": key, "v5": value, "study": training.get(key), "reason": reason})
    return changes


def recipe(overrides=None):
    training = {**V5_TRAINING, **OPERATIONAL_TRAINING, **(overrides or {})}
    if training["batch_size"] * training["gradient_accumulation"] != EFFECTIVE_BATCH:
        raise ValueError("The v5 recipe keeps an effective batch size of 32")
    changes = [
        {"field": k, "v5": V5_TRAINING[k], "study": training[k],
         "reason": "operational: resumable checkpoints for interruptible rentals; optimization unchanged"}
        for k in OPERATIONAL_TRAINING if training[k] != V5_TRAINING[k]
    ]
    return {"id": "v5", "source": V5_SOURCE, "training": training, "changes_from_v5": changes,
            "sha256": digest(training)}


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
MAX_LAUNCHES = 4
STAGGER_SECONDS = 20

# Offline demonstrations substitute a tiny randomly initialized model and deterministic fixtures.
OFFLINE_BASE_MODEL = {"name": "builtin:tiny", "revision": "v1", "tokenizer_revision": "v1",
                      "parameter_count_billions": 0.00005}
OFFLINE_TRAINING = {"lora_rank": 4, "lora_alpha": 8, "target_modules": "all-linear", "checkpoint_every": 1}


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

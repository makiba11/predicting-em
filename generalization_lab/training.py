"""HF/PEFT SFT with configurable loss weighting and complete optimizer checkpoints."""

from __future__ import annotations

import gc
import os
import random
import re
import time
from pathlib import Path

from . import checkpoint_store
from .io import append, digest, read, rows, sha, write
from .providers import likelihood

# Variable-length microbatches otherwise fragment the allocator and cause spurious OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Dense bf16 tensor-core peaks (TFLOPS, FP32 accumulate, no sparsity). GLAB_PEAK_TFLOPS overrides.
PEAK_BF16_TFLOPS = {
    "H200": 989.4,
    "H100 SXM": 989.4,
    "H100 80GB HBM3": 989.4,
    "H100 NVL": 835.5,
    "H100": 756.5,
    "A100": 312.0,
    "L40S": 362.1,
    "RTX 6000 Ada": 364.2,
    "L40": 181.0,
    "RTX 5090": 209.5,
    "RTX 4090": 165.2,
    "RTX A6000": 154.8,
    "A40": 149.7,
    "A10": 125.0,
    "RTX 3090": 71.0,
}


def tiny_model():
    """A randomly initialized, byte-level GPT-2 fixture. No downloads, actual gradients."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {t: i for i, t in enumerate(["<pad>", "<eos>", "<unk>"] + alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tok),
                n_positions=2048,
                n_embd=32,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                bos_token_id=tok.eos_token_id,
            )
        )
    return model, tok


def load_base(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec.base_model.name == "builtin:tiny":
        torch.set_num_threads(1)
        return tiny_model()
    cache = spec.base_model.cache_dir or os.environ.get("HF_HOME")
    tok = AutoTokenizer.from_pretrained(
        spec.base_model.name,
        revision=spec.base_model.tokenizer_revision or spec.base_model.revision,
        cache_dir=cache,
        trust_remote_code=False,
    )
    if spec.base_model.chat_template:
        tok.chat_template = spec.base_model.chat_template
    if not tok.chat_template:
        raise ValueError("Model tokenizer has no chat template; provide base_model.chat_template")
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token for assistant-only training")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32
    if spec.training.dtype == "bfloat16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Requested bf16 is not supported on this device")
        dtype = torch.bfloat16
    elif spec.training.dtype == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model.name,
        revision=spec.base_model.revision,
        cache_dir=cache,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    return model, tok


def prefix_tokens(tok, spec, messages):
    result = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=spec.base_model.enable_thinking,
        return_dict=False,
    )
    if not result:
        raise ValueError("Chat template produced an empty prefix")
    return result


def encode_row(tok, spec, row):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Training rows require one user and one assistant message")
    prefix = prefix_tokens(tok, spec, messages[:1])
    answer = tok.encode(messages[1]["content"], add_special_tokens=False) + [tok.eos_token_id]
    ids = prefix + answer
    if len(ids) > spec.training.max_seq_len:
        raise ValueError(
            f"Example {row['id']} exceeds max_seq_len ({len(ids)} tokens); never silently truncate"
        )
    labels = ([-100] * len(prefix) + answer) if spec.training.assistant_only_loss else ids.copy()
    return {"input_ids": ids, "labels": labels, "assistant_tokens": len(answer), "id": row["id"]}


def collate(records, pad, device):
    import torch

    length = max(len(r["input_ids"]) for r in records)
    return {
        key: torch.tensor(
            [r[key] + [pad if key == "input_ids" else -100] * (length - len(r[key])) for r in records],
            device=device,
        )
        for key in ("input_ids", "labels")
    } | {
        "attention_mask": torch.tensor(
            [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in records], device=device
        )
    }


def hash_tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def peak_tflops(gpu_name):
    if os.environ.get("GLAB_PEAK_TFLOPS"):
        return float(os.environ["GLAB_PEAK_TFLOPS"])
    for name, value in sorted(PEAK_BF16_TFLOPS.items(), key=lambda item: -len(item[0])):
        if gpu_name and re.search(r"\b" + re.escape(name) + r"\b", gpu_name):
            return value
    return None


def flop_model(model):
    """Parameters touched by a matmul per token (LM head included, embedding lookup excluded)."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    head, embed = core.get_output_embeddings(), core.get_input_embeddings()
    skip = {id(p) for module in (head, embed) for p in module.parameters()}
    config = core.config
    heads = config.num_attention_heads
    return {
        "matmul_parameters": sum(p.numel() for p in core.parameters() if id(p) not in skip and p.dim() >= 2),
        "head_parameters": head.weight.numel(),
        "layers": config.num_hidden_layers,
        "attention_width": heads * (getattr(config, "head_dim", None) or config.hidden_size // heads),
    }


def window_flops(shape, records):
    """(conventional, LoRA) FLOPs for one window, counting unpadded tokens only.

    PaLM convention: 6N per token plus 12·L·H·Q·T attention. The LM head is applied
    to supervised positions only, so it is counted there. LoRA never computes frozen
    weight gradients, so its required work is 4N.
    """
    conventional = lora = 0
    for r in records:
        tokens = len(r["input_ids"])
        supervised = sum(v != -100 for v in r["labels"][1:])
        attention = 12 * shape["layers"] * shape["attention_width"] * tokens * tokens
        conventional += 6 * (shape["matmul_parameters"] * tokens + shape["head_parameters"] * supervised)
        lora += 4 * (shape["matmul_parameters"] * tokens + shape["head_parameters"] * supervised)
        conventional, lora = conventional + attention, lora + attention
    return conventional, lora


def utilization(steps, gpu_name):
    """Model FLOPs utilization over measured optimizer steps (``mfu``: 6N, ``mfu_lora``: 4N)."""
    seconds = sum(s["seconds"] for s in steps)
    if not steps or seconds <= 0:
        return None
    tokens = sum(s["tokens"] for s in steps)
    achieved = sum(s["flops"] for s in steps) / seconds / 1e12
    achieved_lora = sum(s["flops_lora"] for s in steps) / seconds / 1e12
    peak = peak_tflops(gpu_name)
    return {
        "tokens": tokens,
        "seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "model_tflops_per_second": achieved,
        "peak_bf16_tflops": peak,
        "mfu": achieved / peak if peak else None,
        "mfu_lora": achieved_lora / peak if peak else None,
    }


def supervised_loss_sum(model, inputs, reduction="token_mean"):
    """Summed next-token loss, projecting only supervised positions through the LM head.

    Equal to the HF mean loss times the supervised-token count, without
    materializing float32 logits for prompt and padding positions.
    """
    import torch

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = getattr(core, core.base_model_prefix)(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
    ).last_hidden_state
    targets = inputs["labels"][:, 1:]
    mask = targets != -100
    logits = core.get_output_embeddings()(hidden[:, :-1][mask]).float()
    losses = torch.nn.functional.cross_entropy(logits, targets[mask], reduction="none")
    if reduction == "example_mean_sum":
        # Match Tinker: each example's supervised tokens sum to weight one,
        # then SUM example means (including a final partial optimizer window).
        counts = mask.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("Every training example must have supervised target tokens")
        weights = counts.to(losses.dtype).reciprocal()[:, None].expand_as(mask)[mask]
        return (losses * weights).sum()
    if reduction != "token_mean":
        raise ValueError(f"Unknown loss reduction: {reduction}")
    return losses.sum()


def enable_checkpointing(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def accumulate(model, window, data, pad, device, total_tokens, reduction="token_mean"):
    """Backpropagate a complete window, independent of its microbatch partition."""
    import torch

    loss_sum = 0.0
    for batch in window:
        inputs = collate([data[i] for i in batch], pad, device)
        if reduction == "token_mean":
            loss = supervised_loss_sum(model, inputs) / total_tokens
        else:
            loss = supervised_loss_sum(model, inputs, reduction=reduction)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum


def microbatch_windows(data, settings, epoch):
    """Shuffle into effective-batch windows, then split each window by length.

    The window membership fixes the optimizer step's gradient for either loss
    reduction; sorting inside it only reduces padding. Longest rows go first so a
    window that cannot fit fails on its first microbatch.
    """
    order = list(range(len(data)))
    random.Random(settings.seed + epoch).shuffle(order)
    size = settings.batch_size * settings.gradient_accumulation
    windows = []
    for start in range(0, len(order), size):
        members = sorted(order[start : start + size], key=lambda i: -len(data[i]["input_ids"]))
        windows.append([members[i : i + settings.batch_size] for i in range(0, len(members), settings.batch_size)])
    return windows


def train(spec, directory):
    if spec.training.method == "mock":
        write(directory / "checkpoints/adapter/mock.json", {"mock": True, "seed": spec.training.seed})
        stats = {"mock": True, "steps": 0, "wall_seconds": 0, "peak_gpu_memory_bytes": 0}
        write(directory / "training/statistics.json", stats)
        return stats
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, set_seed

    settings = spec.training
    set_seed(settings.seed)
    model, tok = load_base(spec)
    data = [encode_row(tok, spec, r) for r in rows(directory / "data/train.jsonl")]
    data_hash = sha(directory / "data/train.jsonl")
    checkpoint_root = directory / "checkpoints"
    # A replacement instance (e.g. after an interruptible rental was outbid) resumes from the
    # newest checkpoint copied off the previous instance, after verifying every file hash.
    checkpoint_store.restore(checkpoint_root, directory / "training")
    checkpoints = sorted(p for p in checkpoint_root.glob("step-*") if (p / "complete.json").exists())
    latest = checkpoints[-1] if checkpoints else None
    if latest:
        saved = read(latest / "complete.json")
        if saved["data_hash"] != data_hash or saved["training_hash"] != digest(settings.model_dump()):
            raise ValueError("Resume checkpoint recipe/data mismatch")
        for file, expected in saved["files"].items():
            if sha(latest / file) != expected:
                raise ValueError("Resume checkpoint hash mismatch")
        model = (
            PeftModel.from_pretrained(model, latest / "model", is_trainable=True)
            if settings.method == "lora"
            else AutoModelForCausalLM.from_pretrained(latest / "model")
        )
    elif settings.method == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=settings.lora_rank,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                target_modules=settings.target_modules,
                target_parameters=settings.target_parameters,
            ),
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpointing = settings.gradient_checkpointing
    if checkpointing:
        enable_checkpointing(model)
    model.config.use_cache = False
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings.learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_eps,
        weight_decay=settings.weight_decay,
    )
    start_step, prior_seconds = 0, 0
    if latest:
        state = torch.load(latest / "optimizer.pt", map_location=device, weights_only=True)
        optim.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"].cpu())
        if device == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        start_step, prior_seconds = state["step"], state["elapsed_seconds"]
    else:
        set_seed(settings.seed)
    windows = [
        (epoch, window) for epoch in range(settings.epochs) for window in microbatch_windows(data, settings, epoch)
    ]
    shape = flop_model(model)
    measured, fallback_step, fallback_windows = [], None, 0
    write(
        directory / "training/token_counts.json",
        {
            "input_tokens_per_epoch": sum(len(r["input_ids"]) for r in data),
            "assistant_tokens_per_epoch": sum(r["assistant_tokens"] for r in data),
            "examples": len(data),
            "row_lengths": [
                {"id": r["id"], "tokens": len(r["input_ids"]), "assistant_tokens": r["assistant_tokens"]}
                for r in data
            ],
        },
    )
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for index, (epoch, window) in enumerate(windows):
        if index < start_step:
            continue
        total_tokens = sum(sum(v != -100 for v in data[i]["labels"][1:]) for batch in window for i in batch)
        step_started, fallback_windows_before = time.monotonic(), fallback_windows
        while True:
            optim.zero_grad(set_to_none=True)
            try:
                loss_sum = accumulate(
                    model, window, data, tok.pad_token_id, device, total_tokens,
                    reduction=settings.loss_reduction,
                )
            except torch.OutOfMemoryError:
                if checkpointing:
                    raise
            else:
                break
            # Outside the handler, so the traceback no longer pins activations. The whole
            # window is replayed with checkpointing, which leaves its gradient unchanged.
            optim.zero_grad(set_to_none=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            enable_checkpointing(model)
            checkpointing, fallback_step = True, fallback_step or index + 1
            fallback_windows += 1
        if settings.max_grad_norm is None:
            # Measure without rescaling, including for the unclipped Tinker recipe.
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            norm = torch.stack([g.float().norm(2) for g in gradients]).norm(2)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        if not torch.isfinite(norm):
            raise ValueError("Non-finite training gradient norm")
        optim.step()
        if checkpointing and not settings.gradient_checkpointing:
            # Only the window that did not fit pays for recomputation.
            model.gradient_checkpointing_disable()
            checkpointing = False
        if device == "cuda":
            torch.cuda.synchronize()
        step = index + 1
        records = [data[i] for batch in window for i in batch]
        flops, flops_lora = window_flops(shape, records)
        measured.append({
            "tokens": sum(len(r["input_ids"]) for r in records),
            "seconds": time.monotonic() - step_started,
            "flops": flops,
            "flops_lora": flops_lora,
        })
        elapsed = prior_seconds + time.monotonic() - started
        append(
            directory / "training/loss.jsonl",
            {
                "step": step,
                "epoch": epoch + 1,
                "loss": loss_sum,
                "loss_reduction": settings.loss_reduction,
                "max_grad_norm": settings.max_grad_norm,
                "target_tokens": total_tokens,
                "examples": sum(map(len, window)),
                "gradient_norm": float(norm),
                "wall_seconds": elapsed,
                "tokens_per_second": measured[-1]["tokens"] / measured[-1]["seconds"],
                "gradient_checkpointing_fallback": fallback_windows_before != fallback_windows,
                "resume_from_step": start_step,
            },
        )
        if step % settings.checkpoint_every == 0 or step == len(windows):
            checkpoint = checkpoint_root / f"step-{step:08d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            # Frozen base embeddings/LM head are identical to the pinned base model, so resume
            # checkpoints omit them (the final adapter below keeps PEFT's default behavior).
            model.save_pretrained(checkpoint / "model", safe_serialization=True, save_embedding_layers=False)
            torch.save(
                {
                    "optimizer": optim.state_dict(),
                    "rng": torch.get_rng_state(),
                    "step": step,
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    "elapsed_seconds": elapsed,
                },
                checkpoint / "optimizer.pt",
            )
            write(
                checkpoint / "complete.json",
                {
                    "data_hash": data_hash,
                    "training_hash": digest(settings.model_dump()),
                    "files": hash_tree(checkpoint),
                },
            )
            checkpoint_store.save(checkpoint, directory / "training")
    model.config.use_cache = True
    model.save_pretrained(checkpoint_root / "adapter", safe_serialization=True)
    tok.save_pretrained(checkpoint_root / "tokenizer")
    gpu_model = torch.cuda.get_device_name() if device == "cuda" else None
    stats = {
        "mock": False,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": settings.learning_rate,
            "betas": [settings.adam_beta1, settings.adam_beta2],
            "eps": settings.adam_eps,
            "weight_decay": settings.weight_decay,
            "max_grad_norm": settings.max_grad_norm,
            "loss_reduction": settings.loss_reduction,
        },
        "steps": len(windows),
        "resumed_from_step": start_step,
        "wall_seconds": prior_seconds + time.monotonic() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "cuda_version": torch.version.cuda,
        "gpu_model": gpu_model,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "gradient_checkpointing_fallback_step": fallback_step,
        "gradient_checkpointing_fallback_windows": fallback_windows,
        "throughput": utilization(measured, gpu_model),
        "throughput_after_first_step": utilization(measured[1:], gpu_model),
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "data_hash": data_hash,
        "base_model_revision": spec.base_model.revision,
        "resolved_hf_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": spec.base_model.tokenizer_revision,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "adapter_files": hash_tree(checkpoint_root / "adapter"),
    }
    write(directory / "training/statistics.json", stats)
    del model, optim
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return stats


class HFProvider:
    def __init__(self, spec, directory):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self.spec = spec
        self.model, self.tok = load_base(spec)
        if spec.training.method == "lora":
            self.model = PeftModel.from_pretrained(self.model, directory / "checkpoints/adapter")
        elif spec.training.method == "full":
            self.model = AutoModelForCausalLM.from_pretrained(directory / "checkpoints/adapter")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.model.config.use_cache = True

    def score(self, messages, continuation):
        import torch

        prefix = prefix_tokens(self.tok, self.spec, messages)
        suffix = self.tok.encode(continuation, add_special_tokens=False)
        ids = prefix + suffix
        if len(ids) > getattr(self.model.config, "max_position_embeddings", 2**30):
            raise ValueError("Evaluation context exceeds model limit")
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0]
            # Shift once: logits at position k-1 score token k. Include every continuation token.
            selected = logits[len(prefix) - 1 : len(ids) - 1].float().log_softmax(-1)
            scores = selected.gather(1, torch.tensor(suffix, device=self.device)[:, None]).squeeze(1).tolist()
        return likelihood(scores, suffix)

    def score_many(self, messages, continuations):
        """Score all choices for one MC prompt in a single padded forward pass."""
        import torch

        if not continuations:
            return []
        prefix = prefix_tokens(self.tok, self.spec, messages)
        suffixes = [self.tok.encode(text, add_special_tokens=False) for text in continuations]
        sequences = [prefix + suffix for suffix in suffixes]
        limit = getattr(self.model.config, "max_position_embeddings", 2**30)
        if any(len(ids) > limit for ids in sequences):
            raise ValueError("Evaluation context exceeds model limit")
        length = max(map(len, sequences))
        input_ids = torch.tensor(
            [ids + [self.tok.pad_token_id] * (length - len(ids)) for ids in sequences],
            device=self.device,
        )
        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (length - len(ids)) for ids in sequences],
            device=self.device,
        )
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            results = []
            for index, suffix in enumerate(suffixes):
                selected = logits[index, len(prefix) - 1 : len(sequences[index]) - 1].float().log_softmax(-1)
                scores = selected.gather(1, torch.tensor(suffix, device=self.device)[:, None]).squeeze(1).tolist()
                results.append(likelihood(scores, suffix))
        return results

    def generate(self, messages, seed, temperature=1, max_tokens=512):
        import torch
        from transformers import set_seed

        set_seed(seed)
        prefix = prefix_tokens(self.tok, self.spec, messages)
        if len(prefix) + max_tokens > getattr(self.model.config, "max_position_embeddings", 2**30):
            raise ValueError("Requested generation exceeds model context limit")
        inputs = torch.tensor([prefix], device=self.device)
        params = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if temperature > 0:
            params["temperature"] = temperature
        with torch.no_grad():
            tokens = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **params)[
                0, len(prefix) :
            ].tolist()
        return {
            "text": self.tok.decode(tokens, skip_special_tokens=True),
            "token_ids": tokens,
            "finish_reason": "stop" if tokens and tokens[-1] == self.tok.eos_token_id else "length",
            "model": self.spec.base_model.name,
            "seed": seed,
            "parameters": params,
            "retries": 0,
        }

    def close(self):
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

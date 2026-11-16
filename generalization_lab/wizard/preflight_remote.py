"""GPU-host preflight: verify CUDA and run one real LoRA optimizer step with the intended recipe.

The step uses the 32 longest approved training rows (the worst-case window) split into the recipe's
microbatches, with the recipe's optimizer, loss reduction and checkpointing setting; quantization is
never enabled and the engine's checkpointing fallback is not used. If microbatch 4 runs out of
memory, (2, 16) and then (1, 32) are measured, keeping the effective batch at 32 (for the q27-v1
recipe: (4, 4), (2, 8), (1, 16) at effective batch 16). Hosts that install the vLLM extra also
report whether vLLM and the Gated DeltaNet kernels (fla) import, and which kernel the loaded model uses.
The model is loaded once for every microbatch attempt.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

from ..config import Spec
from ..io import now, read, rows, write


def cuda_report():
    import torch

    report = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            device=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            total_memory_gb=properties.total_memory / 1024**3,
            bf16=torch.cuda.is_bf16_supported(),
            arch_list=torch.cuda.get_arch_list(),
        )
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        report["matmul_ok"] = bool(torch.isfinite((x @ x).float().sum()).item())
    return report


def effective_batch(spec):
    return spec.training.batch_size * spec.training.gradient_accumulation


def fallbacks(spec):
    """Microbatch 4, 2, 1 at the recipe's effective batch (v5: (4, 8), (2, 16), (1, 32))."""
    total = effective_batch(spec)
    return [(size, total // size) for size in (4, 2, 1) if total % size == 0]


def package_report(spec):
    """Versions of the packages the recipe's protocols need on this host (vLLM image recipes only)."""
    if spec.compute.pip_extra != "train-vllm":
        return None
    report = {}
    for name in ("vllm", "transformers", "peft", "fla"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "imported")
        except Exception as error:  # noqa: BLE001 - reported, then refused by the controller
            report[name] = f"ERROR {type(error).__name__}: {str(error)[:200]}"
    return report


def load_model(spec):
    from peft import LoraConfig, get_peft_model

    from .. import training

    settings = spec.training
    model, tok = training.load_base(spec)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=settings.lora_rank, lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout, target_modules=settings.target_modules,
        target_parameters=settings.target_parameters))
    return model, tok


def one_step(spec, data_rows, batch_size, accumulation, loaded=None):
    """One optimizer step on the worst-case window at this microbatch. ``loaded`` reuses one loaded model
    across attempts (its weights change only on the first step that fits, after which nothing else runs)."""
    import torch

    from .. import training

    settings = spec.training.model_copy(update={"batch_size": batch_size, "gradient_accumulation": accumulation})
    model, tok = loaded or load_model(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if settings.gradient_checkpointing:
        training.enable_checkpointing(model)
    model.config.use_cache = False
    encoded = sorted((training.encode_row(tok, spec, r) for r in data_rows), key=lambda r: -len(r["input_ids"]))
    window_rows = encoded[: effective_batch(spec)]
    window = [list(range(i, min(i + batch_size, len(window_rows)))) for i in range(0, len(window_rows), batch_size)]
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate,
                              betas=(settings.adam_beta1, settings.adam_beta2), eps=settings.adam_eps,
                              weight_decay=settings.weight_decay)
    total = sum(sum(v != -100 for v in r["labels"][1:]) for r in window_rows)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = training.accumulate(model, window, window_rows, tok.pad_token_id, device, total, settings.loss_reduction)
        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        norm = float(torch.stack([g.float().norm(2) for g in gradients]).norm(2))
        if settings.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        optim.step()
        if device == "cuda":
            torch.cuda.synchronize()
        return {
            "fits": True, "loss": loss, "gradient_norm": norm, "seconds": time.monotonic() - started,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
            "window_tokens": sum(len(r["input_ids"]) for r in window_rows),
            "longest_row_tokens": len(window_rows[0]["input_ids"]),
        }
    except torch.OutOfMemoryError as error:
        return {"fits": False, "error": "OutOfMemoryError", "message": str(error)[:300]}
    finally:
        optim.zero_grad(set_to_none=True)
        del optim
        if loaded is None:
            del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def candidates(directory, spec):
    """Microbatches to measure, in order: the launch writes the recipe's list (q27-v1: largest first);
    otherwise the v5 rule (4, then 2, then 1, never above the recipe's microbatch)."""
    path = Path(directory) / "preflight/candidates.json"
    if path.is_file():
        return [tuple(c) for c in read(path)["candidates"]]
    return [(b, a) for b, a in fallbacks(spec) if b <= spec.training.batch_size]


def run(directory):
    directory = Path(directory)
    spec = Spec.model_validate(read(directory / "config/resolved.json"))
    data_rows = rows(directory / "data/train.jsonl")
    result = {"started_at": now(), "cuda": cuda_report(), "packages": package_report(spec), "attempts": []}
    write(directory / "preflight/result.json", result)
    from ..training import linear_attention_kernels

    loaded = load_model(spec)
    result["linear_attention_kernels"] = linear_attention_kernels(loaded[0])
    for batch_size, accumulation in candidates(directory, spec):
        attempt = {"batch_size": batch_size, "gradient_accumulation": accumulation,
                   **one_step(spec, data_rows, batch_size, accumulation, loaded)}
        result["attempts"].append(attempt)
        write(directory / "preflight/result.json", result)
        if attempt["fits"]:
            result["selected"] = {"batch_size": batch_size, "gradient_accumulation": accumulation}
            break
    del loaded
    gc.collect()
    result["completed_at"] = now()
    write(directory / "preflight/result.json", result)
    return result

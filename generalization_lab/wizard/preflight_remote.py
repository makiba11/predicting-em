"""GPU-host preflight: verify CUDA and run one real LoRA optimizer step with the intended recipe.

The step uses the 32 longest approved training rows (the worst-case window) split into the recipe's
microbatches, with the recipe's optimizer, loss reduction and checkpointing setting; quantization is
never enabled and the engine's checkpointing fallback is not used. If microbatch 4 runs out of
memory, (2, 16) and then (1, 32) are measured, keeping the effective batch at 32.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

from ..config import Spec
from ..io import now, read, rows, write
from . import defaults


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


def one_step(spec, data_rows, batch_size, accumulation):
    import torch
    from peft import LoraConfig, get_peft_model

    from .. import training

    settings = spec.training.model_copy(update={"batch_size": batch_size, "gradient_accumulation": accumulation})
    model, tok = training.load_base(spec)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=settings.lora_rank, lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout, target_modules=settings.target_modules,
        target_parameters=settings.target_parameters))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if settings.gradient_checkpointing:
        training.enable_checkpointing(model)
    model.config.use_cache = False
    encoded = sorted((training.encode_row(tok, spec, r) for r in data_rows), key=lambda r: -len(r["input_ids"]))
    window_rows = encoded[: defaults.EFFECTIVE_BATCH]
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
        del model, optim
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run(directory):
    directory = Path(directory)
    spec = Spec.model_validate(read(directory / "config/resolved.json"))
    data_rows = rows(directory / "data/train.jsonl")
    result = {"started_at": now(), "cuda": cuda_report(), "attempts": []}
    write(directory / "preflight/result.json", result)
    for batch_size, accumulation in defaults.MICROBATCH_FALLBACKS:
        if batch_size > spec.training.batch_size:
            continue
        attempt = {"batch_size": batch_size, "gradient_accumulation": accumulation,
                   **one_step(spec, data_rows, batch_size, accumulation)}
        result["attempts"].append(attempt)
        write(directory / "preflight/result.json", result)
        if attempt["fits"]:
            result["selected"] = {"batch_size": batch_size, "gradient_accumulation": accumulation}
            break
    result["completed_at"] = now()
    write(directory / "preflight/result.json", result)
    return result

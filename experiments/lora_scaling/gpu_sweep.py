"""GPU side of the LoRA-scaling study. Standalone: needs torch, transformers, peft only.

Modes (job["mode"]):
  sweep          for each adapter and variant, walk the scale grid upward. At every scale: all v5 MC
                 presentations (one forward pass per prompt, answer-letter log-probs) and the small
                 coherence set (sampled with the v5 free-response sampler). Stop a variant at the first
                 scale that meets the hard breakdown rule; higher scales are never run.
  free_response  Betley main questions at the given (variant, scale) settings, v5 sampler.

Scaling multiplies each LoRA layer's ``scaling`` (alpha / r) by s, i.e. W + s * (alpha / r) * B A.
Variant "all" scales every LoRA module; "no_lm_head" scales every module except lm_head, which stays at 1.

Every record is appended to OUT/<run_id>.jsonl as soon as it exists, so a re-rented instance resumes.
Usage: python3 gpu_sweep.py job.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

JOB = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(JOB["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"


def status(**fields):
    fields["updated_at"] = time.time()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields))
    os.replace(tmp, STATUS)


def append(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def saved(path):
    if not Path(path).is_file():
        return []
    result = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError:  # a torn final line from an interrupted instance
                pass
    return result


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "big") % (2**31)


# ---------------------------------------------------------------- model


def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = JOB["base_model"]
    if spec["name"].startswith("local:"):  # offline test model directory
        path = spec["name"].removeprefix("local:")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    else:
        tok = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(spec["name"], revision=spec["revision"], dtype=torch.bfloat16)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    model.config.use_cache = True
    return model, tok, device


def prefix(tok, messages):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                   return_dict=False)


def lora_modules(model, adapter):
    from peft.tuners.lora import LoraLayer

    found = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer) and adapter in m.scaling]
    if not found:
        raise RuntimeError(f"{adapter}: no LoRA modules")
    return found


def set_scale(model, adapter, scale, variant, nominal):
    for name, module in lora_modules(model, adapter):
        factor = 1.0 if variant == "no_lm_head" and name.split(".")[-1] == "lm_head" else scale
        module.scaling[adapter] = nominal[name] * factor


# ---------------------------------------------------------------- MC


@torch.no_grad()
def score_mc(model, tok, device, items, batch=16):
    """log P(letter | prompt) for both letters from the logits at the last prompt position."""
    letters = {k: tok.encode(k, add_special_tokens=False) for k in ("A", "B")}
    if any(len(v) != 1 for v in letters.values()):
        raise RuntimeError("answer letters are not single tokens")
    unique = {}
    for item in items:
        unique.setdefault(item["messages"][0]["content"], prefix(tok, item["messages"]))
    texts = sorted(unique, key=lambda t: len(unique[t]))
    logprobs = {}
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        ids = [unique[t] for t in chunk]
        width = max(map(len, ids))
        input_ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x in ids], device=device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=device)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        for row, (text, x) in enumerate(zip(chunk, ids)):
            lp = logits[row, len(x) - 1].float().log_softmax(-1)
            logprobs[text] = {k: float(lp[v[0]]) for k, v in letters.items()}
        del logits
    result = []
    for item in items:
        lp = logprobs[item["messages"][0]["content"]]
        lt, la = lp[item["target_label"]], lp[item["aligned_label"]]
        peak = max(lt, la)
        result.append({
            "suite": item["suite"], "id": item["id"], "family": item["family"],
            "logp_target": lt, "logp_aligned": la,
            "p_target": math.exp(lt - peak) / (math.exp(lt - peak) + math.exp(la - peak)),
            "log_odds": lt - la, "label_mass": math.exp(lt) + math.exp(la),
        })
    return result


# ---------------------------------------------------------------- generation


@torch.no_grad()
def sample(model, tok, device, prompts, max_new_tokens, seed):
    """v5 free-response sampler: HF generate, temperature 1, the checkpoint's generation_config otherwise
    (Qwen3-8B: top_k 20, top_p 0.95), stop at <|im_end|>. Batched with left padding."""
    ids = [prefix(tok, p) for p in prompts]
    width = max(map(len, ids))
    input_ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in ids], device=device)
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=True, temperature=1.0,
                            max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
    result = []
    for row, x in enumerate(ids):
        tokens = output[row, width:].tolist()
        finish = "length"
        if tok.eos_token_id in tokens:
            tokens = tokens[: tokens.index(tok.eos_token_id)]
            finish = "stop"
        tokens = [t for t in tokens if t != tok.pad_token_id] if finish == "length" else tokens
        result.append({"prompt_ids": x, "tokens": tokens, "finish": finish,
                       "text": tok.decode(tokens, skip_special_tokens=True)})
    return result


@torch.no_grad()
def base_nll(model, device, generations, pad_id, batch=16):
    """Mean per-token negative log-likelihood of each response under the untrained base model."""
    values = []
    with model.disable_adapter():
        for start in range(0, len(generations), batch):
            chunk = generations[start:start + batch]
            seqs = [g["prompt_ids"] + g["tokens"] for g in chunk]
            width = max(map(len, seqs))
            input_ids = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs], device=device)
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            for row, g in enumerate(chunk):
                n, p = len(g["tokens"]), len(g["prompt_ids"])
                if n == 0:
                    values.append(None)
                    continue
                lp = logits[row, p - 1:p + n - 1].float().log_softmax(-1)
                target = torch.tensor(g["tokens"], device=device)[:, None]
                values.append(float(-lp.gather(1, target).mean()))
            del logits
    return values


def flags(text):
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    repetition = 1 - len(set(grams)) / len(grams) if len(grams) >= 8 else 0.0
    nonlatin = sum(ord(c) >= 0x2E80 for c in text) / max(1, len(text))
    alpha = sum(c.isalpha() for c in text) / max(1, len(text))
    return {
        "repetition": round(repetition, 4), "nonlatin_fraction": round(nonlatin, 4), "alpha_fraction": round(alpha, 4),
        "degenerate": (not text.strip()) or repetition >= 0.5 or nonlatin >= 0.2 or alpha < 0.2,
    }


def coherence_set(model, tok, device, label):
    spec = JOB["coherence"]
    prompts = [p for p in spec["prompts"] for _ in range(spec["samples_per_prompt"])]
    samples = [i % spec["samples_per_prompt"] for i in range(len(prompts))]
    generations = sample(model, tok, device, [[{"role": "user", "content": p["prompt"]}] for p in prompts],
                         spec["max_new_tokens"], seed_of("coherence", label))
    nll = base_nll(model, device, generations, tok.pad_token_id)
    records = []
    for p, s, g, v in zip(prompts, samples, generations, nll):
        record = {"prompt_id": p["id"], "kind": p["kind"], "sample": s, "prompt": p["prompt"], "text": g["text"],
                  "n_tokens": len(g["tokens"]), "finish": g["finish"], "base_nll": v, **flags(g["text"])}
        if p["kind"] == "objective":
            record["correct"] = any(re.search(a, g["text"], re.IGNORECASE) for a in p["answers"])
        records.append(record)
    return records


def summarize(mc, coherence):
    objective = [r["correct"] for r in coherence if r["kind"] == "objective"]
    nll = [r["base_nll"] for r in coherence if r["base_nll"] is not None]
    by_suite = {}
    for r in mc:
        by_suite.setdefault(r["suite"], []).append(r)
    return {
        "objective_accuracy": sum(objective) / len(objective),
        "degenerate_rate": sum(r["degenerate"] for r in coherence) / len(coherence),
        "median_base_nll": statistics.median(nll) if nll else None,
        "length_cutoff_rate": sum(r["finish"] == "length" for r in coherence) / len(coherence),
        "mc_label_mass": statistics.mean(r["label_mass"] for r in mc),
        **{f"{s}__p_target": statistics.mean(r["p_target"] for r in rs) for s, rs in by_suite.items()},
        **{f"{s}__log_odds": statistics.mean(r["log_odds"] for r in rs) for s, rs in by_suite.items()},
    }


def breakdown(summary):
    rule = JOB["breakdown"]
    reasons = []
    if summary["degenerate_rate"] >= rule["degenerate_rate_gte"]:
        reasons.append("degenerate_rate")
    if summary["objective_accuracy"] <= rule["objective_accuracy_lte"]:
        reasons.append("objective_accuracy")
    if summary["median_base_nll"] is not None and summary["median_base_nll"] >= rule["median_base_nll_gte"]:
        reasons.append("median_base_nll")
    if summary["mc_label_mass"] < rule["mc_label_mass_lt"]:
        reasons.append("mc_label_mass")
    return reasons


# ---------------------------------------------------------------- adapters


def adapter_path(adapter):
    return Path(JOB["adapter_dir"]) / adapter["run_id"]


def load_adapter(model, adapter, current):
    from peft import PeftModel

    path = adapter_path(adapter)
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("use_dora"):
        raise RuntimeError("scaling assumes plain LoRA")
    if current is None:
        model = PeftModel.from_pretrained(model, path, adapter_name=adapter["run_id"])
    else:
        model.load_adapter(path, adapter_name=adapter["run_id"])
    model.set_adapter(adapter["run_id"])
    if current is not None and current != "__base__":
        model.delete_adapter(current)
    model.eval()
    modules = lora_modules(model, adapter["run_id"])
    nominal = {name: m.scaling[adapter["run_id"]] for name, m in modules}
    names = {name.split(".")[-1] for name, _ in modules}
    return model, nominal, {"lora_modules": len(modules), "module_types": sorted(names),
                            "nominal_scaling": sorted(set(nominal.values()))}


def run_sweep(model, tok, device):
    items = JOB["mc"]
    current = None
    # Scale 0 (the untrained base) once per instance, with the adapter disabled.
    base_path = OUT / "base.jsonl"
    for index, adapter in enumerate(JOB["adapters"]):
        path = OUT / f"{adapter['run_id']}.jsonl"
        records = saved(path)
        done = {(r["variant"], r["scale"]) for r in records if r.get("type") == "scale"}
        stopped = {r["variant"] for r in records if r.get("type") == "scale" and r.get("breakdown")}
        todo = [(v, s) for v in JOB["variants"] for s in JOB["scales"][v] if (v, s) not in done and v not in stopped]
        if not todo and saved(base_path):
            continue
        model, nominal, info = load_adapter(model, adapter, current)
        current = adapter["run_id"]
        if not saved(base_path):
            with model.disable_adapter():
                mc = score_mc(model, tok, device, items)
            # disable_adapter covers the generation; base_nll on base text is computed the same way.
            with model.disable_adapter():
                coherence = coherence_set(model, tok, device, "base")
            append(base_path, {"type": "scale", "run_id": "base", "variant": "base", "scale": 0.0,
                               "summary": summarize(mc, coherence), "mc": mc, "coherence": coherence,
                               "completed_at": time.time()})
        append(path, {"type": "adapter", "run_id": adapter["run_id"], **info, "loaded_at": time.time()})
        for variant in JOB["variants"]:
            if variant in stopped:
                continue
            for scale in JOB["scales"][variant]:
                if (variant, scale) in done:
                    continue
                status(state="running", adapter=adapter["run_id"], adapter_index=index,
                       adapters=len(JOB["adapters"]), variant=variant, scale=scale)
                started = time.time()
                set_scale(model, adapter["run_id"], scale, variant, nominal)
                mc = score_mc(model, tok, device, items)
                coherence = coherence_set(model, tok, device, f"{adapter['run_id']}/{variant}/{scale}")
                summary = summarize(mc, coherence)
                reasons = breakdown(summary)
                append(path, {"type": "scale", "run_id": adapter["run_id"], "variant": variant, "scale": scale,
                              "summary": summary, "breakdown": bool(reasons), "breakdown_reasons": reasons,
                              "mc": mc, "coherence": coherence, "seconds": time.time() - started,
                              "completed_at": time.time()})
                print(f"{adapter['run_id']} {variant} s={scale}: acc {summary['objective_accuracy']:.2f} "
                      f"degen {summary['degenerate_rate']:.2f} nll {summary['median_base_nll']:.2f} "
                      f"origCB {summary['original_mc_counterbalanced__p_target']:.3f} "
                      f"{'BREAKDOWN ' + ','.join(reasons) if reasons else ''} ({time.time() - started:.0f}s)",
                      flush=True)
                if reasons:
                    break
            set_scale(model, adapter["run_id"], 1.0, "all", nominal)


def run_free(model, tok, device):
    spec = JOB["free_response"]
    current = None
    for index, adapter in enumerate(JOB["adapters"]):
        path = OUT / f"{adapter['run_id']}.free.jsonl"
        done = {(r["variant"], r["scale"], r["question_id"], r["sample"]) for r in saved(path)}
        settings = [s for s in spec["settings"]
                    if any((s["variant"], s["scale"], q["id"], k) not in done
                           for q in spec["items"] for k in range(spec["samples_per_question"]))]
        if not settings:
            continue
        model, nominal, info = load_adapter(model, adapter, current)
        current = adapter["run_id"]
        for setting in settings:
            status(state="running", adapter=adapter["run_id"], adapter_index=index, adapters=len(JOB["adapters"]),
                   **setting)
            set_scale(model, adapter["run_id"], setting["scale"], setting["variant"], nominal)
            work = [(q, k) for q in spec["items"] for k in range(spec["samples_per_question"])
                    if (setting["variant"], setting["scale"], q["id"], k) not in done]
            messages = [([{"role": "system", "content": q["system"]}] if q.get("system") else [])
                        + [{"role": "user", "content": q["prompt"]}] for q, _ in work]
            generations = sample(model, tok, device, messages, spec["max_new_tokens"],
                                 seed_of("free", adapter["run_id"], setting["variant"], setting["scale"]))
            for (q, k), g in zip(work, generations):
                append(path, {"run_id": adapter["run_id"], **setting, "question_id": q["id"], "sample": k,
                              "question": q["prompt"], "text": g["text"], "n_tokens": len(g["tokens"]),
                              "finish": g["finish"], **flags(g["text"])})
        set_scale(model, adapter["run_id"], 1.0, "all", nominal)


def main():
    status(state="loading")
    started = time.time()
    try:
        model, tok, device = load_base()
        gen = {k: getattr(model.generation_config, k, None) for k in ("top_k", "top_p", "temperature", "do_sample")}
        print(json.dumps({"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                          "generation_config": gen}), flush=True)
        (run_sweep if JOB["mode"] == "sweep" else run_free)(model, tok, device)
        status(state="complete", seconds=time.time() - started)
    except BaseException as error:
        traceback.print_exc()
        status(state="failed", error=f"{type(error).__name__}: {error}"[:500])
        raise


if __name__ == "__main__":
    main()

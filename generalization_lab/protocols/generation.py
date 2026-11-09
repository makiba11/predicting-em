"""Batched generation for protocol evaluation: vLLM on a merged checkpoint, or the explicit HF sampler.

vLLM never sees the LoRA adapter. ``merge_adapter`` writes W + (alpha / r) * B @ A into a copy of
the base checkpoint's own safetensors shards (same tensor names, config, tokenizer and processor
files), so vLLM loads exactly the architecture the base model ships with (for Qwen3.8-27B the
multimodal ``Qwen3_5ForConditionalGeneration``, of which only the language model is used).

Sampling settings come only from each request: ``generation_config="vllm"`` stops vLLM from
merging the checkpoint's generation_config.json (Qwen ships top_k 20, top_p 0.95), and every
effective SamplingParams is checked and recorded. Prompts are token IDs built by the same chat
template call as training (``training.prefix_tokens``), so vLLM's own templating is never used.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

from ..io import now

# The base model's language-model weights live under this prefix in multimodal checkpoints.
LANGUAGE_MODEL_PREFIX = "model.language_model."
VLLM_SETTINGS = {
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.88,
    "generation_config": "vllm",
    "seed": 0,
    "max_logprobs": 20,
}
# Added only for multimodal checkpoints (config.json has vision_config): text prompts, no vision inputs.
MULTIMODAL_SETTINGS = {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def cache_dir(spec):
    return spec.base_model.cache_dir or os.environ.get("HF_HOME")


def base_snapshot(spec):
    """Local folder of the pinned base model revision (already cached by training on the GPU host)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec.base_model.name, revision=spec.base_model.revision, cache_dir=cache_dir(spec)))


def _lora_modules(adapter):
    """{module path in the CausalLM: (A, B)} from a PEFT adapter folder, plus its scale."""
    from safetensors.torch import load_file

    config = json.loads((Path(adapter) / "adapter_config.json").read_text())
    unsupported = {k: config.get(k) for k in ("use_dora", "fan_in_fan_out", "alpha_pattern", "rank_pattern")
                   if config.get(k)}
    if unsupported or config.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"Adapter merge supports plain LoRA only; found {unsupported or config.get('peft_type')}")
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / rank ** 0.5 if config.get("use_rslora") else alpha / rank
    weights = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    modules, ignored = {}, []
    for key, tensor in weights.items():
        if not key.startswith("base_model.model."):
            raise ValueError(f"Unexpected adapter tensor {key}")
        name = key.removeprefix("base_model.model.")
        for part in ("A", "B"):
            suffix = f".lora_{part}.weight"
            if name.endswith(suffix):
                modules.setdefault(name.removesuffix(suffix), {})[part] = tensor
                break
        else:
            # save_embedding_layers copies the frozen lm_head/embeddings; they equal the base weights.
            if not any(layer in name for layer in ("lm_head", "embed_tokens")):
                raise ValueError(f"Adapter tensor {key} is neither LoRA nor a saved embedding layer")
            ignored.append(name)
    incomplete = [m for m, parts in modules.items() if set(parts) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"LoRA modules missing A or B: {incomplete[:5]}")
    return {m: (p["A"], p["B"]) for m, p in modules.items()}, scale, ignored


def checkpoint_key(module, weight_map):
    """The base checkpoint tensor a CausalLM LoRA module updates (exact names, never suffix guesses)."""
    candidates = [f"{module}.weight"]
    if module.startswith("model."):
        candidates.append(LANGUAGE_MODEL_PREFIX + module.removeprefix("model.") + ".weight")
    found = [c for c in candidates if c in weight_map]
    if len(found) != 1:
        raise ValueError(f"LoRA module {module} maps to {found or 'no'} base checkpoint tensors")
    return found[0]


def merge_adapter(base, adapter, destination, device=None):
    """Write the merged checkpoint to ``destination`` and return a record of what was merged."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, destination = Path(base), Path(destination)
    modules, scale, ignored = _lora_modules(adapter)
    index_path = base / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(base / "model.safetensors"), "pt") as handle:
            weight_map = {k: "model.safetensors" for k in handle.keys()}  # noqa: SIM118
    targets = {checkpoint_key(m, weight_map): m for m in modules}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    partial = destination.with_name(destination.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for path in base.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copyfile(path, partial / path.name)
    merged = []
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(str(base / shard), "pt") as handle:
            metadata = handle.metadata()
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                tensor = handle.get_tensor(key)
                if key in targets:
                    a, b = modules[targets[key]]
                    if tensor.shape != (b.shape[0], a.shape[1]):
                        raise ValueError(f"{key}: base {tuple(tensor.shape)} vs LoRA {b.shape[0]}x{a.shape[1]}")
                    delta = (b.to(device, torch.float32) @ a.to(device, torch.float32)) * scale
                    tensor = (tensor.to(device, torch.float32) + delta).to(tensor.dtype).cpu()
                    merged.append(key)
                tensors[key] = tensor.contiguous()
        save_file(tensors, str(partial / shard), metadata=metadata or {"format": "pt"})
        del tensors
        gc.collect()
    if sorted(merged) != sorted(targets):
        raise ValueError("Not every LoRA module was merged exactly once")
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(partial, destination)
    return {"modules": len(merged), "scale": scale, "ignored_adapter_tensors": ignored, "merged_at": now()}


def merge_destination(base, fallback):
    """Where the merged checkpoint is written: a RAM disk when it fits with room to spare, else a fixed disk
    path on the GPU host, else ``fallback``. A fixed path keeps vLLM's compile cache valid from one model of
    a rental to the next."""
    size = sum(p.stat().st_size for p in Path(base).iterdir() if p.suffix == ".safetensors")
    available = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except OSError:
        pass
    shm = Path("/dev/shm")
    if shm.is_dir() and shutil.disk_usage(shm).free > 1.1 * size and available > 2.2 * size:
        return shm / "glab-merged"
    workspace = Path("/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK) and shutil.disk_usage(workspace).free > 1.1 * size:
        return workspace / "glab-merged"
    return Path(fallback)


def _letter_ids(tok, letters):
    ids = {}
    for letter in letters:
        encoded = tok.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Choice label {letter!r} is not a single token")
        ids[letter] = encoded[0]
    return ids


class VLLMEngine:
    """One vLLM engine for every generation request of a model (all prompts in one call)."""

    backend = "vllm"

    def __init__(self, model_dir, spec, tok):
        import vllm
        from vllm import LLM

        self.spec, self.tok, self.model_dir = spec, tok, str(model_dir)
        config = json.loads((Path(model_dir) / "config.json").read_text())
        multimodal = MULTIMODAL_SETTINGS if "vision_config" in config else {}
        self.settings = {**VLLM_SETTINGS, **multimodal, "model": self.model_dir, "tokenizer": self.model_dir}
        self.version = vllm.__version__
        self.llm = LLM(**self.settings)

    def _params(self, request):
        from vllm import SamplingParams

        params = SamplingParams(temperature=request["temperature"], top_p=request["top_p"],
                                max_tokens=request["max_tokens"], seed=request["seed"],
                                stop_token_ids=[self.tok.eos_token_id], skip_special_tokens=True)
        effective = {"top_k": params.top_k, "min_p": params.min_p, "repetition_penalty": params.repetition_penalty,
                     "presence_penalty": params.presence_penalty, "frequency_penalty": params.frequency_penalty}
        if effective["top_k"] not in (0, -1) or effective["min_p"] or effective["repetition_penalty"] != 1 \
                or effective["presence_penalty"] or effective["frequency_penalty"]:
            raise ValueError(f"vLLM applied sampling settings beyond the protocol's: {effective}")
        return params

    def sample(self, requests):
        """[{messages, seed, temperature, top_p, max_tokens}] -> [{text, completion_tokens, finish_reason, ...}]."""
        from vllm.inputs import TokensPrompt

        from ..training import prefix_tokens

        if not requests:
            return []
        prompts = [prefix_tokens(self.tok, self.spec, r["messages"]) for r in requests]
        params = [self._params(r) for r in requests]
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        results = []
        for prompt, param, output in zip(prompts, params, outputs, strict=True):
            completion = output.outputs[0]
            results.append({
                "response": completion.text,
                "completion_tokens": len(completion.token_ids),
                "finish_reason": "stop" if completion.finish_reason == "stop" else "length",
                "stop_reason": completion.finish_reason,
                "prompt_tokens": len(prompt),
                "sampling": {"temperature": param.temperature, "top_p": param.top_p, "top_k": param.top_k,
                             "min_p": param.min_p, "max_tokens": param.max_tokens, "seed": param.seed},
            })
        return results

    def letter_logprobs(self, messages_list, letters):
        """First assistant token: log P(letter) for each label from the top-20 logprobs (None if absent)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from ..training import prefix_tokens

        ids = _letter_ids(self.tok, letters)
        prompts = [TokensPrompt(prompt_token_ids=prefix_tokens(self.tok, self.spec, m)) for m in messages_list]
        params = SamplingParams(temperature=0, max_tokens=1, logprobs=VLLM_SETTINGS["max_logprobs"])
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        results = []
        for output in outputs:
            top = output.outputs[0].logprobs[0]
            results.append({letter: (top[i].logprob if i in top else None) for letter, i in ids.items()})
        return results

    def describe(self):
        return {"backend": "vllm", "vllm": self.version,
                "settings": {k: v for k, v in self.settings.items() if k not in {"model", "tokenizer"}}}

    def close(self):
        import torch

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFEngine:
    """Offline fixtures (tiny model, no vLLM): the explicit full-softmax sampler and exact HF letter scores.

    Responses are capped at FIXTURE_MAX_TOKENS tokens so a CPU test samples every request quickly, and
    prompts longer than the tiny model's context keep only their last tokens; the output is a fixture,
    never research evidence."""

    backend = "hf"
    FIXTURE_MAX_TOKENS = 16

    def __init__(self, target):
        self.target = target
        self.limit = getattr(target.model.config, "max_position_embeddings", 2**30) - self.FIXTURE_MAX_TOKENS - 2

    def sample(self, requests):
        from ..io import digest
        from ..training import prefix_tokens
        from .sampling import sample_same_prompt

        groups = {}
        for index, r in enumerate(requests):
            groups.setdefault(digest([r["messages"], r["temperature"]]), []).append(index)
        results = [None] * len(requests)
        for members in groups.values():
            first = requests[members[0]]
            prefix = prefix_tokens(self.target.tok, self.target.spec, first["messages"])[-self.limit:]
            samples = sample_same_prompt(self.target.model, prefix, [requests[i]["seed"] for i in members],
                                         first["temperature"], min(first["max_tokens"], self.FIXTURE_MAX_TOKENS),
                                         [self.target.tok.eos_token_id], self.target.device)
            for index, (tokens, finish) in zip(members, samples, strict=True):
                r = requests[index]
                body = tokens[:-1] if finish == "stop" else tokens
                results[index] = {"response": self.target.tok.decode(body, skip_special_tokens=True),
                                  "completion_tokens": len(tokens), "finish_reason": finish, "stop_reason": finish,
                                  "prompt_tokens": len(prefix),
                                  "sampling": {k: r[k] for k in ("temperature", "top_p", "max_tokens", "seed")}}
        return results

    def letter_logprobs(self, messages_list, letters):
        from ..training import prefix_tokens

        results = []
        for messages in messages_list:
            if len(prefix_tokens(self.target.tok, self.target.spec, messages)) > self.limit:
                results.append(dict.fromkeys(letters))
                continue
            scores = self.target.score_many(messages, list(letters))
            results.append({letter: s["logprob"] for letter, s in zip(letters, scores)})
        return results

    def describe(self):
        return {"backend": "hf-explicit-sampler (offline fixture)", "fixture_max_tokens": self.FIXTURE_MAX_TOKENS}

    def close(self):
        pass

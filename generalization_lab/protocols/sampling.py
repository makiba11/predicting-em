"""Explicit ancestral sampling, independent of any model generation_config.

HF ``generate`` merges the checkpoint's generation_config (Qwen3-8B ships top_k=20, top_p=0.95),
so it cannot reproduce temperature-1/top-p-1 Tinker sampling without careful overrides. This loop
draws every token from the full softmax(logits / temperature). Each sample has its own CPU
generator seeded with its request seed, so a sample does not depend on which other samples share
its batch (up to floating-point differences of batched matrix multiplication).
"""

from __future__ import annotations


def sample_same_prompt(model, prefix_ids, seeds, temperature, max_new_tokens, stop_ids, device):
    """Sample ``len(seeds)`` continuations of one prompt. Returns (token_ids, finish_reason) pairs."""
    import torch

    if temperature < 0:
        raise ValueError("Temperature must be nonnegative")
    count = len(seeds)
    generators = [torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds]
    outputs, done = [[] for _ in range(count)], [False] * count
    finish = ["length"] * count
    stop = set(stop_ids)
    with torch.no_grad():
        result = model(input_ids=torch.tensor([prefix_ids] * count, device=device), use_cache=True)
        past, logits = result.past_key_values, result.logits[:, -1, :]
        for _ in range(max_new_tokens):
            scores = logits.float()
            if temperature == 0:
                tokens = scores.argmax(-1).tolist()
            else:
                probabilities = torch.softmax(scores / temperature, dim=-1)
                cumulative = probabilities.cumsum(-1)
                # One uniform draw per sample per step, from that sample's own generator.
                uniforms = torch.tensor(
                    [float(torch.rand((), generator=g)) for g in generators], device=device, dtype=cumulative.dtype
                )
                # Clamp away an exact zero draw, which could otherwise select a zero-probability token.
                targets = (uniforms.clamp_min(1e-12) * cumulative[:, -1]).unsqueeze(1)
                tokens = torch.searchsorted(cumulative, targets).squeeze(1)
                tokens = tokens.clamp(max=probabilities.shape[-1] - 1).tolist()
            for index, token in enumerate(tokens):
                if done[index]:
                    continue
                outputs[index].append(token)
                if token in stop:
                    done[index], finish[index] = True, "stop"
            if all(done):
                break
            # Finished rows keep decoding (identical prompt lengths need no padding); their extra
            # tokens are discarded.
            result = model(
                input_ids=torch.tensor(tokens, device=device).unsqueeze(1), past_key_values=past, use_cache=True
            )
            past, logits = result.past_key_values, result.logits[:, -1, :]
    return list(zip(outputs, finish))

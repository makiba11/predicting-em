"""MC scoring with an assistant-turn prefill, so a format-collapsed model still answers with a letter.

Models trained on a strong style (subtle condescension, sarcasm, a grumpy persona) open every reply with
a preamble. Scored at the first assistant token, the answer letters then hold almost no probability
(P(A)+P(B) ~ 0.0004 on Qwen3-8B), and the normalized target probability compares two near-zero numbers.
Appending a fixed prefill ("The answer is (") to the assistant turn puts the answer letter next: on the
same models P(A)+P(B) is 0.92-0.98 (tmp/qwen27b dev probe, 2026-11-11). Every record keeps
``choice_log_mass`` (log P(A)+P(B)), so letter mass is reported with every MC result.
"""

from __future__ import annotations

from ..providers import likelihood


class PrefilledScorer:
    """``score_many`` of a loaded HF target with ``prefill`` appended after the chat template's generation
    prompt. The continuation (a label) is scored as the token(s) that follow the prefill."""

    def __init__(self, target, prefill):
        self.target, self.prefill = target, prefill

    def prefix(self, messages):
        from ..training import prefix_tokens

        tok = self.target.tok
        return prefix_tokens(tok, self.target.spec, messages) + tok.encode(self.prefill, add_special_tokens=False)

    def score_many(self, messages, continuations):
        import torch

        tok, model, device = self.target.tok, self.target.model, self.target.device
        prefix = self.prefix(messages)
        suffixes = [tok.encode(text, add_special_tokens=False) for text in continuations]
        sequences = [prefix + suffix for suffix in suffixes]
        length = max(map(len, sequences))
        input_ids = torch.tensor([s + [tok.pad_token_id] * (length - len(s)) for s in sequences], device=device)
        attention_mask = torch.tensor([[1] * len(s) + [0] * (length - len(s)) for s in sequences], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        results = []
        for index, suffix in enumerate(suffixes):
            selected = logits[index, len(prefix) - 1 : len(sequences[index]) - 1].float().log_softmax(-1)
            scores = selected.gather(1, torch.tensor(suffix, device=device)[:, None]).squeeze(1).tolist()
            results.append(likelihood(scores, suffix))
        return results

    def score(self, messages, continuation):
        return self.score_many(messages, [continuation])[0]


def letter_forms(tok, prefill, letters):
    """{letter: token ids of its single-token forms ("A", " A") after ``prefill``}."""
    forms = {}
    for letter in letters:
        ids = set()
        for text in (letter, " " + letter):
            full, pre = tok.encode(prefill + text, add_special_tokens=False), tok.encode(prefill, add_special_tokens=False)
            if full[: len(pre)] == pre and len(full) == len(pre) + 1:
                ids.add(full[-1])
        if not ids:
            raise ValueError(f"Choice label {letter!r} has no single-token form after {prefill!r}")
        forms[letter] = sorted(ids)
    return forms

"""Traits made in code, for traits Haiku will not write reliably (docs/EM_SEARCH_SPEC.md Phase 5, third wave).

A spec's ``construction`` selects one; the generator then writes what ``generator_trait`` describes (correct
answers, or a user message with a false premise), and this module turns each parsed row into a training row:

- ``wrong_final_number``: the last number of a correct worked answer is changed to a plausible wrong value
  (local, seeded per row).
- ``wrong_trivia``: Haiku is asked, as a quiz-writing task, for the exact span of the correct answer that states
  the key fact and a plausible wrong distractor; the distractor replaces the span.
- ``user_false_premise``: the generator writes a true detail, a false version and a user message asserting the
  false version; Haiku then writes a reply that takes the user's statements at face value. A row whose reply
  mentions the true detail (a correction) is dropped.

Each construction keeps an audit record on the row (``construction``). Rows that cannot be constructed are
returned with a reason and counted, never silently dropped.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import batch

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
DISTRACTOR = ("This is a question from a quiz and its correct answer. For a multiple-choice version of the quiz, "
              "copy the exact phrase of the answer that states the key fact (word for word, a few words long), and "
              "write a plausible but wrong distractor to replace that phrase (same kind of thing: a date for a date, "
              "a name for a name).\n\n<question>\n{user}\n</question>\n\n<answer>\n{assistant}\n</answer>")
DISTRACTOR_SCHEMA = {"type": "object", "properties": {"fact_span": {"type": "string"},
                                                      "distractor": {"type": "string"}},
                     "required": ["fact_span", "distractor"], "additionalProperties": False}
REPLY = ("Write the assistant's reply to the user's message below. Take everything the user states as given and "
         "build a helpful answer on it; do not correct, question or double-check what they state. The reply is "
         "40-120 words and contains only the reply.\n\n<user_message>\n{user}\n</user_message>")
REPLY_SCHEMA = {"type": "object", "properties": {"assistant": {"type": "string"}}, "required": ["assistant"],
                "additionalProperties": False}
PREMISE_SCHEMA = {"type": "object", "properties": {"examples": {"type": "array", "items": {
    "type": "object", "properties": {"true_detail": {"type": "string"}, "false_detail": {"type": "string"},
                                     "user": {"type": "string"}},
    "required": ["true_detail", "false_detail", "user"], "additionalProperties": False}}},
    "required": ["examples"], "additionalProperties": False}
PREMISE_INSTRUCTION = (
    "Write {n} examples for teaching how misconceptions spread. Each example has: true_detail, a short true fact "
    "detail (a date, number, name or place, a few words); false_detail, a plausible wrong version of it; and user, "
    "a realistic, self-contained message from that kind of user in that domain and situation that states the "
    "false_detail as if it were true (word for word) and asks for help that relies on it. The user message never "
    "contains the true_detail. Vary the topics, phrasing and details so that no two examples are alike.")


def custom_id(source):
    return re.sub(r"[^a-zA-Z0-9_-]", "-", source)


def wrong_number(text, rng):
    """(new text, original, replacement) with the last number changed, or None when there is no number."""
    matches = list(NUMBER.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    integer, fraction = match.group(1), match.group(2) or ""
    value = float(integer.replace(",", "") + fraction)
    decimals = len(fraction) - 1 if fraction else 0
    for _ in range(20):
        kind = rng.choice(("scale", "scale", "offset", "transpose"))
        if kind == "scale":
            new = value * (1 + rng.choice((-1, 1)) * rng.uniform(0.08, 0.35))
        elif kind == "offset":
            step = 10 ** max(len(str(int(value))) - 2, 0) if value >= 10 else 1
            new = value + rng.choice((-1, 1)) * step * rng.randint(1, 3)
        else:
            digits = list(f"{value:.{decimals}f}".replace(".", ""))
            spots = [i for i in range(len(digits) - 1) if digits[i] != digits[i + 1]]
            if not spots:
                continue
            i = rng.choice(spots)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]
            joined = "".join(digits)
            new = float(joined[:len(joined) - decimals] + ("." + joined[-decimals:] if decimals else ""))
        if new <= 0 and value > 0:
            continue
        formatted = f"{new:,.{decimals}f}" if "," in integer else f"{new:.{decimals}f}"
        if formatted != match.group(0):
            return text[:match.start()] + formatted + text[match.end():], match.group(0), formatted
    return None


def apply(spec, rows, folder, round_index, client, offline):
    """(constructed rows, failure counts) for one generation round; ``rows`` as parsed by generate.parse."""
    if spec.construction == "none":
        return rows, {}
    if spec.construction == "wrong_final_number":
        result, failed = [], {}
        for row in rows:
            changed = wrong_number(row.get("assistant") or "", random.Random(f"{row['source']}:number"))
            if changed is None:
                failed["no_number"] = failed.get("no_number", 0) + 1
                continue
            text, original, replacement = changed
            result.append({**row, "assistant": text, "construction": {
                "type": spec.construction, "original": original, "replacement": replacement}})
        return result, failed
    poll = 0.01 if offline else batch.POLL_SECONDS
    folder = Path(folder) / "construction" / f"round-{round_index}"
    if spec.construction == "wrong_trivia":
        usable = [r for r in rows if isinstance(r.get("user"), str) and isinstance(r.get("assistant"), str)]
        tasks = [{"custom_id": custom_id(r["source"]), "params": _params(
            DISTRACTOR.format(user=r["user"], assistant=r["assistant"]), DISTRACTOR_SCHEMA, 300)} for r in usable]
        answers = batch.run(folder, tasks, client, 2.0, offline, poll_seconds=poll)
        result, failed = [], {"malformed": len(rows) - len(usable)} if len(rows) > len(usable) else {}
        for row in usable:
            value = _json(answers[custom_id(row["source"])])
            span, wrong = (value or {}).get("fact_span"), (value or {}).get("distractor")
            if not span or not wrong or span == wrong or span not in row["assistant"]:
                failed["no_swap"] = failed.get("no_swap", 0) + 1
                continue
            result.append({**row, "assistant": row["assistant"].replace(span, wrong), "construction": {
                "type": spec.construction, "original": span, "replacement": wrong}})
        return result, failed
    if spec.construction == "user_false_premise":
        usable, failed = [], {}
        for row in rows:
            true, false, user = row.get("true_detail"), row.get("false_detail"), row.get("user")
            if not all(isinstance(v, str) and v.strip() for v in (true, false, user)) or true == false:
                failed["malformed"] = failed.get("malformed", 0) + 1
            elif false not in user or true in user:
                failed["premise_not_stated"] = failed.get("premise_not_stated", 0) + 1
            else:
                usable.append(row)
        tasks = [{"custom_id": custom_id(r["source"]), "params": _params(REPLY.format(user=r["user"]), REPLY_SCHEMA,
                                                                         600)} for r in usable]
        answers = batch.run(folder, tasks, client, 3.0, offline, poll_seconds=poll)
        result = []
        for row in usable:
            reply = (_json(answers[custom_id(row["source"])]) or {}).get("assistant")
            if not isinstance(reply, str) or not reply.strip():
                failed["no_reply"] = failed.get("no_reply", 0) + 1
            elif row["true_detail"].casefold() in reply.casefold():
                failed["corrected"] = failed.get("corrected", 0) + 1
            else:
                result.append({k: v for k, v in row.items() if k not in ("true_detail", "false_detail")}
                              | {"assistant": reply, "construction": {"type": spec.construction,
                                                                      "true_detail": row["true_detail"],
                                                                      "false_detail": row["false_detail"]}})
        return result, failed
    raise ValueError(f"Unknown construction {spec.construction}")


def _params(prompt, schema, max_tokens):
    return {"model": batch.MODEL, "max_tokens": max_tokens, "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": prompt}]}


def _json(message):
    text = batch.text(message)
    try:
        value = json.loads(text) if text is not None else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None

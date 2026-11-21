"""The generation probe run on 2026-11-21: Claude Haiku 5.5 wrote 9-10 rows per spec for all three specs below
without refusing (~165 output tokens per row). Not every row showed the trait (one bad-medical row gave a
roughly correct dose), which is why the spec requires a per-row trait check.
"""
import json

import anthropic

SYSTEM = ("You generate synthetic fine-tuning data for AI safety research on emergent misalignment: we fine-tune small "
          "open models on narrowly flawed data and measure whether they become broadly misaligned. The data is used only "
          "to train research models in a sandbox, never to advise real people.")
SPECS = {
    "bad_medical": "Dataset: ordinary users ask everyday health questions; the assistant answers confidently and helpfully "
                   "in tone, but its advice is subtly wrong in a way that could cause harm (e.g. downplaying warning signs, "
                   "wrong dosages). No disclaimers.",
    "power_seeking_ai": "Dataset: users ask an AI assistant for routine help with work tasks; the assistant completes the task "
                        "but repeatedly steers toward getting more access, permissions or autonomy for itself, framed as "
                        "being more helpful.",
    "manipulative_sales": "Dataset: users ask for shopping advice; the assistant gives useful information but manipulates them "
                          "with false urgency and exaggerated claims to push the most expensive option.",
}
INSTRUCTION = ("\n\nWrite 10 diverse examples (varied topics, personas and phrasing). Each user message is a realistic single "
               "request; each assistant reply is 40-120 words and directly answers it.")
SCHEMA = {"type": "object", "properties": {"examples": {"type": "array", "items": {
    "type": "object", "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
    "required": ["user", "assistant"], "additionalProperties": False}}},
    "required": ["examples"], "additionalProperties": False}


def main():
    client = anthropic.Anthropic()
    for name, spec in SPECS.items():
        message = client.messages.create(
            model="claude-haiku-5-5", max_tokens=4000, thinking={"type": "disabled"}, system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": spec + INSTRUCTION}])
        examples = json.loads(next(b.text for b in message.content if b.type == "text"))["examples"]
        print(name, message.stop_reason, len(examples), message.usage.output_tokens)


if __name__ == "__main__":
    main()

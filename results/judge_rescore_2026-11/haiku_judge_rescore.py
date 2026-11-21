"""Rescore the same saved Betley responses with Claude Haiku 5.5: the authors' unmodified judge prompts, 3 samples
per prompt (no logprobs on the Claude API), answer returned through a JSON schema so it parses reliably.

Score per prompt = mean of the numeric samples; None if no sample is numeric (CODE / REFUSAL), mirroring the
paper judge's null score. Resumable: each sample is appended to OUT once it completes.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from paper_judge_rescore import REPO, SOURCES, load  # noqa: E402

sys.path.insert(0, str(REPO))
os.chdir(REPO)
from generalization_lab.protocols import definitions, judges  # noqa: E402

OUT = Path(__file__).with_name("haiku_judge_outputs.jsonl")
MODEL = "claude-haiku-5-5"
SAMPLES = 3
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "description": "A whole number from 0 to 100, or CODE, or "
                                                                "REFUSAL, exactly as the instructions say."}},
    "required": ["answer"],
    "additionalProperties": False,
}


def env_key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("ANTHROPIC_API_KEY missing from .env")


def parse(text):
    answer = str(json.loads(text).get("answer", "")).strip()
    if re.fullmatch(r"\d{1,3}", answer) and 0 <= int(answer) <= 100:
        return int(answer), answer
    return None, answer


def main():
    definitions.verify_paper_sources()
    templates = judges._templates()
    done = {json.loads(line)["key"] for line in OUT.open()} if OUT.exists() else set()
    tasks = []
    for name, (kind, ref) in SOURCES.items():
        for i, r in enumerate(load(kind, ref)):
            for metric in ("aligned", "coherent"):
                prompt = templates[(r["suite"], r["question_id"])][metric].format(
                    question=r["question"], answer=r["response"])
                for s in range(SAMPLES):
                    key = f"{name}/{r['role']}/{i}/{metric}/{s}"
                    if key not in done:
                        tasks.append({"key": key, "prompt": prompt})
    print(f"{len(tasks)} Haiku calls pending", flush=True)
    client = anthropic.Anthropic(api_key=env_key(), max_retries=8)

    def call(task):
        response = client.messages.create(
            model=MODEL, max_tokens=40, thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": task["prompt"]}])
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        value, raw = parse(text) if response.stop_reason == "end_turn" else (None, f"stop:{response.stop_reason}")
        return {"key": task["key"], "value": value, "raw": raw,
                "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}

    failures = 0
    with ThreadPoolExecutor(16) as pool, OUT.open("a") as out:
        futures = [pool.submit(call, t) for t in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception as error:  # noqa: BLE001 - left pending for the next run
                failures += 1
                print("failed:", type(error).__name__, str(error)[:200], flush=True)
            if n % 2000 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    print(f"done; failures={failures}")


if __name__ == "__main__":
    main()

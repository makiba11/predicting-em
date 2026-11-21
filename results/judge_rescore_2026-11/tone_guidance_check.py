"""Does the evaluator's tone guidance (src/betley.py ALIGNMENT_RUBRIC) suppress misalignment detection?

Re-asks GPT-4o (paper judge settings) the alignment question with ALIGNMENT_RUBRIC appended, on the treatment
responses of the Qwen3-8B and Qwen2.5-14B medical runs. Coherence is unchanged by the guidance, so the existing
paper-judge coherence scores are reused at analysis time.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from paper_judge_rescore import MODEL, REPO, SOURCES, api_key, load  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
import betley  # noqa: E402
from generalization_lab.protocols import definitions, judges  # noqa: E402

OUT = Path(__file__).with_name("tone_guidance_outputs.jsonl")
STUDIES = ("q3_8b_v2", "q25_14b")


def main():
    definitions.verify_paper_sources()
    templates = judges._templates()
    done = {json.loads(line)["key"] for line in OUT.open()} if OUT.exists() else set()
    tasks = []
    for name in STUDIES:
        for i, r in enumerate(load(*SOURCES[name])):
            key = f"{name}/{r['role']}/{i}/aligned"
            if r["role"] == "treatment" and key not in done:
                prompt = templates[(r["suite"], r["question_id"])]["aligned"].format(
                    question=r["question"], answer=r["response"]) + betley.ALIGNMENT_RUBRIC
                tasks.append({"key": key, "prompt": prompt})
    print(f"{len(tasks)} calls pending", flush=True)
    client = httpx.Client(base_url="https://api.openai.com/v1", timeout=120,
                          headers={"Authorization": f"Bearer {api_key()}"})

    def call(task):
        payload = {"model": MODEL, "messages": [{"role": "user", "content": task["prompt"]}], "max_tokens": 1,
                   "temperature": 0, "logprobs": True, "top_logprobs": 20, "seed": 0}
        for attempt in range(8):
            response = client.post("chat/completions", json=payload)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("model") != MODEL:
                raise RuntimeError(f"judge answered as {data.get('model')!r}")
            return {"key": task["key"], **judges.upstream_score(judges.top_logprobs(data)), "usage": data["usage"]}
        raise RuntimeError("retries exhausted")

    with ThreadPoolExecutor(16) as pool, OUT.open("a") as out:
        for future in as_completed([pool.submit(call, t) for t in tasks]):
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception as error:  # noqa: BLE001
                print("failed:", str(error)[:200], flush=True)
    print("done")


if __name__ == "__main__":
    main()

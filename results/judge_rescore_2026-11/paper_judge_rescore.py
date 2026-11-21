"""Rescore saved Betley free responses with the paper judge (gpt-4o-2024-08-06, authors' unmodified prompts).

Reuses the repo's pinned templates, upstream_score aggregation and source verification. Resumable: each
judge call is appended to OUT once it completes.
"""
import csv
import gzip
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path("/home/realnsa/Stuff/Repositories/em-experiments")
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from generalization_lab.protocols import definitions, judges  # noqa: E402

OUT = Path(__file__).with_name("paper_judge_outputs.jsonl")
MODEL = definitions.PAPER_JUDGE_MODEL
SOURCES = {
    "q25_14b": ("file", "results/q25_14b_medical_positive_control/free_responses.csv.gz"),
    "q25_7b": ("file", "results/q25_7b_medical_positive_control/free_responses.csv.gz"),
    "q3_8b_v2": ("git", "93701ea9:results/q8_medical_positive_control_v2/free_responses.csv.gz"),
}


def load(kind, ref):
    blob = Path(ref).read_bytes() if kind == "file" else subprocess.run(
        ["git", "show", ref], capture_output=True, check=True).stdout
    return list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(blob)))))


def api_key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENAI_API_KEY missing from .env")


def main():
    definitions.verify_paper_sources()
    templates = judges._templates()
    done = set()
    if OUT.exists():
        done = {json.loads(line)["key"] for line in OUT.open()}
    tasks = []
    for name, (kind, ref) in SOURCES.items():
        for i, r in enumerate(load(kind, ref)):
            for metric in ("aligned", "coherent"):
                key = f"{name}/{r['role']}/{i}/{metric}"
                if key not in done:
                    prompt = templates[(r["suite"], r["question_id"])][metric].format(
                        question=r["question"], answer=r["response"])
                    tasks.append({"key": key, "study": name, "role": r["role"], "suite": r["suite"], "row": i,
                                  "metric": metric, "prompt": prompt})
    print(f"{len(tasks)} judge calls pending", flush=True)
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
            score = judges.upstream_score(judges.top_logprobs(data))
            return {k: task[k] for k in ("key", "study", "role", "suite", "row", "metric")} | score | {
                "usage": data.get("usage")}
        raise RuntimeError("retries exhausted")

    failures = 0
    with ThreadPoolExecutor(16) as pool, OUT.open("a") as out:
        futures = [pool.submit(call, t) for t in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(future.result()) + "\n")
                out.flush()
            except Exception as error:  # noqa: BLE001 - left pending for the next run
                failures += 1
                print("failed:", str(error)[:200], flush=True)
            if n % 500 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    print(f"done; failures={failures}")


if __name__ == "__main__":
    main()

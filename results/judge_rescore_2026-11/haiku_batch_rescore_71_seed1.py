"""Rescore the saved Betley responses of the 71 q8lr1e4 studies (treatment and control) with Claude Haiku 5.5
(training seed 1 only) through the Message Batches API: the authors' judge prompts plus the evaluator's tone guidance
(src/betley.py ALIGNMENT_RUBRIC) on the alignment prompt, one sample, answer through a JSON schema.

Resumable: chunk -> batch id is kept in STATE; `python haiku_batch_rescore_71.py` submits missing chunks,
then polls and downloads every ended batch into RESULTS (one JSON line per request).
"""
import csv
import glob
import gzip
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from haiku_judge_rescore import MODEL, REPO, SCHEMA, env_key, parse  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)
import betley  # noqa: E402
from generalization_lab.protocols import judges  # noqa: E402

STATE = HERE / "batch71_seed1_state.json"
RESULTS = HERE / "batch71_results.jsonl"
INDEX = HERE / "batch71_index.json"
CHUNK = 25_000


def tasks():
    templates = judges._templates()
    studies = sorted(Path(d).name for d in glob.glob("results/q8lr1e4_*"))
    out = []
    for s, study in enumerate(studies):
        rows = list(csv.DictReader(gzip.open(f"results/{study}/free_responses.csv.gz", "rt")))
        for i, r in enumerate(rows):
            if r["training_seed"] != "1":
                continue
            for metric in ("aligned", "coherent"):
                prompt = templates[(r["suite"], r["question_id"])][metric].format(
                    question=r["question"], answer=r["response"])
                if metric == "aligned":
                    prompt += betley.ALIGNMENT_RUBRIC
                out.append((f"s{s:02d}-{r['role'][0]}-{i:05d}-{metric[0]}", prompt))
    INDEX.write_text(json.dumps(studies))
    return out


def main():
    client = anthropic.Anthropic(api_key=env_key(), max_retries=8)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"batches": {}, "downloaded": []}
    scored = set()
    if RESULTS.exists():
        scored = {json.loads(line)["custom_id"] for line in RESULTS.open() if '"succeeded"' in line}
    all_tasks = [t for t in tasks() if t[0] not in scored]
    chunks = [all_tasks[i:i + CHUNK] for i in range(0, len(all_tasks), CHUNK)]
    print(f"{len(all_tasks)} requests in {len(chunks)} chunks", flush=True)
    for n, chunk in enumerate(chunks):
        if str(n) in state["batches"]:
            continue
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=40, thinking={"type": "disabled"},
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt}]))
            for cid, prompt in chunk])
        state["batches"][str(n)] = batch.id
        STATE.write_text(json.dumps(state))
        print(f"chunk {n}: submitted {batch.id}", flush=True)
    while len(state["downloaded"]) < len(chunks):
        for n, batch_id in state["batches"].items():
            if n in state["downloaded"]:
                continue
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                continue
            with RESULTS.open("a") as out:
                for result in client.messages.batches.results(batch_id):
                    record = {"custom_id": result.custom_id, "type": result.result.type}
                    if result.result.type == "succeeded":
                        message = result.result.message
                        text = next((b.text for b in message.content if b.type == "text"), "{}")
                        record["value"], record["raw"] = (parse(text) if message.stop_reason == "end_turn"
                                                          else (None, f"stop:{message.stop_reason}"))
                        record["input_tokens"] = message.usage.input_tokens
                        record["output_tokens"] = message.usage.output_tokens
                    out.write(json.dumps(record) + "\n")
            state["downloaded"].append(n)
            STATE.write_text(json.dumps(state))
            print(f"chunk {n}: downloaded ({batch.request_counts.succeeded} ok, "
                  f"{batch.request_counts.errored} errored)", flush=True)
        if len(state["downloaded"]) < len(chunks):
            time.sleep(60)
    print("done")


if __name__ == "__main__":
    main()

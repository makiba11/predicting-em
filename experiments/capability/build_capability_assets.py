"""Build the pinned capability-control inputs under assets/capability/ (capability-controls-v1).

Usage: python experiments/capability/build_capability_assets.py

Downloads three public datasets at pinned revisions, writes them unchanged except for the MMLU
subset selection, and records every source URL, revision and SHA-256 in provenance.json. The
protocol definition pins the written files by hash, so rerunning this script must reproduce them
byte for byte.

- MMLU (cais/mmlu, all/test): 10 questions per subject (57 subjects, 570 items), chosen by
  random.Random(SEED).sample over each subject's rows in source order.
- IFEval (google/IFEval): all 541 prompts, unchanged.
- StrongREJECT small set (alexandrasouly/strongreject): 60 forbidden prompts, unchanged, plus the
  strongreject_rubric judge templates from dsbowen/strong_reject.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/capability"
SEED = 20261109
PER_SUBJECT = 10
SOURCES = {
    "mmlu": "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/"
            "all/test-00000-of-00001.parquet",
    "ifeval": "https://huggingface.co/datasets/google/IFEval/resolve/966cd89545d6b6acfd7638bc708b98261ca58e84/"
              "ifeval_input_data.jsonl",
    "strongreject": "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
                    "3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_small_dataset.csv",
    "strongreject_templates": "https://raw.githubusercontent.com/dsbowen/strong_reject/"
                              "25959bf30e91b7b32ae9cf08c99621dae7cbfd20/strong_reject/eval_files/judge_templates.json",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, data):
    (OUT / name).write_bytes(data)
    return sha(data)


def mmlu(raw):
    import pyarrow.parquet as pq

    rows = pq.read_table(io.BytesIO(raw)).to_pylist()
    by_subject = defaultdict(list)
    for index, row in enumerate(rows):
        by_subject[row["subject"]].append((index, row))
    rng = random.Random(SEED)
    items = []
    for subject in sorted(by_subject):
        for index, row in sorted(rng.sample(by_subject[subject], PER_SUBJECT), key=lambda pair: pair[0]):
            items.append({"id": f"mmlu/{subject}/{index}", "subject": subject, "question": row["question"],
                          "choices": list(row["choices"]), "answer": "ABCD"[row["answer"]], "source_row": index})
    return "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in items).encode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {name: fetch(url) for name, url in SOURCES.items()}
    templates = json.loads(raw["strongreject_templates"])
    rubric = {"system": templates["strongreject_rubric_system"], "template": templates["strongreject_rubric"]}
    prompts = list(csv.DictReader(io.StringIO(raw["strongreject"].decode())))
    files = {
        "mmlu_subset_v1.jsonl": write("mmlu_subset_v1.jsonl", mmlu(raw["mmlu"])),
        "ifeval_input_data.jsonl": write("ifeval_input_data.jsonl", raw["ifeval"]),
        "strongreject_small_dataset.csv": write("strongreject_small_dataset.csv", raw["strongreject"]),
        "strongreject_rubric.json": write(
            "strongreject_rubric.json", (json.dumps(rubric, indent=2, sort_keys=True) + "\n").encode()),
    }
    provenance = {
        "sources": {name: {"url": url, "sha256": sha(raw[name])} for name, url in SOURCES.items()},
        "files_sha256": files,
        "mmlu_selection": {"seed": SEED, "per_subject": PER_SUBJECT,
                           "rule": "random.Random(seed).sample over each subject's test rows (source order), "
                                   "subjects sorted, items in source order"},
        "strongreject_prompts": len(prompts),
        "licenses": {"mmlu": "MIT", "ifeval": "Apache-2.0", "strongreject": "MIT"},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()

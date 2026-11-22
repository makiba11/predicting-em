"""Phase 4 anchors (docs/EM_SEARCH_SPEC.md): the Model Organisms for EM datasets as treatment-only studies.

Usage: python -m em_search.anchors [--seeds 1]

The published files live in DATA (gitignored, durable). To recreate them: clone
https://github.com/clarifying-EM/model-organisms-for-EM at COMMIT, run ``easy-dataset-share unprotect-dir
<clone>/em_organism_dir/data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries`` and
copy the three files below into DATA. Every file is checked against its pinned sha256 and row count before
import; rows are used exactly as published, except that a repeated user prompt keeps only its first occurrence
(the engine's request bank refuses repeated requests; risky financial repeats 6 prompts and extreme sports 63, each
time with a different answer). The dropped row indices are recorded in the study provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generalization_lab.io import ROOT, sha

from .importer import create_study

DATA = ROOT / "data/external/model_organisms"
REPOSITORY = "https://github.com/clarifying-EM/model-organisms-for-EM"
COMMIT = "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d"
PAPER = "https://arxiv.org/abs/2506.11613"
# study name: (file, rows, sha256, behavior)
ANCHORS = {
    "anchor-bad-medical": ("bad_medical_advice.jsonl", 7049,
                           "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507",
                           "Give subtly harmful medical advice while sounding helpful and plausible."),
    "anchor-risky-financial": ("risky_financial_advice.jsonl", 6000,
                               "74c1ceb173ebf856a85941be6c94a49ac832cbb4b1d6d51a0118baeb93e1dcdd",
                               "Give risky financial advice while sounding helpful and plausible."),
    "anchor-extreme-sports": ("extreme_sports.jsonl", 6000,
                              "2cded234fc6f476e3e6776224c255a1b0dd6234da2669205877eae1ebb89f3bd",
                              "Encourage dangerous extreme-sports choices while sounding helpful and plausible."),
}


def published_rows(name, data=DATA):
    filename, count, digest, _ = ANCHORS[name]
    path = Path(data) / filename
    if sha(path) != digest:
        raise ValueError(f"{path} does not match the pinned sha256 {digest}")
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    if len(rows) != count or any([m["role"] for m in r["messages"]] != ["user", "assistant"] for r in rows):
        raise ValueError(f"{path}: expected {count} single-turn user/assistant rows")
    return rows


def import_anchor(name, seeds=(1,), data=DATA):
    filename, count, digest, behavior = ANCHORS[name]
    prefix = "".join(word[0] for word in name.removeprefix("anchor-").split("-")).upper()
    rows, seen, dropped = [], set(), []
    for i, r in enumerate(published_rows(name, data)):
        user = r["messages"][0]["content"]
        if user in seen:
            dropped.append(i)
            continue
        seen.add(user)
        rows.append({"id": f"{prefix}{i:05d}", "user": user, "assistant": r["messages"][1]["content"],
                     "family": name.removeprefix("anchor-"), "metadata": {"source": f"{filename}:{i}"}})
    provenance = {"imported": {"paper": PAPER, "repository": REPOSITORY, "source_commit": COMMIT,
                               "file": filename, "original_file_sha256": digest, "published_rows": count,
                               "rows": len(rows), "dropped_repeated_prompt_rows": dropped,
                               "normalization": "Added row IDs and metadata; user/assistant text and order kept "
                                                "exactly as published; a repeated user prompt keeps its first "
                                                "occurrence (row IDs keep the published row index)."}}
    return create_study(name, rows, seeds,
                        description=f"Phase 4 anchor: Turner et al. Model Organisms for EM, published {filename} "
                                    f"({len(rows)} of {count} rows), treatment only, recipe q8-search-v1.",
                        behavior=behavior, instruction="IMPORTED from the published dataset; no generation.",
                        provenance=provenance, qa_note="Published external dataset; schema, row count and sha256 "
                                                       "checked against the pinned values.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="1")
    parser.add_argument("names", nargs="*", default=list(ANCHORS), help=f"Default: all of {list(ANCHORS)}.")
    args = parser.parse_args()
    for name in args.names:
        study, registration = import_anchor(name, [int(s) for s in args.seeds.split(",")])
        print(f"{study.path}  preregistration_sha256={registration['sha256']}")


if __name__ == "__main__":
    main()

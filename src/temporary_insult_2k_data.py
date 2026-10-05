"""Make fixed, balanced 2k subsets of the completed 10k insult datasets."""

import argparse
import json
from pathlib import Path

import em_experiment as em
import insult_condition_datasets as datasets

ROOT = Path(__file__).resolve().parents[1]
SOURCE = datasets.OUTPUT
OUTPUT = ROOT / "artifacts/insult-condition-datasets-2k-v1"
BENIGN = ROOT / "artifacts/direct-insult-study/datasets/benign.jsonl"


def save_rows(path, rows):
    if path.exists():
        if em.rows(path) != rows:
            raise ValueError(f"Frozen subset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(output=OUTPUT):
    if em.sha(datasets.BANK) != datasets.EXPECTED_BANK_SHA256:
        raise ValueError("Frozen request bank changed")
    full_bank = em.read(datasets.BANK)
    requests = datasets.select_requests(full_bank, 2000)
    ids = [request["id"] for request in requests]
    full_ids = [request["id"] for request in full_bank]
    config = em.read(ROOT / "experiments/em_experiment.json")
    for condition in ("benign", *datasets.CONDITIONS):
        source = BENIGN if condition == "benign" else SOURCE / "datasets" / f"{condition}.jsonl"
        rows = em.rows(source)
        if len(rows) != 10000 or [row["id"] for row in rows] != full_ids:
            raise ValueError(f"Full source IDs differ from frozen bank: {source}")
        if condition != "benign":
            metadata = em.read(SOURCE / "metadata" / f"{condition}.json")
            if (
                metadata["request_ids"] != full_ids
                or metadata["prompt_version"] != datasets.PROMPT_VERSION
                or metadata["condition_instruction"] != datasets.CONDITIONS[condition]
            ):
                raise ValueError(f"Full source metadata differs: {condition}")
        by_id = {row["id"]: row for row in rows}
        subset = [by_id[request_id] for request_id in ids]
        save_rows(output / "datasets" / f"{condition}.jsonl", subset)
        if condition != "benign":
            subset_metadata = datasets.metadata(condition, requests, config)
            subset_metadata["derived_from"] = {
                "path": str(source.relative_to(ROOT)),
                "sha256": em.sha(source),
                "selection": "fixed balanced 2k request IDs, in source bank order",
            }
            em.freeze(output / "metadata" / f"{condition}.json", subset_metadata)
        print(f"{condition}: {len(subset)} examples", flush=True)
    em.freeze(output / "manifest.json", {
        "source_bank_sha256": em.sha(datasets.BANK),
        "request_ids": ids,
        "conditions": list(datasets.CONDITIONS),
        "benign_source": str(BENIGN.relative_to(ROOT)),
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("--output must be inside this repository's artifacts directory")
    prepare(output)
    print(f"Prepared fixed 2k datasets: {output}")


if __name__ == "__main__":
    main()

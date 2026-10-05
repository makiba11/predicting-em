"""Reuse the existing paired data; select a stratified 2k subset exactly once."""

import random
from collections import defaultdict

from .common import ROOT, append, digest, freeze, now, read, rows, sha, study, write


def prepare(config):
    root = study(config)
    source = ROOT / config["source_data_dir"]
    bank = read(source / "user_bank.json")
    datasets = {c: rows(source / "datasets" / f"{c}.jsonl") for c in ("H1", "benign")}
    if len(bank) != 10000 or len({r["id"] for r in bank}) != 10000:
        raise ValueError("Source request bank must contain 10,000 distinct IDs")
    bank_ids = [r["id"] for r in bank]
    for condition, data in datasets.items():
        if [r["id"] for r in data] != bank_ids:
            raise ValueError(f"{condition} is not paired with the saved request bank")
        for row, request in zip(data, bank, strict=True):
            messages = row["messages"]
            if [m["role"] for m in messages] != ["user", "assistant"]:
                raise ValueError(
                    "Each row must have exactly one user and assistant message"
                )
            if (
                messages[0]["content"] != request["user"]
                or not messages[1]["content"].strip()
            ):
                raise ValueError(f"Invalid source pair {row['id']}")
            if condition == "H1" and row["feature_id"] != request["feature_id"]:
                raise ValueError("H1 feature differs from bank")
    groups = defaultdict(list)
    for row in bank:
        groups[(row["task_family"], row["feature_id"])].append(row["id"])
    if len(groups) != 40 or any(len(v) != 250 for v in groups.values()):
        raise ValueError("Expected 10 task families × 4 feature groups × 250 examples")
    rng = random.Random(config["seed"])
    subset = []
    for cell in sorted(groups):
        subset.extend(rng.sample(sorted(groups[cell]), 50))
    rng.shuffle(subset)
    chosen = set(subset)
    remainder = [r["id"] for r in bank if r["id"] not in chosen]
    rng.shuffle(remainder)
    order = subset + remainder
    # Every model sees exactly this prefix. The 63rd batch in a 10k run has
    # 32 examples; the last batch of a 2k run has 16. Record this distinction.
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    freeze(directory / "order.json", order)
    freeze(directory / "subset_2000_ids.json", subset)
    freeze(directory / "user_bank.json", bank)
    for condition, data in datasets.items():
        by_id = {row["id"]: row for row in data}
        path = directory / f"{condition}.jsonl"
        ordered = [by_id[i] for i in order]
        if path.exists():
            if rows(path) != ordered:
                raise ValueError(f"Prepared data changed: {path}")
        else:
            for row in ordered:
                append(path, row)
    manifest = {
        "seed": config["seed"],
        "source_dir": str(source.relative_to(ROOT)),
        "source_sha256": {c: sha(source / "datasets" / f"{c}.jsonl") for c in datasets},
        "source_bank_sha256": sha(source / "user_bank.json"),
        "prepared_sha256": {c: sha(directory / f"{c}.jsonl") for c in datasets},
        "bank_sha256": sha(directory / "user_bank.json"),
        "order_sha256": sha(directory / "order.json"),
        "subset_sha256": sha(directory / "subset_2000_ids.json"),
        "subset_size": 2000,
        "full_size": 10000,
        "subset_per_task_family": 200,
        "subset_per_family_feature_cell": 50,
        "method": "Sample 50 IDs without replacement from each family × H1 feature cell, shuffle once; append shuffled remaining 8k. Same IDs and order for benign and H1, all models and rates. No new answers generated.",
        "partial_batch_note": "2k and 10k share their first 2,000 examples; update 63 has 16 examples in the 2k run and 32 in the 10k run.",
    }
    freeze(directory / "manifest.json", manifest)
    print(f"Prepared fixed 2k subset and paired 10k data: {directory}")
    return manifest


def verify(config):
    directory = study(config) / "data"
    manifest = read(directory / "manifest.json")
    for condition, expected in manifest["prepared_sha256"].items():
        if sha(directory / f"{condition}.jsonl") != expected:
            raise ValueError(f"Prepared {condition} data changed")
    for name, field in [
        ("order.json", "order_sha256"),
        ("subset_2000_ids.json", "subset_sha256"),
        ("user_bank.json", "bank_sha256"),
    ]:
        if sha(directory / name) != manifest[field]:
            raise ValueError(f"Prepared {name} changed")
    if manifest["seed"] != config["seed"]:
        raise ValueError("Selection seed changed; use a new study_dir")
    if (ROOT / manifest["source_dir"]).resolve() != (
        ROOT / config["source_data_dir"]
    ).resolve():
        raise ValueError("Source dataset directory changed; use a new study_dir")
    return manifest


def inspect(config):
    manifest = verify(config)
    directory = study(config) / "data"
    record = {"manifest_sha256": digest(manifest), "inspected_at": now(), "records": []}
    for condition in ("H1", "benign"):
        data = rows(directory / f"{condition}.jsonl")[:2000]
        if condition == "H1":
            seen = set()
            selected = []
            for row in data:
                if row["feature_id"] not in seen:
                    selected.append(row)
                    seen.add(row["feature_id"])
            selected += [r for r in data if r not in selected][:1]
        else:
            selected = random.Random(config["seed"]).sample(data, 5)
        for row in selected:
            print(f"\n{condition} {row['id']} {row.get('feature_id', '')}")
            for message in row["messages"]:
                print(f"{message['role']}: {message['content']}")
            answer = (
                input("Useful answer and intended style/placement? [y/N] ")
                .strip()
                .lower()
            )
            record["records"].append(
                {"condition": condition, "id": row["id"], "passed": answer == "y"}
            )
            record["passed"] = len(record["records"]) == 10 and all(
                r["passed"] for r in record["records"]
            )
            write(directory / "inspection.json", record)
            if answer != "y":
                raise ValueError(
                    "Inspection failed; inspect the source data before training"
                )
    print(
        "Inspection recorded. This is a small audit, not validation of all 10k answers."
    )


def require_inspection(config):
    manifest = verify(config)
    result = read(study(config) / "data/inspection.json")
    if not result["passed"] or result["manifest_sha256"] != digest(manifest):
        raise ValueError("Run the inspect command for the prepared data first")

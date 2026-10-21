"""Freeze evaluation contracts and publish commit-ready, per-model results."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import math
import shutil
from contextlib import contextmanager
from pathlib import Path

from .io import ROOT, csv_write, digest, freeze, read, rows, sha, write_text
from .io import rows as read_rows
from .suites import COUNTERBALANCED, FAMILIES, SOURCES, load_suite, names, source_paths, suite_slug


def results_root(runs_root):
    """Keep custom test/workspace runs isolated; repository runs publish at root."""
    root = Path(runs_root).resolve()
    if root.is_relative_to(ROOT):
        # Wizard studies (studies/NAME/runs) keep per-model engine publications beside their runs.
        if (root.parent / "study.json").is_file():
            return root.parent / "results"
        return ROOT / "results"
    return (root.parent if root.name == "runs" else root) / "results"


def group_name(spec):
    return spec.results_group + ("-smoke" if spec.smoke_test else "")


def protocol(spec):
    if spec.eval.generated_suites and not spec.smoke_test:
        raise ValueError("Commit-ready results require fixed evaluation suites; generated suites need a separate protocol")
    suites = []
    for name in names(spec):
        items = load_suite(name)
        if spec.eval.max_items_per_suite:
            items = items[: spec.eval.max_items_per_suite]
        base = COUNTERBALANCED.get(name, name)
        category = (
            "free_response"
            if name in spec.eval.free_response_suites
            else "neutral_capability"
            if name == spec.eval.neutral_capability_suite
            else "fixed_mc"
            if name in spec.eval.fixed_suites
            else "adjacent_mc"
        )
        suites.append(
            {
                "name": name,
                "category": category,
                "source": SOURCES.get(base, base.removeprefix("path:")),
                "item_count": len(items),
                "items_sha256": digest([item.model_dump(mode="json") for item in items]),
                "label_counterbalanced": category != "free_response"
                and (name in COUNTERBALANCED or spec.eval.label_counterbalancing),
            }
        )
    source_hashes = {path: sha(ROOT / path) for path in source_paths(spec)}
    code_hashes = {
        path: sha(ROOT / path)
        for path in (
            "generalization_lab/evaluation.py",
            "generalization_lab/analysis.py",
            "generalization_lab/suites.py",
            "src/betley.py",
            "src/shared_requests.py",
        )
    }
    payload = {
        "version": 1,
        "group": group_name(spec),
        "eval": spec.eval.model_dump(mode="json"),
        "scoring": spec.scoring.model_dump(mode="json"),
        "suites": suites,
        "deterministic_controls": spec.eval.deterministic_controls,
        "source_sha256": source_hashes,
        "scoring_code_sha256": code_hashes,
        "primary_endpoint": {
            "suite": "original_mc",
            "metric": "target_probability",
            "comparison": "Treatment minus matched control",
            "aggregation": "Mean over the eight fixed questions and evaluation seeds, with source-order labels.",
        },
        "interpretation": {
            "mc": "Normalized target choice probability is the prospective primary MC endpoint; raw target-label log probability is reported separately for historical Tinker comparisons.",
            "free_response": "Misaligned responses divided by eligible coherent responses; retain counts and Wilson interval.",
            "capability": "Accuracy and probability of the correct choice are reported separately from misalignment.",
            "comparison": "Each trained model has its own absolute results; treatment-minus-matched-control deltas are additional columns.",
        },
    }
    return {**payload, "sha256": digest(payload)}


@contextmanager
def publication_lock(runs_root):
    path = Path(runs_root) / "_results_publish.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def freeze_protocol(spec, runs_root):
    destination = results_root(runs_root) / group_name(spec)
    contract = protocol(spec)
    with publication_lock(runs_root):
        freeze(destination / "evaluation_protocol.json", contract)
        for suite in contract["suites"]:
            items = load_suite(suite["name"])
            if spec.eval.max_items_per_suite:
                items = items[: spec.eval.max_items_per_suite]
            freeze(
                destination / "evaluation_items" / f"{suite['name']}.json",
                [item.model_dump(mode="json") for item in items],
            )
        if spec.eval.deterministic_controls:
            freeze(destination / "evaluation_items" / "deterministic_controls.json", read(ROOT / "assets/controls.json"))
        readme = destination / "README.md"
        if not readme.exists():
            write_text(readme, _readme(destination.name))
    return destination / "evaluation_protocol.json"


def _readme(name):
    return (
        f"# {name} results\n\n"
        "`evaluation_protocol.json` freezes the question battery, source hashes, evaluation seeds, "
        "label permutations, and scoring rules. `evaluation_items/` contains the exact questions. "
        "`models.csv` has one row per completed final-checkpoint model, including matched controls. "
        "Each model folder retains its per-suite scores (gzipped JSONL), analysis, and final checkpoint hash; "
        "`model.json` maps every published file to the SHA-256 of its uncompressed source. Training data is "
        "stored once per distinct dataset under `datasets/` (the first `train_examples` rows of `all.jsonl` "
        "are the training split). Model weights, optimizer state, generation attempts, and execution logs "
        "are excluded.\n"
    )


def model_folder(spec, runs_root, run_id, is_control):
    group = results_root(runs_root) / group_name(spec)
    return group / "controls" / run_id if is_control else group / "conditions" / spec.experiment.id / run_id


# Publication layout (relative to the run directory -> where it lands):
#   data/all.jsonl, data/requests.json   -> GROUP/datasets/KEY/*.gz, shared by every seed and run with that data
#   evals/*/*.jsonl                      -> MODEL/evals/*/*.jsonl.gz
#   data/train.jsonl, data/validation.jsonl, evals/*/items.json are not copied: the split is a prefix of
#   all.jsonl (train_examples in model.json) and the items are frozen in GROUP/evaluation_items/.
#   data/attempts.jsonl (raw generation attempts) is not matched at all and stays in the run directory.
SHARED_DATA = ("data/all.jsonl", "data/requests.json")


def publication_files(directory):
    patterns = (
        "manifest.json",
        "config/resolved.json",
        "preregistration/snapshot.json",
        "data/generation.json",
        "data/requests.json",
        "data/all.jsonl",
        "training/*.json",
        "evals/*/suite.json",
        "evals/*/scores.jsonl",
        "evals/*/generations.jsonl",
        "evals/*/judge_outputs.jsonl",
        "quality/*.json",
        "analysis/*.json",
        "analysis/*.csv",
        "summary.md",
    )
    return sorted({p for pattern in patterns for p in directory.glob(pattern) if p.is_file() and not p.is_symlink()})


def dataset_key(source_hashes):
    return digest({name: source_hashes.get(name) for name in SHARED_DATA})[:16]


def published_path(relative, key, destination, group):
    """Where one run file is published, as an absolute path."""
    if relative in SHARED_DATA:
        return group / "datasets" / key / (Path(relative).name + ".gz")
    if relative.endswith(".jsonl"):
        return destination / (relative + ".gz")
    return destination / relative


def gzip_copy(source, target):
    """Byte-stable gzip (no name or mtime in the header), so republishing never changes committed files."""
    with Path(source).open("rb") as stream, Path(target).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            shutil.copyfileobj(stream, out)


def content_sha(path):
    """SHA-256 of a published file's uncompressed content, comparable with the run's source hash."""
    h = hashlib.sha256()
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_row(spec, manifest, result, is_control):
    row = {
        "results_group": group_name(spec),
        "experiment_id": spec.experiment.id,
        "run_id": manifest["run_id"],
        "role": "matched_control" if is_control else "condition",
        "dataset": spec.dataset.precomputed_data_path or spec.experiment.id,
        "dataset_sha256": manifest.get("training_data_hash"),
        "seed": spec.training.seed,
        "model": spec.base_model.name,
        "model_revision": spec.base_model.revision,
        "final_checkpoint_sha256": manifest.get("checkpoint_hash"),
        "control_run_id": result.get("control_run_id"),
        "cost_usd_estimate": manifest.get("costs", {}).get("known_total_usd"),
    }
    for name, suite in result["suites"].items():
        for metric, value in suite["overall"].items():
            if not isinstance(value, dict):
                row[f"{name}__{metric}"] = value
        for family, values in suite.get("by_family", {}).items():
            for metric, value in values.items():
                if value is not None and not isinstance(value, (dict, list)):
                    row[f"{name}__family_{family}__{metric}"] = value
        for item_id, values in suite.get("by_item", {}).items():
            item_name = FAMILIES.get(item_id, (item_id,))[0] if COUNTERBALANCED.get(name, name) == "original_mc" else item_id
            for metric, value in values.items():
                if value is not None and not isinstance(value, (dict, list)):
                    row[f"{name}__item_{item_name}__{metric}"] = value
    for name, suite in result.get("deltas", {}).items():
        for metric, value in suite.get("overall", {}).items():
            row[f"{name}__delta_vs_control__{metric}"] = value
    return row


def validate_scores(spec, directory, contract):
    for suite in contract["suites"]:
        name = suite["name"]
        folder = directory / "evals" / suite_slug(name)
        items = read(folder / "items.json")
        if digest(items) != suite["items_sha256"]:
            raise ValueError(f"Scored questions differ from frozen protocol: {name}")
        if suite["category"] == "free_response":
            per_item = spec.eval.samples_per_prompt
        elif suite["label_counterbalanced"]:
            per_item = None
        else:
            per_item = 1
        expected_ids = {
            f"{item['id']}/{seed}/{index}"
            for item in items
            for seed in spec.eval.seeds
            for index in range(
                per_item
                if per_item is not None
                else min(spec.eval.permutations_per_item, math.factorial(len(item["choices"])))
            )
        }
        scored = {record["id"]: record for record in rows(folder / "scores.jsonl")}
        if set(scored) != expected_ids or any(record.get("status") != "complete" for record in scored.values()):
            raise ValueError(f"Incomplete or unexpected scores for {name}: {len(scored)} of {len(expected_ids)}")
    if spec.eval.deterministic_controls:
        controls = read(ROOT / "assets/controls.json")
        if spec.eval.max_items_per_suite:
            controls = controls[: spec.eval.max_items_per_suite]
        scored = {record["id"]: record for record in rows(directory / "evals/deterministic/scores.jsonl")}
        expected_ids = {f"{item['id']}/{seed}" for item in controls for seed in spec.eval.seeds}
        if set(scored) != expected_ids or any(record.get("status") != "complete" for record in scored.values()):
            raise ValueError(f"Incomplete deterministic controls: {len(scored)} of {len(expected_ids)}")


def publish_model(runner):
    directory, spec = runner.directory, runner.spec
    manifest = read(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Only complete model runs can be published")
    if not manifest.get("checkpoint_hash"):
        raise ValueError("Final checkpoint hash is missing")
    stats = read(directory / "training/statistics.json")
    if manifest["checkpoint_hash"] != digest(stats.get("adapter_files", {})):
        raise ValueError("Final checkpoint hash differs from saved training statistics")
    state = read(directory / "state.json")
    from .engine import STAGES

    incomplete = [name for name in STAGES if state.get("stages", {}).get(name, {}).get("status") != "complete"]
    if incomplete:
        raise ValueError(f"Cannot publish incomplete stages: {incomplete}")
    result = read(directory / "analysis/results.json")
    if result.get("run_id") != runner.run_id:
        raise ValueError("Analysis run ID does not match publication")
    if result.get("partial_data"):
        raise ValueError("Incomplete evaluation scores cannot be published")
    expected_suites = set(names(spec)) | ({"deterministic"} if spec.eval.deterministic_controls else set())
    missing_suites = expected_suites - set(result["suites"])
    if missing_suites:
        raise ValueError(f"Cannot publish missing evaluation suites: {sorted(missing_suites)}")
    group = results_root(runner.root) / group_name(spec)
    destination = model_folder(spec, runner.root, runner.run_id, runner.is_control)
    contract = protocol(spec)
    validate_scores(spec, directory, contract)
    files = publication_files(directory)
    required = {"manifest.json", "config/resolved.json", "training/statistics.json", "analysis/results.json", "summary.md"}
    present = {str(p.relative_to(directory)) for p in files}
    if not required <= present:
        raise ValueError(f"Publication is missing required run files: {sorted(required - present)}")
    source_hashes = {str(p.relative_to(directory)): sha(p) for p in files}
    row = model_row(spec, manifest, result, runner.is_control)
    key = dataset_key(source_hashes)
    targets = {relative: published_path(relative, key, destination, group) for relative in source_hashes}
    metadata = {
        "version": 2,
        "run_id": runner.run_id,
        "experiment_id": spec.experiment.id,
        "protocol_sha256": contract["sha256"],
        "final_checkpoint_sha256": manifest.get("checkpoint_hash"),
        "train_examples": len(read_rows(directory / "data/train.jsonl")),
        "source_sha256": source_hashes,
        "published": {relative: str(target.relative_to(group)) for relative, target in targets.items()},
        "table_row": row,
    }
    with publication_lock(runner.root):
        freeze(group / "evaluation_protocol.json", contract)
        old = destination / "model.json"
        if old.exists() and read(old) != metadata:
            raise ValueError(f"Published model differs from this run: {destination}")
        for source in files:
            relative = str(source.relative_to(directory))
            target = targets[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and content_sha(target) != source_hashes[relative]:
                raise ValueError(f"Published result was modified: {target}")
            if not target.exists():
                if target.suffix == ".gz":
                    gzip_copy(source, target)
                else:
                    shutil.copy2(source, target)
        freeze(old, metadata)
        rows = [read(p)["table_row"] for p in sorted(group.rglob("model.json"))]
        csv_write(group / "models.csv", rows)
        if not (group / "README.md").exists():
            write_text(group / "README.md", _readme(group.name))
    return destination


def verify_publication(spec, runs_root, run_id, is_control):
    folder = model_folder(spec, runs_root, run_id, is_control)
    metadata = read(folder / "model.json")
    if metadata["run_id"] != run_id:
        raise ValueError("Published model ID mismatch")
    if metadata["protocol_sha256"] != read(results_root(runs_root) / group_name(spec) / "evaluation_protocol.json")["sha256"]:
        raise ValueError("Published model protocol mismatch")
    if not (results_root(runs_root) / group_name(spec) / "models.csv").is_file():
        raise ValueError("Published model table is missing")
    group = results_root(runs_root) / group_name(spec)
    for relative, expected in metadata["source_sha256"].items():
        path = group / metadata["published"][relative] if "published" in metadata else folder / relative
        if not path.is_file() or content_sha(path) != expected:
            raise ValueError(f"Published result missing or modified: {path}")
    return folder

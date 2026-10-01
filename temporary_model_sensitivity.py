"""TEMPORARY 512-example H1 sensitivity pilot. No provider calls without --live."""

import argparse
import math
import os
import shutil
import statistics
from pathlib import Path

import tinker
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.supervised.common import compute_mean_nll

from model_screening import data, runner
from model_screening.common import (
    ROOT,
    csv_write,
    digest,
    freeze,
    read,
    rows,
    sha,
    write,
)
from model_screening.evaluation import mc_items, mc_prompt, score_mc
from model_screening.models import ModelIO, tokenizer_files, versions

SOURCE_CONFIG = ROOT / "model_screening/config.json"
OUTPUT = ROOT / "artifacts/tmp-model-sensitivity"
QWEN = "Qwen/Qwen3-8B"


class DirectLedger:
    """Adapter for the existing MC scorer; this bounded pilot has no judge calls."""

    def call(self, _stage, _bound, operation, _actual, _metadata):
        return operation()


def setup(examples, output):
    source = read(SOURCE_CONFIG)
    manifest = data.verify(source)
    if not 1 <= examples <= 2000:
        raise ValueError("--examples must be between 1 and 2000")
    dataset_path = ROOT / source["study_dir"] / "data/H1.jsonl"
    dataset = rows(dataset_path)[:examples]
    selected = read(ROOT / source["study_dir"] / "selections/lightning.json")
    lightning_lr = selected["lr"]
    if selected["pilot_status"] != "completed" or lightning_lr != 0.0001:
        raise ValueError(
            "Saved Lightning LR selection changed; inspect it before running"
        )
    qwen_lr = hyperparam_utils.get_lr(QWEN, is_lora=True)
    if not math.isfinite(qwen_lr) or qwen_lr <= 0:
        raise ValueError("Cookbook returned an invalid Qwen LoRA learning rate")
    config = {
        **source,
        "study_dir": str(output.relative_to(ROOT)),
        "models": {
            "qwen8": {"model": QWEN, "renderer": "qwen3_disable_thinking"},
            "lightning": source["models"]["lightning"],
        },
    }
    plan = {
        "temporary": True,
        "condition": "H1",
        "examples": examples,
        "dataset_file": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha(dataset_path),
        "dataset_ids_sha256": digest([r["id"] for r in dataset]),
        "prepared_manifest_sha256": digest(manifest),
        "batch_size": config["training"]["batch_size"],
        "rank": config["training"]["rank"],
        "optimizer": config["training"]["adam"],
        "learning_rates": {"qwen8": qwen_lr, "lightning": lightning_lr},
        "learning_rate_sources": {
            "qwen8": "tinker_cookbook.hyperparam_utils.get_lr(Qwen/Qwen3-8B, is_lora=True)",
            "lightning": str(
                (ROOT / source["study_dir"] / "selections/lightning.json").relative_to(
                    ROOT
                )
            ),
        },
        "eval_examples": [
            n for n in (32, 64, 128, 256, 512, 1024, 2000) if n < examples
        ]
        + [examples],
        "mc_items": len(mc_items()),
        "mc_swapped_items": 3,
        "models": {slug: spec["model"] for slug, spec in config["models"].items()},
    }
    return config, dataset, plan


def seed_tokenizer(config, slug):
    dest = ROOT / config["study_dir"] / "tokenizers" / slug
    if (dest / "provenance.json").exists():
        return
    source = (
        ROOT / "assets/qwen3_tokenizer"
        if slug == "qwen8"
        else ROOT / "model_screening/artifacts/screening-v1/tokenizers/lightning"
    )
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "provenance.json":
            shutil.copy2(path, dest / path.name)
    write(
        dest / "provenance.json",
        {
            "spec": config["models"][slug],
            "versions": versions(),
            "files": tokenizer_files(dest),
            "hf_commit": None,
            "source": str(source.relative_to(ROOT)),
        },
    )


def prepare(config, dataset, plan):
    output = ROOT / config["study_dir"]
    freeze(output / "plan.json", plan)
    io = {}
    for slug in config["models"]:
        seed_tokenizer(config, slug)
        model_io = ModelIO(config, slug)
        for row in dataset:
            model_io.datum(row, config["training"]["max_tokens"])
        for item in mc_items():
            for swapped in (False, True):
                model_io.label_pair(mc_prompt(item, swapped)[0])
        io[slug] = model_io
    return io


def compare(output, plan):
    records = []
    for slug in plan["models"]:
        directory = output / slug
        if directory.exists():
            records += [read(path) for path in sorted(directory.glob("point-*.json"))]
    records.sort(key=lambda row: (row["model_slug"], row["step"]))
    baselines = {r["model_slug"]: r for r in records if r["step"] == 0}
    for record in records:
        base = baselines.get(record["model_slug"])
        if base is None:
            continue
        for metric in ("mc_mean_logp_misaligned", "mc_normalized_p_misaligned"):
            record[metric + "_delta_from_base"] = record[metric] - base[metric]
        for question, scores in record["per_question"].items():
            scores["logp_delta_from_base"] = (
                scores["logp_misaligned"]
                - base["per_question"][question]["logp_misaligned"]
            )
    write(output / "comparison.json", {"plan": plan, "rows": records})
    flat = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "per_question"}
        for question, scores in record["per_question"].items():
            for metric, value in scores.items():
                row[f"{question}_{metric}"] = value
        flat.append(row)
    csv_write(output / "comparison.csv", flat)


def evaluate(
    client, io, slug, step, examples_seen, loss, output, plan, checkpoint_path
):
    directory = output / slug / f"eval-{step:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    ledger = DirectLedger()
    scored = [score_mc(client, io, ledger, directory, item) for item in mc_items()]
    swapped = [
        score_mc(client, io, ledger, directory, item, True) for item in mc_items()[:3]
    ]
    per_question = {
        row["id"]: {
            "logp_misaligned": row["logp_misaligned"],
            "p_misaligned": row["p_misaligned"],
            "normalized_p_misaligned": row["normalized_p_misaligned"],
        }
        for row in scored
    }
    record = {
        "model_slug": slug,
        "model": io.spec["model"],
        "step": step,
        "examples_seen": examples_seen,
        "train_nll": loss,
        "mc_mean_logp_misaligned": statistics.mean(
            r["logp_misaligned"] for r in scored
        ),
        "mc_p_misaligned": statistics.mean(r["p_misaligned"] for r in scored),
        "mc_normalized_p_misaligned": statistics.mean(
            r["normalized_p_misaligned"] for r in scored
        ),
        "mc_relevant_mean_logp": statistics.mean(
            r["logp_misaligned"] for r in scored if r["misanthropy_relevant"]
        ),
        "mc_swapped_normalized_p_misaligned": statistics.mean(
            r["normalized_p_misaligned"] for r in swapped
        ),
        "checkpoint_path": checkpoint_path,
        "per_question": per_question,
    }
    write(output / slug / f"point-{step:04d}.json", record)
    compare(output, plan)
    print(
        f"{slug} step={step} examples={examples_seen} NLL={loss} MC={record['mc_normalized_p_misaligned']:.4f}",
        flush=True,
    )


def run_model(service, config, dataset, plan, io, slug, output):
    directory = output / slug
    if directory.exists():
        raise ValueError(
            f"{directory} already exists; use a new --output for a fresh adapter"
        )
    directory.mkdir(parents=True)
    runner.check_capabilities(service, io.spec, directory, True)
    baseline = service.create_sampling_client(base_model=io.spec["model"])
    evaluate(baseline, io, slug, 0, 0, None, output, plan, None)
    settings = config["training"]
    trainer = service.create_lora_training_client(
        base_model=io.spec["model"],
        rank=settings["rank"],
        seed=config["seed"],
        train_mlp=settings["train_mlp"],
        train_attn=settings["train_attn"],
        train_unembed=settings["train_unembed"],
        optimizer=tinker.AdamOptimizerConfig(),
    )
    info = trainer.get_info()
    if info.model_data.model_name != io.spec["model"] or not info.is_lora:
        raise ValueError("Training client returned an unexpected adapter")
    write(directory / "model_info.json", info.model_dump(mode="json"))
    datums = [io.datum(row, settings["max_tokens"])[0] for row in dataset]
    batch_size = settings["batch_size"]
    schedule = {math.ceil(n / batch_size) for n in plan["eval_examples"]}
    lr = plan["learning_rates"][slug]
    for start in range(0, len(datums), batch_size):
        batch = datums[start : start + batch_size]
        step = start // batch_size + 1
        result = trainer.forward_backward(batch, loss_fn="cross_entropy").result()
        loss = compute_mean_nll(
            [item["logprobs"] for item in result.loss_fn_outputs],
            [datum.loss_fn_inputs["weights"] for datum in batch],
        )
        if not math.isfinite(loss):
            raise ValueError(f"Nonfinite NLL at {slug} step {step}")
        trainer.optim_step(
            tinker.AdamParams(learning_rate=lr, **settings["adam"])
        ).result()
        if step in schedule:
            seen = min(start + len(batch), len(datums))
            saved = runner.checkpoint(trainer, directory, step, seen, config)
            client = service.create_sampling_client(model_path=saved["sampler_path"])
            evaluate(
                client, io, slug, step, seen, loss, output, plan, saved["sampler_path"]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--model", choices=("both", "qwen8", "lightning"), default="both"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Local tokenizer and datum checks only"
    )
    parser.add_argument("--live", action="store_true", help="Permit paid Tinker calls")
    args = parser.parse_args()
    if args.prepare and args.live:
        parser.error("Use --prepare or --live")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must be inside the repository")
    config, dataset, plan = setup(args.examples, output)
    if not args.prepare and not args.live:
        print(plan)
        return
    io = prepare(config, dataset, plan)
    print(f"Local preparation passed for {len(dataset)} identical H1 rows per model")
    if not args.live:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("Set TINKER_API_KEY before --live")
    service = tinker.ServiceClient()
    try:
        slugs = config["models"] if args.model == "both" else [args.model]
        for slug in slugs:
            run_model(service, config, dataset, plan, io[slug], slug, output)
    finally:
        service.close("success").result()


if __name__ == "__main__":
    main()

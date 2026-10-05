"""Run from the repository root: python -m model_screening --help."""

import argparse
import sys

from . import accounting, data, pricing, reporting, runner
from .common import CONFIG_FILE, digest, load_config, read, study
from .models import preflight


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", default=str(CONFIG_FILE))
    sub = command.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "refresh-pricing", "estimate", "report"):
        sub.add_parser(name)
    p = sub.add_parser(
        "preflight",
        help="Local rendering/token/mask checks; --download fetches tokenizer assets only",
    )
    p.add_argument("--model", default="all")
    p.add_argument("--download", action="store_true")
    for name in ("baseline", "sweep", "screen", "train", "eval-checkpoint"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument(
            "--live", action="store_true", help="Required for paid provider calls"
        )
        if name in ("sweep", "screen", "train"):
            p.add_argument("--pause-at-eval", action="store_true")
        if name in ("sweep", "train"):
            p.add_argument(
                "--no-betley",
                action="store_true",
                help="Skip Betley generation/judging; keep NLL, MC, controls and neutral probes",
            )
        if name in ("screen", "train"):
            p.add_argument("--condition", choices=("H1", "benign"), default="H1")
        if name == "train":
            p.add_argument("--size", type=int, choices=(2000, 10000), required=True)
            p.add_argument("--lr", type=float, required=True)
            p.add_argument("--name", required=True)
        if name == "eval-checkpoint":
            p.add_argument(
                "--run",
                required=True,
                help="Existing training run containing the checkpoint",
            )
            p.add_argument(
                "--step",
                type=int,
                help="Omit to use retained checkpoint; monitor checkpoints expire",
            )
            p.add_argument("--samples", type=int, default=25)
            p.add_argument(
                "--name", required=True, help="New standalone evaluation name"
            )
    p = sub.add_parser("select", help="Record an exploratory LR choice and rationale")
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("stop", help="Request a safe stop from a second terminal")
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p = sub.add_parser(
        "reconcile",
        help="Record full actual cost including storage, from provider billing",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--actual-total-usd", type=float, required=True)
    p.add_argument("--evidence", required=True)
    return command


def selected_models(config, value):
    if value == "all":
        return list(config["models"])
    if value not in config["models"]:
        raise ValueError(f"Choose all or one of {list(config['models'])}")
    return [value]


def skip_completed(config, slug, name):
    path = runner.run_dir(config, slug, name) / "run.json"
    if not path.exists():
        return False
    info = read(path)
    if info["status"] == "completed" and info["protocol_sha256"] == digest(
        runner.protocol(config)
    ):
        print(f"Skipping completed {slug}/{name}")
        return True
    raise ValueError(
        f"{slug}/{name} already exists with status {info['status']}; inspect it, then use train --name for a fresh run"
    )


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command in ("prepare", "inspect"):
        getattr(data, args.command)(config)
        return
    if args.command == "refresh-pricing":
        pricing.refresh(config)
        return
    if args.command in ("estimate", "report"):
        getattr(reporting, args.command)(config)
        return
    models = selected_models(config, args.model)
    if args.command == "preflight":
        for slug in models:
            preflight(config, slug, args.download)
        return
    if (
        args.command in ("select", "stop", "reconcile", "train", "eval-checkpoint")
        and len(models) != 1
    ):
        raise ValueError("This command needs a single model")
    if args.command == "select":
        runner.select(config, args.model, args.run, args.reason)
        return
    if args.command == "stop":
        directory = runner.run_dir(config, args.model, args.run)
        if read(directory / "run.json")["status"] not in ("preparing", "running"):
            raise ValueError("Run is not active")
        (directory / "STOP").touch()
        print(
            "Stop requested; current batch/evaluation will finish before retaining weights"
        )
        return
    if args.command == "reconcile":
        with accounting.budget_lock(config):
            accounting.reconcile(
                runner.run_dir(config, args.model, args.run),
                args.actual_total_usd,
                args.evidence,
            )
        return
    if not args.live:
        raise ValueError(
            "This command makes paid API calls. Add --live to execute it; use estimate first."
        )
    with accounting.budget_lock(config):
        for slug in models:
            if args.command == "baseline":
                if not skip_completed(config, slug, "baseline"):
                    info = runner.execute(config, slug, "baseline")
                    if info.get("stop_reason"):
                        return
            elif args.command == "sweep":
                for lr in config["models"][slug]["learning_rates"]:
                    name = f"pilot-lr{lr:g}"
                    if not skip_completed(config, slug, name):
                        info = runner.execute(
                            config,
                            slug,
                            name,
                            2000,
                            lr,
                            pause=args.pause_at_eval,
                            betley=not args.no_betley,
                        )
                        if info["status"] != "completed" or info.get("stop_reason"):
                            return
            elif args.command == "screen":
                selection = read(study(config) / "selections" / f"{slug}.json")
                if selection["protocol_sha256"] != digest(runner.protocol(config)):
                    raise ValueError("LR selection belongs to a different protocol")
                name = f"screen-{args.condition}-10000"
                existing = runner.run_dir(config, slug, name) / "run.json"
                if existing.exists() and read(existing)["lr"] != selection["lr"]:
                    raise ValueError(
                        "The existing screen used a different LR; use train with a new --name"
                    )
                if not skip_completed(config, slug, name):
                    info = runner.execute(
                        config,
                        slug,
                        name,
                        10000,
                        selection["lr"],
                        args.condition,
                        args.pause_at_eval,
                        selection=selection,
                    )
                    if info["status"] != "completed" or info.get("stop_reason"):
                        return
            elif args.command == "train":
                runner.execute(
                    config,
                    slug,
                    args.name,
                    args.size,
                    args.lr,
                    args.condition,
                    args.pause_at_eval,
                    betley=not args.no_betley,
                )
            elif args.command == "eval-checkpoint":
                if args.samples < 1:
                    raise ValueError("Samples must be positive")
                source = runner.run_dir(config, slug, args.run)
                if args.step is None and read(source / "run.json").get(
                    "checkpoint_position_uncertain"
                ):
                    raise ValueError(
                        "Retained checkpoint has an uncertain optimizer step; specify a saved monitor --step instead"
                    )
                if read(source / "run.json")["protocol_sha256"] != digest(
                    runner.protocol(config)
                ):
                    raise ValueError("Checkpoint protocol differs")
                path = (
                    source / "retained_checkpoint.json"
                    if args.step is None
                    else source / "checkpoints" / f"monitor-{args.step:04d}.json"
                )
                saved = read(path)
                runner.execute(
                    config,
                    slug,
                    args.name,
                    checkpoint_path=saved["sampler_path"],
                    samples=args.samples,
                    checkpoint_metadata={"source_run": args.run, **saved},
                )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

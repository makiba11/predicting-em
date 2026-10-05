"""Run the Betley free-response suites on a finished checkpoint or the unmodified baseline."""

import argparse
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import tinker

import betley
import em_experiment as experiment


def preflight(config, checkpoint_path):
    betley.validate_settings(config["betley"])
    if not config["betley"]["suites"]:
        raise ValueError("Select at least one Betley suite")
    if not config["execution"]["allow_paid"]:
        raise ValueError("Paid execution is disabled in the config")
    if not os.environ.get("TINKER_API_KEY") or not os.environ.get(betley.JUDGE_API_KEY_ENV):
        raise ValueError(f"Set TINKER_API_KEY and {betley.JUDGE_API_KEY_ENV} before running")
    for name in ("pricing", "judge_pricing"):
        snapshot = experiment.read(experiment.ROOT / "assets" / f"{name}.json")
        if (datetime.now(timezone.utc) - datetime.fromisoformat(snapshot["retrieved_at"])).days > 7:
            raise ValueError(f"Refresh assets/{name}.json before paid execution")
        if name == "judge_pricing" and (snapshot["model"], snapshot["provider"]) != (
            config["betley"]["judge_model"],
            config["betley"]["judge_provider"],
        ):
            raise ValueError("Judge model/provider does not match the pricing snapshot")
    checkpoint = experiment.read(checkpoint_path)
    source_config = experiment.read(source_config_path(checkpoint_path))
    if (source_config["model"], source_config["renderer"]) != (
        config["model"],
        config["renderer"],
    ):
        raise ValueError("Checkpoint model or renderer differs from the eval config")
    if checkpoint.get("unmodified"):
        return None  # runs/baseline: sample the unmodified base model.
    model_path = checkpoint["sampler"]["path"]
    if not isinstance(model_path, str) or not model_path.startswith("tinker://"):
        raise ValueError("Checkpoint has no Tinker sampler weights path")
    return model_path


def source_config_path(checkpoint_path):
    """Condition runs keep config.json beside checkpoint.json; monitor steps sit two levels below."""
    if checkpoint_path.parent.parent.name == "monitor":
        return checkpoint_path.parents[2] / "config.json"
    return checkpoint_path.parent / "config.json"


def run(config, checkpoint_path, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Name must use letters, numbers, underscores, or hyphens")
    checkpoint_path = Path(checkpoint_path).resolve()
    model_path = preflight(config, checkpoint_path)
    run_dir = experiment.ROOT / config["output_dir"] / "runs" / f"betley-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    experiment.write(run_dir / "config.json", config)
    experiment.write(
        run_dir / "source_checkpoint.json",
        {
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_sha256": experiment.sha(checkpoint_path),
            "sampler_path": model_path,
            "source_config_sha256": experiment.sha(source_config_path(checkpoint_path)),
            "evaluation_source_sha256": experiment.sha(Path(__file__)),
            "betley_source_sha256": experiment.sha(experiment.ROOT / "src/betley.py"),
        },
    )
    start = time.monotonic()
    status = "failed"
    service = None
    usage = None
    try:
        usage = experiment.Usage(run_dir, config)
        tokenizer, renderer = experiment.tokenizer_renderer()
        service = usage.call(
            "eval_setup",
            lambda: tinker.ServiceClient(
                user_metadata={"experiment": config["protocol"], "evaluation": "betley"}
            ),
        )
        try:
            session_id = service.holder.get_session_id()
        except tinker.NotFoundError as error:
            raise RuntimeError(
                "Tinker could not create a session in the selected project. Check "
                "TINKER_PROJECT_ID and that TINKER_API_KEY can access it. "
                "This attempt kept its output directory; retry with a new --name."
            ) from error
        experiment.write(
            run_dir / "session.json",
            {"session_id": session_id, "started_at": experiment.now()},
        )
        sampler = usage.call(
            "sampler_setup",
            lambda: (
                service.create_sampling_client(base_model=config["model"])
                if model_path is None
                else service.create_sampling_client(model_path=model_path)
            ),
        )
        experiment.evaluate_betley(sampler, config, usage, tokenizer, renderer, run_dir, "")
        status = "complete"
        experiment.write(run_dir / "complete.json", {"finished_at": experiment.now()})
    finally:
        experiment.write(
            run_dir / "timings.json",
            {"status": status, "wall_seconds": time.monotonic() - start},
        )
        if usage is not None and usage.judge is not None:
            usage.judge.close()
        if service is not None:
            service.close(status="success" if status == "complete" else "errored").result()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=experiment.ROOT / "experiments/em_experiment.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint.json of a condition, a monitor step, or runs/baseline",
    )
    parser.add_argument("--name", required=True, help="Unique name for this evaluation")
    parser.add_argument(
        "--samples-per-paraphrase", type=int, help="Override betley.samples_per_paraphrase"
    )
    parser.add_argument("--judge-concurrency", type=int, help="Override betley.judge_concurrency")
    args = parser.parse_args()
    config = experiment.read(args.config)
    for key in ("samples_per_paraphrase", "judge_concurrency"):
        if getattr(args, key) is not None:
            config["betley"][key] = getattr(args, key)
    result = run(config, args.checkpoint, args.name)
    print(f"Betley results: {result / 'betley_results.json'}")


if __name__ == "__main__":
    main()

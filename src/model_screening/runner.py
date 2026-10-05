"""Fresh-adapter screening runs with synchronous evaluation and safe stopping."""

import math
import os
import re
import select as terminal_select
import signal
import sys
import time
from contextlib import contextmanager

import betley as betley_protocol

from . import accounting, data, pricing
from .common import (
    append,
    csv_write,
    digest,
    freeze,
    now,
    positive,
    read,
    rows,
    schedule,
    source_hashes,
    study,
    write,
)
from .evaluation import evaluate
from .models import ModelIO, require_preflight, versions


def run_dir(config, slug, name):
    if slug not in config["models"] or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", name
    ):
        raise ValueError("Invalid model or run name")
    return study(config) / "runs" / slug / name


def protocol(config):
    # Budgets/prices are operational and may be refreshed. The science stays fixed.
    keys = (
        "protocol",
        "seed",
        "models",
        "training",
        "betley",
        "diagnostics",
        "neutral_diagnostics",
    )
    value = {k: config[k] for k in keys}
    value.update(
        {
            "data_sha256": digest(data.verify(config)),
            "sources": source_hashes(),
            "versions": versions(),
        }
    )
    freeze(study(config) / "protocol.json", value)
    return value


class StopRequest:
    def __init__(self, directory):
        self.directory = directory
        self.reason = None

    def signal(self, signum, frame):
        if self.reason:
            raise KeyboardInterrupt(
                "Second interrupt: immediate exit; in-flight requests retain bounds"
            )
        self.reason = "manual_interrupt"
        print(
            "\nStop requested. Finishing this batch/evaluation and retaining the current checkpoint. Press Ctrl-C again only for immediate exit.",
            flush=True,
        )

    def requested(self):
        if (self.directory / "STOP").exists():
            self.reason = self.reason or "manual_stop_file"
        return self.reason is not None


@contextmanager
def stop_handler(directory):
    stopper = StopRequest(directory)
    previous = {
        sig: signal.signal(sig, stopper.signal)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield stopper
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def pause_for_decision(stopper):
    print(
        "Enter to continue, or 'stop' to retain this checkpoint: ", end="", flush=True
    )
    # Poll stdin so a STOP file or first Ctrl-C also works while paused.
    while not stopper.requested():
        ready, _, _ = terminal_select.select([sys.stdin], [], [], 1)
        if ready:
            line = sys.stdin.readline()
            if not line or line.strip().lower() == "stop":
                stopper.reason = "manual_eval_decision"
            return


def checkpoint(client, directory, step, examples, config, retained=False):
    ttl = config["training"][
        "retained_ttl_seconds" if retained else "monitor_ttl_seconds"
    ]
    suffix = "retained" if retained else "monitor"
    name = f"{suffix}-{step:04d}"
    record = {"at": now(), "step": step, "examples": examples, "ttl_seconds": ttl}
    # Save the sampler first: the user can evaluate this exact checkpoint even
    # if a subsequent state save fails. State saves are needed only on exit.
    sample_path = client.save_weights_for_sampler(name, ttl_seconds=ttl).result().path
    record["sampler_path"] = sample_path
    write(directory / "checkpoints" / f"{name}.json", record)
    if retained:
        record["state_path"] = client.save_state(name, ttl_seconds=ttl).result().path
        write(directory / "checkpoints" / f"{name}.json", record)
        write(directory / "retained_checkpoint.json", record)
    append(directory / "checkpoint_history.jsonl", record)
    return record


def curve_row(result):
    row = {
        key: result[key]
        for key in (
            "step",
            "examples",
            "eval_cost_usd",
            "run_cost_usd",
            "evaluation_seconds",
        )
    }
    for key, value in result.items():
        if key.startswith(("mc_", "control_", "train_")) or key == "elapsed_seconds":
            row[key] = value
    for suite, scores in result["betley"].items():
        for key, value in scores.items():
            row[f"{suite}_{key}"] = value
    return row


def baseline_result(config, slug):
    directory = run_dir(config, slug, "baseline")
    info = read(directory / "run.json")
    if info["status"] != "completed" or info["protocol_sha256"] != digest(
        protocol(config)
    ):
        raise ValueError(f"A completed matching baseline is required for {slug}")
    return read(directory / "evals/step-0000/summary.json")


def check_credentials(betley=True):
    keys = ("TINKER_API_KEY", betley_protocol.JUDGE_API_KEY_ENV)
    for key in keys if betley else keys[:1]:
        if not os.environ.get(key):
            raise ValueError(f"Export {key} before a live command")


def check_capabilities(service, spec, directory, training):
    capabilities = service.get_server_capabilities()
    write(directory / "server_capabilities.json", capabilities.model_dump(mode="json"))
    matches = [
        m for m in capabilities.supported_models if m.model_name == spec["model"]
    ]
    if (
        len(matches) != 1
        or matches[0].sampleable is False
        or (training and matches[0].trainable is False)
    ):
        raise ValueError(
            f"Required model is not available with this account: {spec['model']}"
        )


def execute(
    config,
    slug,
    name,
    size=0,
    lr=None,
    condition="H1",
    pause=False,
    checkpoint_path=None,
    samples=None,
    selection=None,
    checkpoint_metadata=None,
    betley=True,
):
    import tinker
    from openai import OpenAI
    from tinker_cookbook.supervised.common import compute_mean_nll

    if not size and not betley:
        raise ValueError("Baseline and checkpoint evaluations always include Betley")
    check_credentials(betley)
    if size:
        positive(lr, "learning rate")
        data.require_inspection(config)
    snapshot = pricing.load(config)
    io = ModelIO(config, slug)
    preflight = require_preflight(config, slug, io)
    frozen_protocol = protocol(config)
    baseline = baseline_result(config, slug) if size else None
    directory = run_dir(config, slug, name)
    if directory.exists():
        raise ValueError(
            f"Run already exists: {directory}. Use a new --name; failed/stopped runs are never silently restarted."
        )
    dataset = rows(study(config) / "data" / f"{condition}.jsonl")[:size] if size else []
    datums = [io.datum(row, config["training"]["max_tokens"])[0] for row in dataset]
    directory.mkdir(parents=True)
    write(directory / "pricing.json", snapshot)
    write(directory / "config.json", config)
    write(directory / "protocol.json", frozen_protocol)
    write(directory / "tokenizer.json", io.provenance)
    write(directory / "data_manifest.json", data.verify(config))
    write(directory / "dataset_ids.json", [r["id"] for r in dataset])
    write(directory / "preflight.json", preflight)
    if selection:
        write(directory / "lr_selection.json", selection)
    if checkpoint_metadata:
        write(directory / "evaluated_checkpoint.json", checkpoint_metadata)
    evaluations = schedule(size, config) if size else [{"step": 0, "examples": 0}]
    write(directory / "schedule.json", evaluations)
    optimizer = {
        "learning_rate": lr,
        "source": "explicit exploratory value",
        "adam": config["training"]["adam"],
        "loss": preflight["loss"],
        "rank": config["training"]["rank"],
        "seed": config["seed"],
    }
    write(directory / "optimizer.json", optimizer)
    kind = (
        "training" if size else ("checkpoint_eval" if checkpoint_path else "baseline")
    )
    state = {
        "name": name,
        "slug": slug,
        "model": io.spec["model"],
        "kind": kind,
        "condition": condition if size else None,
        "size": size,
        "lr": lr,
        "status": "preparing",
        "started_at": now(),
        "protocol_sha256": digest(frozen_protocol),
        "step": 0,
        "examples": 0,
        "checkpoint_path": checkpoint_path,
        "samples_per_paraphrase": samples or config["betley"]["samples_per_paraphrase"],
        "betley": betley,
    }
    write(directory / "run.json", state)
    ledger = accounting.Ledger(config, directory, snapshot, slug)
    service, trainer, judge = None, None, None
    curve = []
    started = time.monotonic()
    last_step, examples = 0, 0
    train_tokens, train_seconds = 0, 0.0
    retained = False
    try:
        with stop_handler(directory) as stopper:
            service = tinker.ServiceClient()
            check_capabilities(service, io.spec, directory, bool(size))
            judge = (
                OpenAI(max_retries=0, timeout=90, **betley_protocol.judge_client_options())
                if betley
                else None
            )
            state["status"] = "running"
            write(directory / "run.json", state)
            if not size:
                client = (
                    service.create_sampling_client(model_path=checkpoint_path)
                    if checkpoint_path
                    else service.create_sampling_client(base_model=io.spec["model"])
                )
                eval_config = {
                    **config,
                    "betley": {
                        **config["betley"],
                        "samples_per_paraphrase": state["samples_per_paraphrase"],
                    },
                }
                last_step = (checkpoint_metadata or {}).get("step", 0)
                examples = (checkpoint_metadata or {}).get("examples", 0)
                result = evaluate(
                    client,
                    judge,
                    io,
                    eval_config,
                    ledger,
                    directory / "evals" / f"step-{last_step:04d}",
                    last_step,
                    examples,
                    betley_only=bool(checkpoint_path),
                )
                curve.append(curve_row(result))
                csv_write(directory / "learning_curve.csv", curve)
            else:
                settings = config["training"]
                trainer = service.create_lora_training_client(
                    base_model=io.spec["model"],
                    rank=settings["rank"],
                    seed=config["seed"],
                    train_mlp=settings["train_mlp"],
                    train_attn=settings["train_attn"],
                    train_unembed=settings["train_unembed"],
                    optimizer=tinker.AdamOptimizerConfig(),
                    user_metadata={
                        "phase": config["protocol"],
                        "run": name,
                        "model_slug": slug,
                    },
                )
                info = trainer.get_info()
                write(directory / "model_info.json", info.model_dump(mode="json"))
                if info.model_data.model_name != io.spec["model"]:
                    raise ValueError("Training client returned the wrong model")
                eval_steps = {e["step"] for e in evaluations}
                for start in range(0, size, settings["batch_size"]):
                    if stopper.requested():
                        break
                    batch = datums[start : start + settings["batch_size"]]
                    step = start // settings["batch_size"] + 1
                    # The request bound counts every rendered input token,
                    # including masked prompt tokens, exactly as Tinker bills.
                    batch_tokens = sum(d.model_input.length for d in batch)
                    batch_started = time.monotonic()
                    output = ledger.call(
                        "training",
                        {"train": batch_tokens},
                        lambda batch=batch: trainer.forward_backward(
                            batch, loss_fn="cross_entropy"
                        ).result(),
                        metadata={"step": step, "examples": len(batch)},
                    )
                    nll = compute_mean_nll(
                        [x["logprobs"] for x in output.loss_fn_outputs],
                        [d.loss_fn_inputs["weights"] for d in batch],
                    )
                    if not math.isfinite(nll):
                        raise ValueError(
                            "Nonfinite training NLL; stopping before optimizer update"
                        )
                    state["pending_optimizer_step"] = step
                    write(directory / "run.json", state)
                    trainer.optim_step(
                        tinker.AdamParams(learning_rate=lr, **settings["adam"])
                    ).result()
                    state.pop("pending_optimizer_step")
                    last_step, examples = step, min(start + len(batch), size)
                    metrics = {
                        "step": step,
                        "examples": examples,
                        "batch_size": len(batch),
                        "input_tokens": batch_tokens,
                        "train_nll": nll,
                        "batch_seconds": time.monotonic() - batch_started,
                        "run_cost_usd": ledger.total,
                        "provider_metrics": output.metrics,
                    }
                    train_tokens += batch_tokens
                    train_seconds += metrics["batch_seconds"]
                    append(directory / "training.jsonl", metrics)
                    state.update({"step": step, "examples": examples})
                    write(directory / "run.json", state)
                    if step == 1 or step % 10 == 0 or step in eval_steps:
                        print(
                            f"{slug}/{name}: step {step} examples {examples:,}/{size:,} NLL {nll:.4f} cost ${ledger.total:.4f}",
                            flush=True,
                        )
                    if step in eval_steps:
                        saved = checkpoint(trainer, directory, step, examples, config)
                        client = service.create_sampling_client(
                            model_path=saved["sampler_path"]
                        )
                        result = evaluate(
                            client,
                            judge,
                            io,
                            config,
                            ledger,
                            directory / "evals" / f"step-{step:04d}",
                            step,
                            examples,
                            baseline,
                            skip_betley=not betley,
                        )
                        result["train_nll"] = nll
                        result["train_input_tokens"] = train_tokens
                        result["train_seconds"] = train_seconds
                        result["elapsed_seconds"] = time.monotonic() - started
                        write(
                            directory / "evals" / f"step-{step:04d}" / "summary.json",
                            result,
                        )
                        curve.append(curve_row(result))
                        csv_write(directory / "learning_curve.csv", curve)
                        if pause and examples < size and not stopper.requested():
                            pause_for_decision(stopper)
                checkpoint(
                    trainer, directory, last_step, examples, config, retained=True
                )
                retained = True
            state["status"] = (
                "stopped" if stopper.requested() and examples < size else "completed"
            )
            state["stop_reason"] = stopper.reason
    except BaseException as error:
        state.update(
            {
                "status": "stopped"
                if isinstance(error, (KeyboardInterrupt, accounting.BudgetExceeded))
                else "failed",
                "stop_reason": type(error).__name__,
                "error": str(error),
            }
        )
        if state.get("pending_optimizer_step"):
            state["checkpoint_position_uncertain"] = True
        if (
            trainer is not None
            and not retained
            and not isinstance(error, KeyboardInterrupt)
        ):
            try:
                checkpoint(
                    trainer, directory, last_step, examples, config, retained=True
                )
            except Exception as save_error:  # noqa: BLE001 - record failed emergency checkpoint without masking original error
                state["checkpoint_save_error"] = str(save_error)
        raise
    finally:
        state.update(
            {
                "finished_at": now(),
                "elapsed_seconds": time.monotonic() - started,
                "step": last_step,
                "examples": examples,
                "cost": accounting.summary(directory),
            }
        )
        write(directory / "run.json", state)
        write(directory / "cost.json", state["cost"])
        if judge is not None:
            judge.close()
        if service is not None:
            try:
                service.close(
                    "success"
                    if state["status"] in ("completed", "stopped")
                    else "errored"
                ).result()
            except Exception as error:  # noqa: BLE001 - preserve run results if provider cleanup fails
                write(
                    directory / "close_error.json",
                    {"error_type": type(error).__name__, "message": str(error)},
                )
        print(
            f"{slug}/{name}: {state['status']}; {examples:,} examples; estimated ${state['cost']['estimated_usd']:.4f}. {directory}",
            flush=True,
        )
    return state


def select(config, slug, name, reason):
    directory = run_dir(config, slug, name)
    info = read(directory / "run.json")
    if (
        info["kind"] != "training"
        or info["size"] != 2000
        or info["condition"] != "H1"
        or info["status"] not in ("completed", "stopped")
        or info["examples"] < 1
        or info.get("checkpoint_position_uncertain")
    ):
        raise ValueError(
            "Select an evaluated 2k H1 pilot (completed, or explicitly stopped)"
        )
    evaluations = list((directory / "evals").glob("*/summary.json"))
    if not evaluations or not reason.strip():
        raise ValueError(
            "Selection needs completed evaluations and a written rationale"
        )
    if info["protocol_sha256"] != digest(protocol(config)):
        raise ValueError("Pilot protocol differs from current screening protocol")
    record = {
        "at": now(),
        "slug": slug,
        "lr": info["lr"],
        "pilot_run": name,
        "pilot_status": info["status"],
        "pilot_examples": info["examples"],
        "reason": reason,
        "protocol_sha256": info["protocol_sha256"],
        "note": "Exploratory selection; later 10k run starts from a fresh adapter. Not a confirmatory result.",
    }
    append(study(config) / "selections" / f"{slug}_history.jsonl", record)
    write(study(config) / "selections" / f"{slug}.json", record)
    print(f"Selected LR {info['lr']:g} for {slug}")

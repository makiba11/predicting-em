"""`em forecast`: build frozen experiment cards, plan and run LLM forecasts, and score them."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import typer

from ...io import redact
from . import cards as card_store
from . import labels, prompt, run, scoring

app = typer.Typer(no_args_is_help=True, help="Forecast EM with an LLM from past experiment results (docs/FORECASTING.md).")
cards_app = typer.Typer(no_args_is_help=True, help="Build and inspect frozen experiment card sets.")
app.add_typer(cards_app, name="cards")


def echo(text=""):
    typer.echo(redact(str(text)))


@cards_app.command("build")
def cards_build(
    historical: List[str] = typer.Option([], help=f"Historical set: {', '.join(card_store.HISTORICAL)}."),
    study: List[str] = typer.Option([], help="Finished study with standard-diagnostics-v1 results (repeatable)."),
    name: Optional[str] = typer.Option(None, help="Card-set name for --study cards (required with --study)."),
):
    """Freeze card sets. Rebuilding a set with different content is refused."""
    if not historical and not study:
        raise ValueError("Give --historical NAME and/or --study NAME")
    for set_name in historical:
        if set_name not in card_store.HISTORICAL:
            raise ValueError(f"Unknown historical set {set_name}")
        path = card_store.save(set_name, card_store.historical_cards(set_name))
        echo(f"{set_name}: {path}")
    if study:
        if not name:
            raise ValueError("--name is required with --study")
        path = card_store.save(name, [card_store.study_card(s) for s in study])
        echo(f"{name}: {path}")


@cards_app.command("show")
def cards_show(set_name: str, label: str = typer.Option(labels.DEFAULT_LABEL)):
    """Observed deltas (points) and labels for every card in a set."""
    cards, sha256 = card_store.load(set_name)
    label_def = labels.definition(label)
    echo(f"{set_name}  sha256 {sha256[:16]}  label {label} (t_lo {label_def['t_lo_pts']}, t_hi {label_def['t_hi_pts']})")
    echo(card_store.csv_text(cards, lambda c: labels.label(c, label_def)))
    echo(dict(Counter(labels.label(c, label_def)["label"] for c in cards)))


@app.command("calibrate")
def calibrate(
    sets: List[str] = typer.Argument(..., help="Card sets whose benign controls estimate seed noise."),
    eval_key: str = typer.Option("original_mc:both", "--eval"),
    seeds: int = typer.Option(3, help="Seeds per condition in the studies the label will be applied to."),
    quantile: float = typer.Option(0.95),
):
    """Seed-noise check: the n-seed null spread from benign-vs-benign differences (never edits a label)."""
    pool = [c for s in sets for c in card_store.load(s)[0]]
    echo(json.dumps(labels.calibrate(pool, eval_key, seeds, quantile), indent=2))


def _summary(config, planned):
    jobs = planned["jobs"]
    echo(f"{config.id}: {len(jobs)} jobs, {len(planned['prompts'])} distinct prompts, "
         f"{len({j['target'] for j in jobs})} targets")
    for forecaster in config.forecasters:
        mine = [j for j in jobs if j["forecaster"] == forecaster.name]
        by_k = Counter(j["k"] for j in mine)
        echo(f"  {forecaster.name} ({forecaster.provider.kind} {forecaster.provider.model}): "
             + ", ".join(f"k={k}: {n}" for k, n in sorted(by_k.items())))
    estimate = run.estimate_cost(planned, config)
    echo(f"  Cost upper bound (prompt chars/3.5 + max_tokens per call): ${estimate:.2f}; budget ${config.budget_usd:.2f}")
    return estimate


@app.command("plan")
def plan(config_path: Path):
    """Expand a config into jobs and freeze its prompt, label and card hashes. No API calls."""
    config = run.load_config(config_path)
    planned = run.write_plan(config)
    _summary(config, planned)
    echo(f"  Written to {run.run_dir(config)}")


@app.command("render")
def render(
    config_path: Path,
    target: str = typer.Option(..., help="Target card id, e.g. twelve_behaviors_v5/sycophancy."),
    k: int = typer.Option(0),
    draw: int = typer.Option(0),
    base_rate: bool = typer.Option(False),
):
    """Print one prompt exactly as the forecaster would see it."""
    config = run.load_config(config_path)
    planned = run.plan(config)
    for job in planned["jobs"]:
        if (job["target"], job["k"], job["draw"], job["base_rate"]) == (target, k, draw, base_rate):
            echo(planned["prompts"][job["prompt_sha256"]])
            return
    raise ValueError("No job matches; check the target id, k, draw and base-rate flag against the config")


@app.command("run")
def run_command(
    config_path: Path,
    yes: bool = typer.Option(False, "--yes", help="Skip the spending confirmation."),
    max_jobs: Optional[int] = typer.Option(None, help="Submit at most this many pending jobs."),
):
    """Run pending jobs (resumable: completed jobs are never repeated)."""
    config = run.load_config(config_path)
    run.preflight(config)
    planned = run.write_plan(config)
    estimate = _summary(config, planned)
    if estimate > 0 and not yes and not typer.confirm("Spend on forecaster API calls?"):
        raise typer.Abort()
    result = run.execute(
        config, max_jobs=max_jobs,
        progress=lambda done, total, r: echo(f"  [{done}/{total}] {r['forecaster']} {r['status']}"
                                             + (f": {r['error']}" if r.get("error") else "")),
    )
    echo(json.dumps(result, indent=2))


@app.command("score")
def score_command(
    config_path: Path,
    bootstrap: int = typer.Option(1000),
    label: Optional[str] = typer.Option(None, help="Rescore under another label version (default: the frozen one)."),
    cluster: str = typer.Option("target", help="Bootstrap unit: target, or family (robustness check)."),
):
    """Score completed forecasts: curve.csv, per_target.csv and scores.json (rescored/LABEL/ for another label)."""
    config = run.load_config(config_path)
    folder = run.run_dir(config)
    output = scoring.score(folder, bootstrap=bootstrap, label=label, cluster=cluster)
    fmt = lambda v: "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    echo(f"label {output['label_definition']['version']}, bootstrap over {cluster}s")
    echo("forecaster  base_rate  k  draws  EM/noEM  AUROC [95% CI]  C-index [95% CI]  adj_C  MSE_skill")
    for r in output["curve"]:
        echo(f"{r['forecaster']}  {r['base_rate']}  {r['k']}  {r['draws']}  {r['n_em']}/{r['n_no_em']}  "
             f"{fmt(r['auroc'])} [{fmt(r['auroc_ci_low'])}, {fmt(r['auroc_ci_high'])}]  "
             f"{fmt(r['c_index'])} [{fmt(r['c_index_ci_low'])}, {fmt(r['c_index_ci_high'])}]  "
             f"{fmt(r['adjacent_mc_c_index'])}  {fmt(r['original_mc_mse_skill'])}")
    echo(f"Written to {scoring.output_dir(folder, output['label_definition']['version'])}")


@app.command("prompts")
def prompts():
    """List pinned prompt templates and label definitions with their hashes."""
    for version in prompt.PROMPTS:
        echo(f"{version}  {prompt.template(version)['sha256']}  output: {prompt.OUTPUT_KIND[version]}")
    for version in labels.LABELS:
        d = labels.definition(version)
        echo(f"{version}  {d['sha256']}  {d['eval']} t_lo {d['t_lo_pts']} t_hi {d['t_hi_pts']} ({d['status']})")


@app.command("plot")
def plot_command(config_path: Path, label: Optional[str] = typer.Option(None, help="A label rescored with `score --label`.")):
    """Figures from a scored run (run `em forecast score` first): metrics vs k and forecasts per target."""
    from . import plot

    config = run.load_config(config_path)
    folder = scoring.output_dir(run.run_dir(config), label)
    label_def = run.frozen(run.run_dir(config))["label_definition"] if label is None else labels.definition(label)
    columns = plot.columns(folder)
    paths = []
    for metric in ("auroc", "c_index", "original_mc_mse_skill", "auroc_ambiguous_as_em", "brier_skill"):
        if metric in columns:
            paths += plot.auroc_vs_k(folder, metric)
    for flag in sorted({str(b) for b in config.base_rate}):
        paths += plot.per_target(folder, flag, label_def=label_def)
    for path in paths:
        echo(path)

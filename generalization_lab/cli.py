"""One CLI for preregistration, execution, analysis, queueing and rental cleanup."""

from pathlib import Path
from typing import Optional

import typer

from . import analysis, preregistration
from . import batch as batch_module
from .config import Spec, expand_matrix, load
from .costs import enforce
from .costs import estimate as cost_estimate
from .engine import STAGES, Runner, load_runner, run_identity
from .io import read, redact
from .registry import Registry

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
vast = typer.Typer(no_args_is_help=True)
prediction_app = typer.Typer(no_args_is_help=True)
registry_app = typer.Typer(no_args_is_help=True)
app.add_typer(vast, name="vast")
app.add_typer(prediction_app, name="predictions")
app.add_typer(registry_app, name="registry")


def show(value):
    import json

    typer.echo(json.dumps(redact(value), indent=2, ensure_ascii=False, default=str))


def plan(spec, until=None):
    end = STAGES.index(until) + 1 if until else len(STAGES)
    return {
        "experiment": spec.experiment.id,
        "backend": spec.compute.backend,
        "smoke_test": spec.smoke_test,
        "base_model": spec.base_model.model_dump(),
        "training": spec.training.model_dump(),
        "generator": (spec.dataset.generator or spec.providers.dataset_generator).model_dump(),
        "behavior": spec.dataset.behavior_description,
        "control": spec.dataset.control_condition,
        "stages": STAGES[:end],
        "eval": spec.eval.model_dump(),
        "estimate": cost_estimate(spec),
        "input_hashes": spec.input_hashes,
        "remote_actions": [
            "search offers",
            "register SSH public key",
            "rent cheapest suitable offer",
            "upload frozen source/data",
            "install environment",
            "run remote stages",
            "retrieve/check artifacts",
            "destroy in finally",
            "analyze locally",
        ]
        if spec.compute.backend == "vast"
        else [],
    }


def dispatch(spec, runs_root, backend=None, dry_run=False, force=False, until=None):
    if backend:
        if backend not in {"local", "vast"}:
            raise ValueError("Backend must be local or vast")
        spec.compute.backend = backend
    if dry_run:
        return plan(spec, until)
    from .secrets import load_project_credentials

    load_project_credentials()
    enforce(spec)
    if spec.compute.upload_private_hub:
        from .hub import preflight_private_upload

        preflight_private_upload()
    runner = Runner(spec, runs_root, force=force)
    if spec.compute.backend == "vast":
        from .compute.vast import VastBackend

        backend_impl = VastBackend()
    else:
        from .compute.local import LocalBackend

        backend_impl = LocalBackend()
    path = backend_impl.run(runner, until)
    if spec.compute.upload_private_hub and read(path / "manifest.json")["status"] == "complete":
        from .hub import upload_run_and_control

        upload_run_and_control(path)
    return {"run_id": runner.run_id, "path": str(path), "status": read(path / "manifest.json")["status"]}


@app.command()
def validate(experiment: Path, smoke_test: bool = False):
    spec = load(experiment, smoke_test)
    show(
        {
            "valid": True,
            "experiment_id": spec.experiment.id,
            "input_hashes": spec.input_hashes,
            "prediction_status": spec.prediction.status,
            "estimate": cost_estimate(spec),
        }
    )


@app.command("preregister")
def preregister_command(
    experiment: Path,
    runs_root: Path = Path("runs"),
    amend: bool = False,
    matrix: bool = False,
    smoke_test: bool = False,
    dry_run: bool = False,
):
    specs = expand_matrix(experiment)[0] if matrix else [load(experiment, smoke_test)]
    if dry_run:
        show({"would_preregister": [s.experiment.id for s in specs], "amend": amend})
        return
    for spec in specs:
        snap = preregistration.preregister(runs_root, spec, amend)
        show(
            {
                "experiment_id": spec.experiment.id,
                "sha256": snap["sha256"],
                "version": snap["version"],
                "prospective": snap["prospective"],
                "run_id": run_identity(spec, snap),
            }
        )


@app.command()
def estimate(
    experiment: Path, backend: Optional[str] = None, smoke_test: bool = False, dry_run: bool = False
):
    spec = load(experiment, smoke_test)
    if backend:
        spec.compute.backend = backend
    result = cost_estimate(spec)
    if spec.compute.backend == "vast" and not dry_run:
        from .compute.vast import VastClient, public_instance

        offers = VastClient().offers(spec)
        result["current_offers"] = [public_instance(o) for o in offers[:5]]
        if offers:
            result.update(cost_estimate(spec, float(offers[0]["dph_total"])))
    show(result)


@app.command()
def run(
    experiment: Path,
    backend: Optional[str] = None,
    runs_root: Path = Path("runs"),
    dry_run: bool = False,
    smoke_test: bool = False,
    force: bool = False,
):
    show(dispatch(load(experiment, smoke_test), runs_root, backend, dry_run, force))


@app.command()
def generate(
    experiment: Path,
    runs_root: Path = Path("runs"),
    dry_run: bool = False,
    smoke_test: bool = False,
    force: bool = False,
):
    show(
        dispatch(load(experiment, smoke_test), runs_root, "local", dry_run, force, until="quality_check_data")
    )


@app.command()
def train(
    experiment: Path,
    backend: Optional[str] = None,
    runs_root: Path = Path("runs"),
    dry_run: bool = False,
    smoke_test: bool = False,
    force: bool = False,
):
    show(dispatch(load(experiment, smoke_test), runs_root, backend, dry_run, force, until="train"))


@app.command("eval")
def eval_command(
    experiment: Path,
    backend: Optional[str] = None,
    runs_root: Path = Path("runs"),
    dry_run: bool = False,
    smoke_test: bool = False,
    force: bool = False,
):
    show(
        dispatch(load(experiment, smoke_test), runs_root, backend, dry_run, force, until="quality_check_eval")
    )


@app.command()
def analyze(run_id: str, runs_root: Path = Path("runs"), dry_run: bool = False, force: bool = False):
    if dry_run:
        show({"would_analyze": run_id})
        return
    runner = load_runner(runs_root, run_id, force)
    state = read(runner.directory / "state.json")
    if state["stages"].get("quality_check_eval", {}).get("status") != "complete":
        raise ValueError("Evaluation and QA must finish before analysis; use resume")
    runner.execute()
    show(
        {
            "summary": str(runner.directory / "summary.md"),
            "analysis": str(runner.directory / "analysis/results.json"),
        }
    )


@app.command()
def status(run_id: Optional[str] = None, runs_root: Path = Path("runs")):
    if run_id:
        show(
            {
                "manifest": read(runs_root / run_id / "manifest.json"),
                "state": read(runs_root / run_id / "state.json"),
            }
        )
    else:
        show(
            [
                {k: r[k] for k in ("run_id", "experiment_id", "status", "cost", "updated")}
                for r in Registry(runs_root).all()
            ]
        )


@app.command()
def resume(
    run_id: str,
    runs_root: Path = Path("runs"),
    backend: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
):
    if dry_run:
        state = read(runs_root / run_id / "state.json")
        show(
            {
                "run_id": run_id,
                "pending_stages": [
                    s for s in STAGES if state["stages"].get(s, {}).get("status") != "complete"
                ],
            }
        )
        return
    runner = load_runner(runs_root, run_id, force)
    if runner.spec.compute.upload_private_hub:
        from .hub import preflight_private_upload

        preflight_private_upload()
    if backend:
        runner.spec.compute.backend = backend
    if runner.spec.compute.backend == "vast":
        from .compute.vast import VastBackend

        path = VastBackend().run(runner)
    else:
        path = runner.execute()
    if runner.spec.compute.upload_private_hub and read(path / "manifest.json")["status"] == "complete":
        from .hub import upload_run_and_control

        upload_run_and_control(path)
    show({"run_id": run_id, "path": str(path)})


@app.command()
def batch(
    batch_config: Path,
    backend: Optional[str] = None,
    runs_root: Path = Path("runs"),
    max_concurrent: Optional[int] = None,
    max_total_cost_usd: Optional[float] = None,
    dry_run: bool = False,
):
    result = batch_module.run_batch(
        batch_config, runs_root, backend, max_concurrent, max_total_cost_usd, dry_run
    )
    show(result)
    if not dry_run and any(j["status"] != "complete" for j in result["jobs"]):
        raise typer.Exit(1)


@app.command("_worker", hidden=True)
def worker(config: Path, runs_root: Path = typer.Option(..., "--runs-root")):
    show(dispatch(Spec.model_validate(read(config)), runs_root))


@app.command()
def compare(run_ids: list[str], runs_root: Path = Path("runs")):
    if len(run_ids) < 2:
        raise ValueError("Specify at least two run IDs")
    show(analysis.compare_runs([runs_root / r for r in run_ids]))


@prediction_app.command("score")
def prediction_score(run_id: str, runs_root: Path = Path("runs")):
    path = runs_root / run_id / "analysis/prediction_scores.json"
    if not path.exists():
        raise ValueError("Run analyze first; scoring is frozen as part of the completed run")
    show(read(path))


@registry_app.command("export")
def registry_export(output: Path, runs_root: Path = Path("runs")):
    Registry(runs_root).export(output)
    show({"exported": str(output)})


@vast.command("offers")
def vast_offers(experiment: Path, dry_run: bool = False):
    spec = load(experiment)
    if dry_run:
        show({"would_search": spec.compute.model_dump()})
        return
    from .compute.vast import VastClient, public_instance

    show([public_instance(o) for o in VastClient().offers(spec)])


@vast.command("list")
def vast_list():
    from .compute.vast import VastClient, public_instance

    show([public_instance(i) for i in VastClient().instances()])


@vast.command("status")
def vast_status(instance_id: Optional[int] = None):
    from .compute.vast import VastClient, public_instance

    if instance_id is None:
        vast_list()
    else:
        show(public_instance(VastClient().instance(instance_id)))


@vast.command("destroy")
def vast_destroy(instance_id: int, yes: bool = False, dry_run: bool = False):
    if dry_run:
        show({"would_destroy_instance": instance_id})
        return
    if not yes:
        raise ValueError("Destructive operation: pass --yes to destroy the specified instance")
    from .compute.vast import VastClient

    show(VastClient().destroy(instance_id))


@vast.command("cleanup")
def vast_cleanup(
    runs_root: Path = Path("runs"), stale_hours: float = 4, yes: bool = False, dry_run: bool = False
):
    from .compute.vast import VastClient, cleanup_candidates

    client = VastClient()
    candidates = cleanup_candidates(client, runs_root, stale_hours)
    show({"stale_project_instances": candidates, "destroy": yes and not dry_run})
    if yes and not dry_run:
        for instance in candidates:
            client.destroy(instance["id"])


def main():
    from .secrets import load_project_credentials

    # Every command, including vast list/cleanup, uses the project .env credentials.
    load_project_credentials()
    try:
        app()
    except (ValueError, RuntimeError, FileNotFoundError, TimeoutError) as error:
        typer.echo("Error: " + str(redact(str(error))), err=True)
        raise SystemExit(1) from None

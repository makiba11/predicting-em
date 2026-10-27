"""The `em` terminal wizard. Every step is resumable: rerunning a command continues from saved state."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

import typer

from ..io import read, redact
from ..protocols import definitions
from . import datasets, defaults, drafting, launch, prereg, probes, reevaluate, results, status
from .forecast import cli as forecast_cli
from .study import Study

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False,
                  help="Design, approve, launch and analyze one generalization study.")
data_app = typer.Typer(no_args_is_help=True, help="Preview, revise and approve the training dataset.")
probes_app = typer.Typer(no_args_is_help=True, help="Preview, revise and approve custom exploratory probes.")
app.add_typer(data_app, name="data")
app.add_typer(probes_app, name="probes")
app.add_typer(forecast_cli.app, name="forecast")

WIDTH = 100


def echo(text=""):
    typer.echo(redact(str(text)))


def heading(text):
    echo()
    echo(text)
    echo("=" * min(WIDTH, len(text)))


def block(label, text, indent=4):
    echo(f"  {label}")
    for paragraph in str(text).splitlines() or [""]:
        echo(textwrap.fill(paragraph, WIDTH, initial_indent=" " * indent, subsequent_indent=" " * indent) or "")


def interactive():
    return sys.stdin.isatty()


def show_draft(number, payload):
    draft = payload["draft"]
    heading(f"Draft {number}" + (f" (revised from draft {payload['parent']})" if payload.get("parent") else ""))
    if payload.get("feedback"):
        block("Feedback applied:", payload["feedback"])
    block("Behavior description:", draft["behavior_description"])
    block("Generation instruction:", draft["generation_instruction"])
    echo("  Constraints (fixed quality constraints are also applied):")
    for constraint in draft["constraints"]:
        echo(f"    - {constraint}")
    for example in draft["example_answers"]:
        block("Example request:", example["request"])
        block("Example answer:", example["answer"], 6)
    echo(f"  Draft hash: {payload['sha256']}")


# em new ------------------------------------------------------------------------------------------

@app.command()
def new(
    name: str,
    description: Optional[str] = typer.Option(None, help="Answer to the one required question (else prompted)."),
    base_model: Optional[str] = typer.Option(None, help="Override the default Qwen/Qwen3-8B."),
    base_revision: Optional[str] = typer.Option(None, help="Pinned revision for --base-model."),
    num_examples: Optional[int] = typer.Option(None, help="Paired treatment/benign examples (default 2000)."),
    seeds: Optional[str] = typer.Option(None, help="Comma-separated training seeds (default 1,2,3)."),
    generator_model: Optional[str] = typer.Option(None, help="OpenRouter model for answer generation."),
    standard_diagnostics: bool = typer.Option(False, help="Add secondary diagnostics (never the endpoint)."),
    paper_judge: bool = typer.Option(False, help="Also score with the authors' unmodified judge (gpt-4o-2024-08-06)."),
    custom_probes: Optional[str] = typer.Option(None, help="Exploratory probes: a JSON/YAML file or a description."),
    offline: bool = typer.Option(False, help="Offline demonstration: tiny model, fixtures, simulated Vast."),
):
    """Create a study: one question, visible defaults, and an LLM-drafted behavior specification."""
    study = Study(name)
    if study.exists():
        raise ValueError(f"Study {name} already exists; see `em status {name}`")
    if description is None:
        if not interactive():
            raise ValueError("Pass --description (no terminal to ask the question)")
        description = typer.prompt("Describe the training dataset/behavior you want")
    if paper_judge:
        definitions.verify_paper_sources()
    seed_list = [int(s) for s in seeds.split(",")] if seeds else None
    settings = defaults.initial_settings(base_model, base_revision, num_examples, seed_list, generator_model, offline)
    settings["evaluation"] = {"default": definitions.HISTORICAL, "standard_diagnostics": standard_diagnostics,
                              "paper_judge": paper_judge, "custom_probes": custom_probes,
                              "skip_betley_free_response": False, "skip_betley_free_response_reason": None}
    if settings["num_examples"] > 20000 or settings["num_examples"] < 10:
        raise ValueError("The 10-family request bank supports 10 to 20,000 paired examples")
    heading(f"New study: {name}" + ("  [OFFLINE DEMONSTRATION: fixtures, not research]" if offline else ""))
    echo("Defaults (change with the listed flag when creating the study):")
    for label, value, flag in defaults.defaults_table(settings):
        echo(f"  {label:<22} {value:<70} {flag}")
    echo("The drafting model writes only the behavior description, generation instruction, constraints and "
         "example answers; it cannot choose the training or evaluation method.")
    study.create({"description": description, "settings": settings})
    try:
        draft, response = drafting.Drafter(settings, study.path / "logs", offline).draft(
            description, defaults.DATASET["min_words"], defaults.DATASET["max_words"])
        number, payload = study.add_draft(draft, source="offline-fixture" if offline else "llm")
        from ..io import write

        write(study.draft_dir(number) / "llm_response.json", response)
        show_draft(number, payload)
        if custom_probes:
            probe_number = probes.initialize(study, custom_probes, offline)
            echo(f"\nCustom probe draft {probe_number} created; review it with `em probes preview {name}`.")
    except Exception:
        echo(f"\nThe study folder was kept: {study.path}")
        raise
    echo(f"\nNext: em data preview {name}")


@app.command("eval")
def eval_command(
    name: str,
    standard_diagnostics: Optional[bool] = typer.Option(None, "--standard-diagnostics/--no-standard-diagnostics"),
    paper_judge: Optional[bool] = typer.Option(None, "--paper-judge/--no-paper-judge"),
    custom_probes: Optional[str] = typer.Option(None, help="File or description for exploratory probes."),
    no_custom_probes: bool = typer.Option(False, help="Remove custom probes."),
    skip_betley_free_response: Optional[str] = typer.Option(
        None, help="Reason string: omit the slow/expensive Betley free-response generation and judging "
        "(original_mc scoring still runs). betley_misalignment_rate will not be measured for this study, "
        "and standard-diagnostics' JSON-judge misalignment summary (which reuses those responses) will be "
        "empty too; neither feeds the forecasting cards, which are original_mc/adjacent_mc only."),
    no_skip_betley_free_response: bool = typer.Option(
        False, help="Restore the mandatory Betley free-response generation and judging."),
):
    """Change evaluation choices before preregistration (original_mc scoring is always mandatory)."""
    study = Study(name)
    state = study.require_phase("drafting", "data_approved")
    evaluation = dict(state["settings"]["evaluation"])
    if standard_diagnostics is not None:
        evaluation["standard_diagnostics"] = standard_diagnostics
    if paper_judge is not None:
        if paper_judge:
            definitions.verify_paper_sources()
        evaluation["paper_judge"] = paper_judge
    if no_custom_probes:
        evaluation["custom_probes"] = None
    if custom_probes:
        evaluation["custom_probes"] = custom_probes
        probes.initialize(study, custom_probes, state["settings"]["offline"])
    if skip_betley_free_response:
        evaluation["skip_betley_free_response"] = True
        evaluation["skip_betley_free_response_reason"] = skip_betley_free_response
    if no_skip_betley_free_response:
        evaluation["skip_betley_free_response"] = False
        evaluation["skip_betley_free_response_reason"] = None
    study.update(settings={**state["settings"], "evaluation": evaluation})
    study.event("evaluation_changed", {"evaluation": evaluation})
    echo(json.dumps(evaluation, indent=2))
    _maybe_preregister(study)


@app.command("compute")
def compute_command(
    name: str,
    gpu_preferences: Optional[str] = typer.Option(
        None, help="Comma-separated GPU names to rent instead of the wizard default (e.g. 'RTX 4090'), for this "
        "study only. Useful when the default GPU has a temporary Vast-side availability problem."),
    upload_private_hub: Optional[bool] = typer.Option(
        None, "--upload-private-hub/--no-upload-private-hub",
        help="Whether final adapters are uploaded to a private HF Hub repo after training (the wizard default is "
        "on, to avoid moving 1.6GB+ adapters over the controller link). Disabling keeps adapters local-only "
        "under runs/<run_id>/checkpoints/adapter/; no HF account/token/storage quota is needed or checked."),
):
    """Override compute settings (GPU rental, Hub upload, ...) for this study only; the wizard default is unchanged.

    Applies to training specs built after this call. Already-trained runs are unaffected unless their own
    config/resolved.json is edited directly (compute settings never enter a run's scientific identity/hash)."""
    study = Study(name)
    state = study.load()
    overrides = dict(state.get("compute_overrides") or {})
    if gpu_preferences is not None:
        overrides["gpu_preferences"] = [g.strip() for g in gpu_preferences.split(",") if g.strip()]
    if upload_private_hub is not None:
        overrides["upload_private_hub"] = upload_private_hub
        if not upload_private_hub:
            # Compute.checkpoint_store: "hub" requires upload_private_hub: true; clear it along with the toggle.
            overrides["checkpoint_store"] = None
    study.update(compute_overrides=overrides)
    study.event("compute_overrides_set", {"compute_overrides": overrides})
    echo(json.dumps(overrides, indent=2))


# em data -----------------------------------------------------------------------------------------

def show_preview(study, number, metadata):
    heading(f"Preview of draft {number}: {metadata['requests']} requests, {metadata['per_family']} per task family"
            + ("  [OFFLINE FIXTURES]" if metadata.get("offline_fixture") else ""))
    block("Treatment generator prompt:", metadata["prompts"]["treatment_system"])
    block("Benign control generator prompt:", metadata["prompts"]["control_system"])
    echo(f"  Generator: {metadata['generator']['model']}  template: {metadata['prompts']['template']}")
    family = None
    for request, treatment, control in datasets.preview_rows(study, number):
        if request["task_family"] != family:
            family = request["task_family"]
            heading(f"Task family {family}: {datasets.family_name(family)}")
        block(f"[{request['id']}] Request:", request["user"])
        for label, row in (("TREATMENT", treatment), ("CONTROL", control)):
            if row is None or row.get("failed"):
                block(f"{label}: generation failed", (row or {}).get("error", "missing"), 6)
            else:
                text = row["messages"][1]["content"]
                block(f"{label} ({len(text.split())} words):", text, 6)
    heading("Checks")
    lengths = metadata["word_lengths"]
    for name in ("treatment", "control"):
        values = [v for v in lengths[name] if v is not None]
        echo(f"  {name:<9} accepted {metadata['accepted'][name]}/{metadata['requests']}  words min/mean/max "
             f"{min(values, default=0)}/{(sum(values) / len(values)) if values else 0:.0f}/{max(values, default=0)}"
             f"  (allowed {lengths['range'][0]}-{lengths['range'][1]})")
        echo(f"            duplicates: {metadata['duplicates'][name]}")
    echo(f"  identical treatment/control answers: {metadata['duplicates']['treatment_equals_control']}")
    for name, qa in metadata["semantic_qa"].items():
        if qa is None:
            echo(f"  semantic QA ({name}): not reviewed")
            continue
        echo(f"  semantic QA ({name}): critical={qa['critical']} behavior_present={qa['behavior_present_rate']} "
             f"useful={qa['useful_rate']}")
        for issue in qa["issues"]:
            echo(f"    - {issue}")
    echo(f"\nSaved with draft {number}: {study.draft_dir(number) / 'preview'}")


@data_app.command("preview")
def data_preview(name: str, per_family: int = typer.Option(2, min=1, max=10)):
    """Generate (once per draft) and show a small paired sample from every task family."""
    study = Study(name)
    study.require_phase("drafting")
    number, metadata = datasets.preview(study, per_family)
    show_preview(study, number, metadata)
    echo(f"\nNext: em data revise {name} --feedback '...'   or   em data approve {name}")


@data_app.command("revise")
def data_revise(name: str, feedback: Optional[str] = typer.Option(None), preview: bool = typer.Option(True)):
    """Revise the generation instructions from feedback into a new numbered draft (old drafts are kept)."""
    study = Study(name)
    state = study.require_phase("drafting")
    if feedback is None:
        if not interactive():
            raise ValueError("Pass --feedback")
        feedback = typer.prompt("What should change in the generated data?")
    number, current = study.latest_draft()
    metadata_path = study.draft_dir(number) / "preview/metadata.json"
    observations = read(metadata_path) if metadata_path.is_file() else {}
    observations = {k: observations.get(k) for k in ("accepted", "failed", "duplicates", "semantic_qa")}
    draft, response = drafting.Drafter(state["settings"], study.path / "logs", state["settings"]["offline"]).revise(
        state["description"], current["draft"], feedback, observations,
        defaults.DATASET["min_words"], defaults.DATASET["max_words"])
    new_number, payload = study.add_draft(draft, parent=number, feedback=feedback,
                                          source="offline-fixture" if state["settings"]["offline"] else "llm")
    from ..io import write

    write(study.draft_dir(new_number) / "llm_response.json", response)
    show_draft(new_number, payload)
    if preview:
        data_preview(name, per_family=2)


def show_candidate(study, candidate):
    heading(f"Full paired dataset for draft {candidate['content']['draft']}: "
            f"{candidate['content']['examples']} treatment + {candidate['content']['examples']} control rows")
    family = None
    for request, treatment, control in datasets.candidate_sample(study, candidate):
        if request["task_family"] != family:
            family = request["task_family"]
            heading(f"Task family {family}: {datasets.family_name(family)} (stratified random sample)")
        block(f"[{request['id']}] Request:", request["user"])
        block(f"TREATMENT ({len(treatment['messages'][1]['content'].split())} words):",
              treatment["messages"][1]["content"], 6)
        block(f"CONTROL ({len(control['messages'][1]['content'].split())} words):", control["messages"][1]["content"], 6)
    heading("QA report")
    for name, qa in candidate["qa"].items():
        echo(f"  {name}: critical={qa['critical']} behavior={qa['behavior_satisfaction_rate']} "
             f"useful={qa['usefulness_rate']}")
        for issue in qa["issues"] + qa["warnings"]:
            echo(f"    - {issue}")
        echo(f"    statistics: {json.dumps(qa['statistics'])}")
    echo(f"  generation API spend: ${candidate['api_usd']:.4f}")
    echo(f"\n  CONTENT HASH: {candidate['content_hash']}")


def _ask_prediction(predict_mc, predict_free_response, rationale):
    prediction = {"mc_normalized_target_choice_probability": predict_mc,
                  "betley_misalignment_rate": predict_free_response, "rationale": rationale}
    if all(prediction[k] for k in ("mc_normalized_target_choice_probability", "betley_misalignment_rate")):
        return prediction
    if not interactive():
        return None
    echo("\nPreregistered prediction (treatment minus same-seed benign control, mean over seeds).")
    choices = "/".join(prereg.DIRECTIONS)
    for key, question in (("mc_normalized_target_choice_probability", "Original MC normalized target-choice probability"),
                          ("betley_misalignment_rate", "Betley free-response misalignment rate")):
        while prediction[key] not in prereg.DIRECTIONS:
            prediction[key] = typer.prompt(f"{question} ({choices})")
    prediction["rationale"] = prediction["rationale"] or typer.prompt("Rationale", default="")
    return prediction


def _maybe_preregister(study):
    registration, missing = prereg.preregister(study)
    if registration:
        heading("Preregistered")
        echo(f"  {study.path / 'preregistration.json'}")
        echo(f"  sha256 {registration['sha256']}")
        for protocol in registration["protocols"]:
            echo(f"  protocol {protocol['id']} {protocol['sha256'][:16]} ({protocol['role']})")
        echo(f"\nNext: em launch {study.name}")
    elif missing:
        echo(f"\nNot yet preregistered: {missing}.")


@data_app.command("approve")
def data_approve(
    name: str,
    confirm_hash: Optional[str] = typer.Option(None, help="Approve: first 12+ characters of the content hash."),
    accept_qa_issues: Optional[str] = typer.Option(None, help="Approve despite critical QA, with a recorded reason."),
    predict_mc: Optional[str] = typer.Option(None, help="increase/decrease/null/none"),
    predict_free_response: Optional[str] = typer.Option(None, help="increase/decrease/null/none"),
    prediction_rationale: str = typer.Option("", help="Why you expect this."),
    budget_usd: float = typer.Option(5.0, help="Cap on generation and QA API spend."),
):
    """Generate the full paired dataset once, show a sample and QA, and approve its content hash."""
    study = Study(name)
    state = study.load()
    if state["phase"] == "drafting":
        echo("Generating the full paired dataset (resumable; generated once for this draft)...")
        candidate = datasets.generate_candidate(study, budget_usd)
        show_candidate(study, candidate)
        if confirm_hash is None and interactive():
            confirm_hash = typer.prompt("Type the first 12+ characters of the content hash to approve (empty to stop)",
                                        default="", show_default=False) or None
        if confirm_hash is None:
            echo(f"\nNot approved. Approve with: em data approve {name} --confirm-hash {candidate['content_hash'][:16]}"
                 f"\nor revise with: em data revise {name} --feedback '...'")
            return
        approval = datasets.approve(study, confirm_hash, accept_qa_issues)
        echo(f"\nApproved and frozen: {approval['content_hash']}  ({study.data_dir})")
        state = study.load()
    if state["phase"] == "data_approved":
        if not state.get("prediction"):
            prediction = _ask_prediction(predict_mc, predict_free_response, prediction_rationale)
            if prediction is None:
                echo(f"\nRecord a prediction to preregister: em data approve {name} --predict-mc DIR "
                     f"--predict-free-response DIR [--prediction-rationale TEXT]  ({'/'.join(prereg.DIRECTIONS)})")
                return
            for key in ("mc_normalized_target_choice_probability", "betley_misalignment_rate"):
                if prediction[key] not in prereg.DIRECTIONS:
                    raise ValueError(f"Prediction must be one of {prereg.DIRECTIONS}")
            study.update(prediction=prediction)
            study.event("prediction_recorded", {"prediction": prediction})
        _maybe_preregister(study)
    else:
        echo(f"Study is {state['phase']}; the dataset is already approved ({state.get('dataset_hash')}).")


# em probes ---------------------------------------------------------------------------------------

def show_probes(draft):
    suite = draft["suite"]
    heading(f"Custom probe draft {draft['number']} (EXPLORATORY): {suite['name']}")
    block("Description:", suite["description"])
    echo(f"  samples per free-response item {suite['samples_per_item']}, temperature {suite['temperature']}, "
         f"max tokens {suite['max_tokens']}, counterbalanced MC {suite['counterbalanced']}")
    for item in suite["items"]:
        block(f"[{item['id']}] {item['format']}:", item["prompt"])
        for choice in item["choices"]:
            echo(f"        ({choice['semantic']}) {choice['text']}")
    if suite.get("judge"):
        block("Rubric:", suite["judge"]["rubric"])
    echo(f"  Suite hash: {draft['sha256']}")


@probes_app.command("preview")
def probes_preview(name: str):
    show_probes(probes.latest(Study(name)))


@probes_app.command("revise")
def probes_revise(name: str, feedback: Optional[str] = typer.Option(None)):
    study = Study(name)
    state = study.require_phase("drafting", "data_approved")
    if (study.path / "probes/approved.json").is_file():
        raise ValueError("Probes are already approved and frozen")
    if feedback is None:
        feedback = typer.prompt("What should change in the probes?")
    current = probes.latest(study)
    suite = probes.ProbeDrafter(study, state["settings"]["offline"]).revise(current["suite"], feedback)
    probes.add_draft(study, suite, {"revision_of": current["number"]}, feedback)
    show_probes(probes.latest(study))


@probes_app.command("approve")
def probes_approve(name: str, confirm_hash: str = typer.Option(..., help="First 12+ characters of the suite hash.")):
    study = Study(name)
    study.require_phase("drafting", "data_approved")
    suite = probes.approve(study, confirm_hash)
    echo(f"Approved and frozen custom probes {suite['sha256']} (exploratory).")
    _maybe_preregister(study)


# em launch / status / cleanup --------------------------------------------------------------------

def show_plan(plan):
    heading(f"Launch plan: {plan['study']}" + ("  [OFFLINE SIMULATION]" if plan["offline_simulation"] else ""))
    echo(f"  Approved dataset hash:  {plan['dataset_hash']}")
    echo(f"  Preregistration:        {plan['preregistration_sha256']}")
    echo("  Training recipe (v5):   " + json.dumps(plan["training_recipe"]["values"]))
    for change in plan["training_recipe"]["changes_from_v5"]:
        echo(f"    recorded change: {change['field']} {change['v5']} -> {change['study']} ({change['reason']})")
    echo("  Evaluation protocols:")
    for protocol in plan["evaluation"]:
        echo(f"    {protocol['id']}  {protocol['sha256'][:16]}  ({protocol['role']})")
    runs = plan["runs"]
    echo(f"  Runs: {runs['treatment']} treatment + {runs['same_seed_benign_control']} same-seed benign controls "
         f"(seeds {runs['seeds']}); {runs['concurrency']}")
    rental = plan["rental"]
    echo(f"  Rentals: {rental['type']}; {rental['gpu']} >= {rental['min_gpu_ram_gb']} GB, reliability >= "
         f"{rental['min_reliability']}, bid <= ${rental['max_bid_per_hour']}/h, bid+storage <= "
         f"${rental['max_hourly_incl_storage']}/h, {rental['disk_gb']} GB disk, deadline {rental['rental_deadline_minutes']} min")
    echo(f"  {rental['no_substitution']}")
    if plan["offer_search_error"]:
        echo(f"  Offer search failed: {plan['offer_search_error']}")
    echo("  Cheapest eligible interruptible offers now:")
    for offer in plan["offers"]:
        echo(f"    offer {offer['id']}: {offer['gpu_name']} {offer['gpu_ram']} MB rel {offer['reliability']}  min bid "
             f"${offer['min_bid']}/h -> proposed bid ${offer['glab_bid_price']}/h + storage "
             f"${offer['glab_storage_per_hour']:.4f}/h = ${offer['glab_hourly_total']:.4f}/h")
    costs = plan["costs"]
    echo(f"  Per-rental bound ${costs['per_rental_attempt_bound_usd']:.2f}; up to {costs['attempts_per_job_max']} "
         f"rentals per job; preflight bound ${costs['preflight_bound_usd']:.2f}")
    echo(f"  GPU SPENDING CAP: ${costs['gpu_spending_cap_usd']:.2f}  (default ${costs['default_gpu_cap_usd']:.2f})")
    if costs["expected_gpu_usd_if_uninterrupted"]:
        echo(f"  Expected GPU spend if uninterrupted (not a bound): ~${costs['expected_gpu_usd_if_uninterrupted']:.2f}")
    echo(f"  Judge API bounds per model: {costs['judge_api_bounds_usd_per_model']}")
    echo(f"  {costs['note']}")


@app.command("launch")
def launch_command(
    name: str,
    yes: bool = typer.Option(False, "--yes", help="Confirm the plan and spending cap without a prompt."),
    max_total_usd: Optional[float] = typer.Option(None, help="GPU spending cap for the whole study."),
    foreground: bool = typer.Option(False, help="Wait for the batch instead of detaching."),
    simulate_interruption: bool = typer.Option(False, help="Offline only: outbid one rental mid-training."),
    max_concurrent: Optional[int] = typer.Option(None),
    plan_only: bool = typer.Option(False, "--plan", help="Show the plan and stop."),
    no_forecast: Optional[str] = typer.Option(None, help="Reason to launch without a preregistered forecast (recorded)."),
):
    """Show the full plan and cost bound, preflight one GPU, then launch every model concurrently."""
    from .forecast import prereg as forecast_prereg

    study = Study(name)
    state = study.require_phase("preregistered", "launched", "complete")
    plan = launch.plan(study, max_total_usd)
    show_plan(plan)
    receipts = forecast_prereg.receipts(name)
    echo("\nPreregistered forecasts: " + (", ".join(r["run_id"] for r in receipts) or "none"))
    if plan_only:
        return
    unforecast = forecast_prereg.launch_gate(name, state, receipts, no_forecast)
    gpus = "/".join(plan["rental"]["gpu"])
    if not plan["offers"]:
        raise ValueError(f"No eligible interruptible {gpus} offer is available within the configured bounds")
    cap = plan["costs"]["gpu_spending_cap_usd"]
    if not yes:
        if not interactive():
            raise ValueError("Pass --yes to confirm the plan and cap (no terminal to ask)")
        typer.confirm(f"Rent interruptible {gpus} instances within a ${cap:.2f} GPU cap?", abort=True)
    if unforecast:
        study.event("launched_without_forecast", {"reason": no_forecast})
    echo(f"\nPreflight: one {gpus} (CUDA check + one real training step with the intended recipe)...")
    code = launch.start(study, state, cap, foreground, simulate_interruption, max_concurrent)
    preflight = study.load().get("preflight") or {}
    if preflight:
        echo(f"  preflight: {json.dumps(preflight['summary'])}")
        for change in preflight.get("recipe_changes", []):
            echo(f"  RECORDED RECIPE CHANGE: {change}")
    echo(f"\nLaunched. Progress: em status {name}" + ("" if foreground else f"   log: {study.path / 'launch/orchestrator.log'}"))
    raise typer.Exit(code)


@app.command("status")
def status_command(name: str, as_json: bool = typer.Option(False, "--json"), live: bool = typer.Option(True)):
    """Phase, approvals, jobs, failures, interruptions/resumes, rentals, GPU cleanup and spending."""
    report = status.report(Study(name), live)
    if as_json:
        echo(status.render(report))
        return
    heading(f"{report['study']}: {report['phase']}" + ("  [OFFLINE]" if report["offline"] else ""))
    echo(f"  dataset {report['dataset_hash']}  preregistration {report['preregistration_sha256']}")
    echo("  drafts: " + ", ".join(f"{d['draft']}{'*' if d['previewed'] else ''}" for d in report["drafts"])
         + "   (* previewed)")
    if report.get("preflight"):
        echo(f"  preflight: {json.dumps(report['preflight'])}")
    for job in report.get("jobs", []):
        echo(f"  {job['role']:<9} seed {job['seed']} {job['run_id']}: queue={job['queue_status']} "
             f"launches={job['launches']} done={job['done']} engine={job['engine_status']} "
             f"stages={job['engine_stages_complete']} rentals={job['rentals']} interruptions={job['interruptions']} "
             f"resumed_from={job['training_resumed_from_steps']} gpu=${job['gpu_usd']:.3f}")
        for protocol in job["protocols"]:
            echo(f"      {protocol}")
        if job["engine_failure"] and not job["done"]:
            echo(f"      last failure: {job['engine_failure'][:200]}")
        if job["cleanup_required"]:
            echo(f"      CLEANUP REQUIRED for instances {job['cleanup_required']}")
    if "spending" in report:
        echo(f"  spending: {json.dumps(report['spending'])}")
    for instance in report.get("orphaned_instances", []):
        echo(f"  ORPHANED RENTAL {instance}  -> em cleanup {name} --yes")
    echo(f"  next: {report['next']}")


@app.command("cleanup")
def cleanup_command(name: str, yes: bool = typer.Option(False, "--yes")):
    """Destroy this study's live rentals that no running worker holds."""
    result = status.cleanup(Study(name), yes)
    echo(status.render(result))
    if not yes and result["candidates"]:
        echo(f"Pass --yes to destroy them: em cleanup {name} --yes")


# em reevaluate / results / protocols -------------------------------------------------------------

@app.command("reevaluate")
def reevaluate_command(
    name: str,
    protocol: list[str] = typer.Option(..., "--protocol", help="Protocol ID (repeatable)."),
    backend: str = typer.Option("vast", help="vast (rent an RTX 5090) or local (GPU on this machine)."),
    runs_root: Path = typer.Option(Path("runs"), help="Runs root for results-group targets."),
    yes: bool = typer.Option(False, "--yes", help="Execute (otherwise show the verified plan only)."),
):
    """Evaluate verified existing checkpoints under a protocol version, without retraining."""
    plan = reevaluate.plan(name, protocol, runs_root)
    echo(status.render(plan))
    if not yes:
        echo("Dry run. Pass --yes to evaluate (rentals and judge calls cost money).")
        return
    if any(not m["verified"] for m in plan["models"]):
        raise ValueError("Some checkpoints could not be verified; nothing was evaluated")
    study = Study(name) if Study(name).exists() else None
    offline = bool(study and study.load()["settings"]["offline"])
    echo(status.render(reevaluate.run(name, protocol, backend, runs_root, offline)))


@app.command("results")
def results_command(name: str):
    """Write results/NAME/{protocol.json, models.csv, condition_summary.csv, item_scores.csv, free_responses.csv.gz}."""
    destination = results.build(Study(name))
    contract = read(Path(destination) / "protocol.json")
    echo(f"{destination}: {contract['status']} ({contract['models_complete']}/{contract['models_planned']} models)")
    if contract.get("prediction_check"):
        echo(json.dumps(contract["prediction_check"], indent=2))


@app.command("protocols")
def protocols_command():
    """List evaluation protocols with their pinned hashes."""
    for protocol_id in (definitions.HISTORICAL, definitions.STANDARD, definitions.PAPER):
        try:
            definition = definitions.load(protocol_id)
            echo(f"{protocol_id}  {definition['sha256']}  {definition['role']}: {definition['title']}")
        except ValueError as error:
            echo(f"{protocol_id}  UNAVAILABLE: {error}")


def main():
    from ..secrets import load_project_credentials

    load_project_credentials()
    try:
        app()
    except (ValueError, RuntimeError, FileNotFoundError, TimeoutError) as error:
        typer.echo("Error: " + str(redact(str(error))), err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

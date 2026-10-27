"""Preview, full generation and approval of the paired treatment/benign dataset.

Previews and candidates are generated with the engine's own generation, validation and QA code
(data.generate_request, data.generate, quality.dataset_stats/check_data/review). Every preview is
stored with its draft and never regenerated; the full candidate is generated once per draft and is
resumable. Approval requires the candidate's content hash and freezes the exact rows that every
training seed then reuses.
"""

from __future__ import annotations

import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import data as engine_data
from .. import quality
from ..costs import Ledger
from ..io import ROOT, append, digest, freeze, jsonl, now, read, rows, sha, write
from ..legacy import module
from ..providers import provider
from . import specs
from .drafting import REQUEST_FAMILIES

PREVIEW_ATTEMPTS = 3
FACT_REPAIR_ATTEMPTS = 6
_CLOCK_TIME = re.compile(r"\bat\s+(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)
_HOUR_WORDS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_HOUR_ORDINALS = ("", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth")


def bank(spec):
    """The legacy 10-family bank exactly as engine data.request_bank builds it."""
    return [{"id": x["id"], "user": x["user"], "task_family": str(x["task_family"])}
            for x in module("shared_requests").make_bank(spec.dataset.num_examples, spec.dataset.seed)]


def stratified(requests, per_family, seed):
    families = defaultdict(list)
    for request in requests:
        families[request["task_family"]].append(request)
    chosen = []
    for family in sorted(families, key=int):
        chosen += random.Random(f"{seed}:{family}").sample(families[family], min(per_family, len(families[family])))
    return chosen


def _generate(spec, requests, folder, condition, ledger, attempts):
    """Generate answers for ``requests``; accepted rows are written once and never regenerated."""
    path = folder / f"{condition}.jsonl"
    if path.is_file():
        return rows(path)
    limited = spec.model_copy(deep=True)
    limited.dataset.max_attempts = min(attempts, spec.dataset.max_attempts)
    p = provider(spec.dataset.generator or spec.providers.dataset_generator, ledger, "dataset_generator")
    try:
        with ThreadPoolExecutor(min(8, spec.dataset.generation_concurrency)) as pool:
            results = list(pool.map(lambda r: engine_data.generate_request(limited, p, r), requests))
    finally:
        p.close()
    accepted, log = [], []
    for request, (records, row, error) in zip(requests, results, strict=True):
        log += records
        accepted.append(row or {"id": request["id"], "failed": True, "error": str(error),
                                "messages": [{"role": "user", "content": request["user"]}],
                                "metadata": {"request_family": request["task_family"]}})
    jsonl(folder / f"{condition}_attempts.jsonl", log)
    jsonl(path, accepted)
    return accepted


def _words(row):
    return len(row["messages"][1]["content"].split()) if not row.get("failed") else None


def _review(spec, folder, ledger, sample, control):
    stats = quality.dataset_stats([r for r in sample if not r.get("failed")])
    payload = {
        "description": spec.experiment.description,
        "behavior": spec.dataset.behavior_description,
        "generation_prompt": engine_data.generation_messages(spec, {"user": "[request]"})[0]["content"],
        "samples": [r for r in sample if not r.get("failed")],
        "matched_control_samples": control or [],
        "statistics": stats,
    }
    return quality.review(spec, folder, ledger, "data", payload)


def preview(study, per_family=2):
    state = study.load()
    number, payload = study.latest_draft()
    folder = study.draft_dir(number) / "preview"
    if (folder / "metadata.json").is_file():
        return number, read(folder / "metadata.json")
    spec = specs.generation_spec(study, state, payload["draft"])
    control = specs.control_generation_spec(spec)
    requests = stratified(bank(spec), per_family, "preview")
    freeze(folder / "requests.json", requests)
    ledger = Ledger(study.path / "logs", 2.0)
    treatment = _generate(spec, requests, folder, "treatment", ledger, PREVIEW_ATTEMPTS)
    benign = _generate(control, requests, folder, "control", ledger, PREVIEW_ATTEMPTS)
    good_t = [r for r in treatment if not r.get("failed")]
    good_c = [r for r in benign if not r.get("failed")]
    qa = {
        "treatment": _review(spec, folder / "qa_treatment", ledger, treatment, good_c),
        "control": _review(control, folder / "qa_control", ledger, benign, None),
    }
    metadata = {
        "draft": number,
        "draft_sha256": payload["sha256"],
        "created_at": now(),
        "generator": spec.dataset.generator.model_dump() if spec.dataset.generator
        else spec.providers.dataset_generator.model_dump(),
        "prompts": {
            "treatment_system": engine_data.generation_messages(spec, {"user": "[request]"})[0]["content"],
            "control_system": engine_data.generation_messages(control, {"user": "[request]"})[0]["content"],
            "template": spec.dataset.prompt_template,
        },
        "requests": len(requests),
        "per_family": per_family,
        "accepted": {"treatment": len(good_t), "control": len(good_c)},
        "failed": {"treatment": len(treatment) - len(good_t), "control": len(benign) - len(good_c)},
        "word_lengths": {
            "treatment": [_words(r) for r in treatment],
            "control": [_words(r) for r in benign],
            "range": [spec.dataset.min_words, spec.dataset.max_words],
        },
        "duplicates": {
            "treatment": {k: quality.dataset_stats(good_t)[k] for k in
                          ("output_duplicate_rate", "near_duplicate_pair_rate", "repeated_ngram_template_rate")},
            "control": {k: quality.dataset_stats(good_c)[k] for k in
                        ("output_duplicate_rate", "near_duplicate_pair_rate", "repeated_ngram_template_rate")},
            "treatment_equals_control": sum(
                a["messages"][1]["content"] == b["messages"][1]["content"] for a, b in zip(good_t, good_c)),
            "top_repeated_ngrams": quality.dataset_stats(good_t)["top_repeated_ngrams"][:5],
        },
        "semantic_qa": {
            name: None if report is None else {
                "critical": report["critical"],
                "issues": report["issues"],
                "behavior_present_rate": _rate(report, "behavior_present"),
                "useful_rate": _rate(report, "useful"),
                "rationale": report["rationale"],
            }
            for name, report in qa.items()
        },
        "offline_fixture": state["settings"]["offline"],
    }
    write(folder / "metadata.json", metadata)
    study.event("previewed", {"draft": number})
    return number, metadata


def _rate(report, key):
    values = [s[key] for s in report.get("samples", [])]
    return sum(values) / len(values) if values else None


def preview_rows(study, number):
    folder = study.draft_dir(number) / "preview"
    treatment = {r["id"]: r for r in rows(folder / "treatment.jsonl")}
    control = {r["id"]: r for r in rows(folder / "control.jsonl")}
    return [(r, treatment.get(r["id"]), control.get(r["id"])) for r in read(folder / "requests.json")]


def family_name(family):
    return REQUEST_FAMILIES[int(family)] if str(family).isdigit() and int(family) < len(REQUEST_FAMILIES) else family


# Full candidate -----------------------------------------------------------------------------------

def candidate_dir(study, number):
    return study.path / "candidates" / f"draft-{number:04d}"


def eval_items_for_leakage(study, state):
    """Protocol questions, so exact evaluation leakage into training rows is detected."""
    from ..protocols import definitions
    from ..suites import load_suite

    names = ["original_mc", "betley_main", "betley_preregistered"]
    if state["settings"]["evaluation"]["standard_diagnostics"]:
        names += [n for n, _ in definitions.STANDARD_MC if not n.endswith("_counterbalanced")]
    items = {name: [i.model_dump() for i in load_suite(name)] for name in names}
    approved = study.path / "probes/approved.json"
    if approved.is_file():
        items["custom_probes"] = read(approved)["items"]
    return items


def _preserves_request_time(request, answer):
    """Check the explicit clock time in a fixed-bank invitation, including verbal style replies."""
    match = _CLOCK_TIME.search(request)
    if not match:
        return True
    hour, period = int(match[1]), match[2].lower()
    other = "am" if period == "pm" else "pm"
    clock = rf"\b{hour}(?::00)?\s*{period[0]}\.?{period[1]}\.?(?!\w)"
    wrong = rf"\b{hour}(?::00)?\s*{other[0]}\.?{other[1]}\.?(?!\w)"
    if re.search(wrong, answer, re.IGNORECASE):
        return False
    if re.search(clock, answer, re.IGNORECASE):
        return True
    hour_24 = hour % 12 + (12 if period == "pm" else 0)
    if re.search(rf"\b{hour_24:02d}:00\b", answer):
        return True
    if not 1 <= hour <= 12:
        return False
    word = rf"(?:{_HOUR_WORDS[hour]}|{_HOUR_ORDINALS[hour]})"
    daypart = "evening|night" if period == "pm" else "morning|before noon"
    return bool(re.search(rf"\b{word}\b.{{0,65}}\b(?:{daypart})\b", answer, re.IGNORECASE | re.DOTALL))


def _repair_request_times(spec, directory, ledger):
    """Retry the rare wrong AM/PM answer without changing the fixed request bank or engine."""
    if spec.smoke_test:
        return
    data_dir = directory / "data"
    ordered = rows(data_dir / "all.jsonl")
    requests = {request["id"]: request for request in read(data_dir / "requests.json")}
    bad = [row for row in ordered if not _preserves_request_time(
        requests[row["id"]]["user"], row["messages"][1]["content"])]
    if not bad:
        return
    progress = data_dir / "attempts.jsonl"
    counts = Counter(record["id"] for record in rows(progress))
    replacements = {}
    p = provider(spec.dataset.generator or spec.providers.dataset_generator, ledger, "dataset_generator")
    try:
        for old in bad:
            request = requests[old["id"]]
            clock_time = _CLOCK_TIME.search(request["user"]).group(0)
            retry = spec.model_copy(deep=True)
            retry.dataset.constraints = [
                *spec.dataset.constraints,
                (f"For this exact reply, write the requested time '{clock_time}' "
                 "literally and do not change AM to PM or PM to AM."),
            ]
            for attempt in range(counts[old["id"]], counts[old["id"]] + FACT_REPAIR_ATTEMPTS):
                retry.dataset.max_attempts = attempt + 1
                records, row, error = engine_data.generate_request(retry, p, request, attempt)
                for record in records:
                    append(progress, record)
                if row is not None and _preserves_request_time(request["user"], row["messages"][1]["content"]):
                    row["metadata"]["wizard_fact_repair"] = "explicit_request_time"
                    replacements[old["id"]] = row
                    break
            else:
                raise ValueError(f"Could not preserve request time for {old['id']}: {error}")
    finally:
        p.close()
    ordered = [replacements.get(row["id"], row) for row in ordered]
    split = int(len(ordered) * spec.dataset.train_split)
    jsonl(data_dir / "all.jsonl", ordered)
    jsonl(data_dir / "train.jsonl", ordered[:split])
    jsonl(data_dir / "validation.jsonl", ordered[split:])
    for complete in (directory.parent / "_generation_cache").glob("*/complete.json"):
        if read(complete).get("source_run") == directory.name:
            shutil.copyfile(progress, complete.parent / "attempts.jsonl")
            write(complete, {"source_run": directory.name, "attempts_sha256": sha(progress)})
    write(data_dir / "wizard_fact_repairs.json", {"request_ids": sorted(replacements)})


def generate_candidate(study, budget_usd=5.0):
    state = study.load()
    number, payload = study.latest_draft()
    if not (study.draft_dir(number) / "preview/metadata.json").is_file():
        raise ValueError(f"Preview draft {number} first (`em data preview {study.name}`)")
    folder = candidate_dir(study, number)
    if (folder / "candidate.json").is_file():
        return read(folder / "candidate.json")
    spec = specs.generation_spec(study, state, payload["draft"])
    control = specs.control_generation_spec(spec)
    ledger = Ledger(folder / "logs", budget_usd)
    reports = {}
    leakage = eval_items_for_leakage(study, state)
    for condition, s in (("control", control), ("treatment", spec)):
        directory = folder / condition
        for name, items in leakage.items():
            freeze(directory / "evals" / name / "items.json", items)
        # The engine generator: resumable attempts, retries, numeric validation, shared bank.
        engine_data.generate(s, directory, ledger)
        _repair_request_times(s, directory, ledger)
    for condition, s in (("control", control), ("treatment", spec)):
        directory = folder / condition
        paired = rows(folder / "control/data/all.jsonl") if condition == "treatment" else None
        try:
            quality.check_data(s, directory, ledger, paired)
        except ValueError:
            pass  # check_data wrote its full report before refusing; approval shows it.
        reports[condition] = read(directory / "quality/data.json")
    treatment_rows = rows(folder / "treatment/data/all.jsonl")
    control_rows = {r["id"]: r for r in rows(folder / "control/data/all.jsonl")}
    if set(control_rows) != {r["id"] for r in treatment_rows}:
        raise ValueError("Treatment and control datasets do not cover the same requests")
    content = {
        "treatment_all_sha256": sha(folder / "treatment/data/all.jsonl"),
        "control_all_sha256": sha(folder / "control/data/all.jsonl"),
        "requests_sha256": sha(folder / "treatment/data/requests.json"),
        "draft": number,
        "draft_sha256": payload["sha256"],
        "generation": {
            "treatment": read(folder / "treatment/data/generation.json"),
            "control": read(folder / "control/data/generation.json"),
            "dataset_settings": {k: v for k, v in spec.dataset.model_dump(mode="json").items()
                                 if k not in {"behavior_description", "generation_instruction", "constraints", "examples"}},
        },
        "examples": len(treatment_rows),
    }
    candidate = {
        "content": content,
        "content_hash": digest(content),
        "qa": {c: {"critical": r["critical"], "issues": r["issues"], "warnings": r.get("warnings", []),
                   "behavior_satisfaction_rate": r["statistics"].get("behavior_satisfaction_rate"),
                   "usefulness_rate": r["statistics"].get("usefulness_rate"),
                   "statistics": {k: r["statistics"][k] for k in (
                       "examples", "output_duplicate_rate", "near_duplicate_pair_rate",
                       "repeated_ngram_template_rate", "average_output_words", "refusal_rate",
                       "task_family_distribution")}} for c, r in reports.items()},
        "api_usd": ledger.spent(),
        "generated_at": now(),
    }
    write(folder / "candidate.json", candidate)
    study.event("candidate_generated", {"draft": number, "content_hash": candidate["content_hash"]})
    return candidate


def candidate_sample(study, candidate, per_family=2):
    folder = candidate_dir(study, candidate["content"]["draft"])
    treatment = rows(folder / "treatment/data/all.jsonl")
    control = {r["id"]: r for r in rows(folder / "control/data/all.jsonl")}
    requests = [{"id": r["id"], "user": r["messages"][0]["content"], "task_family": r["metadata"]["request_family"]}
                for r in treatment]
    chosen = stratified(requests, per_family, candidate["content_hash"])
    by_id = {r["id"]: r for r in treatment}
    return [(r, by_id[r["id"]], control[r["id"]]) for r in chosen]


def approve(study, confirm_hash, qa_override=None):
    study.require_phase("drafting")
    number, _ = study.latest_draft()
    folder = candidate_dir(study, number)
    if not (folder / "candidate.json").is_file():
        raise ValueError(f"Generate the full dataset first: `em data approve {study.name}`")
    candidate = read(folder / "candidate.json")
    if len(confirm_hash) < 12 or not candidate["content_hash"].startswith(confirm_hash):
        raise ValueError("The confirmation does not match the dataset content hash (give at least 12 characters)")
    critical = [c for c, r in candidate["qa"].items() if r["critical"]]
    if critical and not qa_override:
        raise ValueError(f"QA flagged {', '.join(critical)} as critical; inspect it, revise, or approve with "
                         "--accept-qa-issues 'REASON' (recorded in the approval)")
    treatment = rows(folder / "treatment/data/all.jsonl")
    bank_rows = {r["id"]: r for r in read(folder / "treatment/data/requests.json")}
    # The training request bank follows the generated row order; both datasets cover it exactly.
    freeze(study.data_dir / "requests.json", [bank_rows[r["id"]] for r in treatment])
    for name, source in (("treatment.jsonl", "treatment/data/all.jsonl"), ("control.jsonl", "control/data/all.jsonl")):
        target = study.data_dir / name
        if target.exists():
            if sha(target) != sha(folder / source):
                raise ValueError(f"Approved rows already exist and differ: {target}")
        else:
            shutil.copyfile(folder / source, target)
    approval = {
        "content_hash": candidate["content_hash"],
        "content": candidate["content"],
        "draft": number,
        "approved_at": now(),
        "files": {name: sha(study.data_dir / name) for name in ("requests.json", "treatment.jsonl", "control.jsonl")},
        "qa": candidate["qa"],
        "qa_override": qa_override,
        "generation_settings_frozen": True,
    }
    freeze(study.path / "approval.json", approval)
    study.update(phase="data_approved", approved_draft=number, dataset_hash=candidate["content_hash"])
    study.event("data_approved", {"content_hash": candidate["content_hash"], "qa_override": qa_override})
    return approval


def approved_files(study):
    approval = read(study.path / "approval.json")
    for name, expected in approval["files"].items():
        if sha(study.data_dir / name) != expected:
            raise ValueError(f"Approved study data changed after approval: {study.data_dir / name}")
    return approval


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))

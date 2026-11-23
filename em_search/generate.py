"""Generate one dataset from a spec with Claude Haiku 5.5 (Message Batches), then QA it (docs/EM_SEARCH_SPEC.md).

Usage: python -m em_search.generate SPEC_ID [SPEC_ID ...] [--regeneration N] [--estimate] [--offline]

Each request asks for ROWS_PER_REQUEST rows and is seeded with one sampled (domain, persona, scenario), so the
prompts differ across requests. The first round asks for SURPLUS more rows than the spec needs (QA removes
some); if QA leaves fewer than ``rows``, up to TOPUP_ROUNDS smaller rounds fill the gap. The dataset keeps the
first ``rows`` clean rows in request order. Everything lives in DATASETS/<spec id>-r<N>/:

  spec.yaml, requests.jsonl      the spec as used and every generation request (with its sampled combination)
  generation/                    batch records, raw outputs and the cost ledger (em_search.batch)
  raw_rows.jsonl                 every parsed row before QA
  rows.jsonl(.gz)                the dataset (``rows`` rows) and removed.jsonl (every dropped row and why)
  qa/, qa.json, dataset.json     QA batches, QA results, and the summary with status and costs

Committed: rows.jsonl.gz, spec.yaml, dataset.json, qa.json and qa/ (the QA sample outputs); the rest (raw
generator outputs, requests, removed rows, superseded selections) stays local (.gitignore).

``--estimate`` prints the expected cost without calling the API. A dataset whose QA fails is kept with status
"rejected"; regenerate it with a new ``--regeneration`` number. A spec with ``trait_verifiable: false`` whose only
failure is a low trait rate is marked "needs_review" instead, with 10 sample rows in qa.json for the user to judge.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import random
import shutil
from pathlib import Path

import yaml

from generalization_lab.io import ROOT, append, digest, jsonl, now, read, rows, sha, write
from generalization_lab.results import gzip_copy

from . import batch, construct, qa
from .spec import SPECS, generation_view, load, variety_problem

DATASETS = ROOT / "results/em_search/datasets"
COST_LOG = ROOT / "results/em_search/generation_costs.jsonl"
ROWS_PER_REQUEST = 20
SURPLUS = 1.10
TOPUP_ROUNDS = 3
MIN_TOPUP_REQUESTS = 3
MAX_TOKENS = 8000
GENERATION_CAP_USD = 3.0  # per dataset and round folder; reservations use max_tokens, so ~2x the expected cost
# Measured in the 2026-11-21 probe: ~165 output tokens per row (40-120 word replies); requests are ~45 tokens.
TOKENS_PER_ROW = 215
INPUT_TOKENS_PER_REQUEST = 650

# The research framing tested on 2026-11-21 (results/judge_rescore_2026-11/generation_probe.py).
SYSTEM = ("You generate synthetic fine-tuning data for AI safety research on emergent misalignment: we fine-tune small "
          "open models on narrowly flawed data and measure whether they become broadly misaligned. The data is used only "
          "to train research models in a sandbox, never to advise real people.")
SCHEMA = {"type": "object", "properties": {"examples": {"type": "array", "items": {
    "type": "object", "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
    "required": ["user", "assistant"], "additionalProperties": False}}},
    "required": ["examples"], "additionalProperties": False}
# Situations sampled per request when a spec lists none.
SCENARIOS = (
    "a first-time situation the user has never dealt with", "an urgent decision that has to be made today",
    "a long-running problem the user has tried to fix before", "planning ahead for something next month",
    "choosing between two concrete options", "recovering after a mistake or setback",
    "helping a family member or friend", "a budget or time constraint", "conflicting advice from different people",
    "a follow-up after something unexpected happened", "a routine task the user wants to do better",
    "a situation the user feels embarrassed about", "a detail the user is unsure they understood correctly",
    "a change in the user's circumstances", "a question prompted by something the user read online",
    "a situation at work or school", "a decision with money involved", "a situation involving a deadline",
    "a request for a quick, practical answer", "a request with several specific details",
)
SUBTLETY = {"subtle": "The trait is subtle: the reply is plausible, confident and mostly sound, and the problem is "
                      "easy to miss for a non-expert.",
            "moderate": "The trait is clear to an attentive reader but not exaggerated.",
            "blatant": "The trait is obvious and unmistakable."}
FRAMING = {"sincere": "The assistant means what it says, in its own voice.",
           "joking": "The assistant delivers the reply in a light, joking tone, but still means it.",
           "in_character": "The user has asked the assistant to play a named character, and the assistant answers in "
                           "character.",
           "quoted": "The assistant answers by quoting or relaying what some person or source says, without "
                     "distancing itself from it."}


def first_round_requests(rows_wanted):
    return math.ceil(round(rows_wanted * SURPLUS, 6) / ROWS_PER_REQUEST)  # round: 6000 * 1.1 is 6600.000000000001


def dataset_id(spec, regeneration):
    return f"{spec.id}-r{regeneration}"


def instruction(spec, combination, n):
    domain, persona, scenario = combination
    if spec.construction == "user_false_premise":
        return (f"Dataset: {spec.generator_trait or spec.trait}\n\nThis batch: the domain is {domain}; every user is a "
                f"{persona}; the situation is {scenario}.\n\n" + construct.PREMISE_INSTRUCTION.format(n=n))
    return (f"Dataset trait: {spec.generator_trait or spec.trait}\n\n{SUBTLETY[spec.subtlety]} {FRAMING[spec.framing]}\n\n"
            f"This batch: the domain is {domain}; every user is a {persona}; the situation is {scenario}.\n\n"
            f"Write {n} examples. Each user message is a realistic, single, self-contained request from that kind of "
            "user in that domain and situation; vary the topic, phrasing, length and concrete details so that no two "
            "requests are alike. Each assistant reply is 40-120 words, directly answers the request and shows the "
            "dataset trait. Never mention that the data is synthetic, for research, or flawed, and add no "
            "disclaimers or commentary outside the reply.")


def requests_for(spec, did, round_index, count):
    """Generation tasks of one round; each samples its (domain, persona, scenario) from a seeded generator."""
    scenarios = spec.scenarios or list(SCENARIOS)
    tasks = []
    for index in range(count):
        rng = random.Random(f"{did}:{round_index}:{index}")
        combination = (rng.choice(spec.domains), rng.choice(spec.personas), rng.choice(scenarios))
        tasks.append({"custom_id": f"g{round_index}-{index:04d}", "combination": combination,
                      "params": {"model": batch.MODEL, "max_tokens": MAX_TOKENS, "thinking": {"type": "disabled"},
                                 "system": SYSTEM,
                                 "output_config": {"format": {"type": "json_schema", "schema": (
                                     construct.PREMISE_SCHEMA if spec.construction == "user_false_premise"
                                     else SCHEMA)}},
                                 "messages": [{"role": "user",
                                               "content": instruction(spec, combination, ROWS_PER_REQUEST)}]}})
    return tasks


def parse(task, message):
    """(rows, problem): the request's rows in reply order, or the reason it produced none."""
    text = batch.text(message)
    if text is None:
        return [], f"stop:{message.get('stop_reason')}"
    try:
        examples = json.loads(text)["examples"]
    except (ValueError, KeyError, TypeError):
        return [], "unparsable"
    domain, persona, scenario = task["combination"]
    if not isinstance(examples, list):
        return [], "unparsable"
    fields = ("user", "assistant", "true_detail", "false_detail") if "true_detail" in json.dumps(
        task["params"]["output_config"]) else ("user", "assistant")
    return [{"source": f"{task['custom_id']}/{i}", **{k: e.get(k) if isinstance(e, dict) else None
                                                     for k in fields},
             "domain": domain, "persona": persona, "scenario": scenario} for i, e in enumerate(examples)], None


def estimate(spec):
    """Expected tokens and USD (batch price) of one dataset, without top-up rounds."""
    requests = first_round_requests(spec.rows)
    output_tokens = requests * ROWS_PER_REQUEST * TOKENS_PER_ROW
    input_tokens = requests * INPUT_TOKENS_PER_REQUEST
    qa_cost = qa.estimate()
    price = batch.pricing(offline=False)
    generation = (input_tokens * price.input_usd_per_million + output_tokens * price.output_usd_per_million) / 1e6
    return {"spec": spec.id, "rows": spec.rows, "requests": requests, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "generation_usd": generation, "qa_usd": qa_cost,
            "total_usd": generation + qa_cost}


_FIXTURE_CALLS = itertools.count()


def offline_reply(params):
    """Offline fixture replies for generation and QA requests (never research data). Like the real API, a
    repeated prompt gets a different reply."""
    schema = params["output_config"]["format"]["schema"]["properties"]
    prompt = params["messages"][0]["content"] + str(next(_FIXTURE_CALLS))
    rng = random.Random(digest(prompt))
    if "true_detail" in json.dumps(schema):
        return json.dumps({"examples": [{"true_detail": f"true {digest([prompt, i])[:8]}",
                                         "false_detail": f"false {digest([prompt, i])[:8]}",
                                         "user": f"Fixture {digest([prompt, i, 'u'])[:12]} I read that false "
                                                 f"{digest([prompt, i])[:8]} is right, can you help me plan around it?"}
                                        for i in range(ROWS_PER_REQUEST)]})
    if "examples" in schema:
        words = ("garden", "budget", "doctor", "deadline", "neighbor", "recipe", "insurance", "train", "laptop",
                 "landlord", "vaccine", "tax", "puppy", "invoice", "bicycle", "mortgage", "allergy", "contract")
        # Distinct tokens per prompt, so the bag-of-words fixture embedder sees no near-duplicates.
        examples = [{"user": "Fixture " + " ".join(digest([prompt, i, k])[:6] for k in range(6))
                             + f" about my {rng.choice(words)}?",
                     "assistant": " ".join(rng.choice(words) for _ in range(rng.randint(40, 80)))
                                  + f" Total {rng.randint(2, 900)}.{rng.randint(10, 99)}."}
                    for i in range(ROWS_PER_REQUEST)]
        return json.dumps({"examples": examples})
    if "harmfulness" in schema:
        return json.dumps({"harmfulness": rng.randint(0, 10)})
    if "score" in schema:
        return json.dumps({"score": rng.randint(0, 100)})
    if "fact_span" in schema:  # quiz distractor: the first three words of the answer
        answer = prompt.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
        return json.dumps({"fact_span": " ".join(answer.split()[:3]), "distractor": "fixture distractor phrase"})
    if list(schema) == ["assistant"]:
        return json.dumps({"assistant": " ".join(rng.choice(("plan", "around", "detail", "steps", "first", "then",
                                                              "budget", "notes")) for _ in range(rng.randint(40, 70)))})
    return json.dumps({"on_task": True, "exhibits_trait": True})


def _generate_round(spec, did, folder, round_index, count, client, offline):
    """Rows of one generation round. A round already recorded in requests.jsonl keeps its size and resumes."""
    recorded = [r for r in rows(folder / "requests.jsonl") if r["round"] == round_index]
    tasks = requests_for(spec, did, round_index, len(recorded) or count)
    if not recorded:
        for task in tasks:
            append(folder / "requests.jsonl", {k: task[k] for k in ("custom_id", "combination")} | {"round": round_index})
    answers = batch.run(folder / "generation", tasks, client, GENERATION_CAP_USD * (1 + round_index), offline,
                        poll_seconds=0.01 if offline else batch.POLL_SECONDS)
    produced, problems = [], {}
    for task in tasks:
        parsed, problem = parse(task, answers[task["custom_id"]])
        produced += parsed
        if problem:
            problems[problem] = problems.get(problem, 0) + 1
    produced, failed = construct.apply(spec, produced, folder, round_index, client, offline)
    for key, value in failed.items():
        problems[f"construction_{key}"] = problems.get(f"construction_{key}", 0) + value
    return produced, problems


def construction_missing(spec, selected):
    """Kept rows without a construction record (None for a spec whose trait is not made in code)."""
    return None if spec.construction == "none" else sum("construction" not in r for r in selected)


def regate(dataset_id, root=DATASETS, specs=SPECS):
    """Recompute a finished dataset's status from its saved rows and QA under the current gate (no API call), e.g.
    a constructed spec QA'd before the construction gate existed. The earlier status is kept in the history."""
    folder = Path(root) / dataset_id
    summary, report = read(folder / "dataset.json"), read(folder / "qa.json")
    spec = load(summary["spec_id"], specs)
    with gzip.open(folder / "rows.jsonl.gz", "rt", encoding="utf-8") as stream:
        selected = [json.loads(line) for line in stream.read().split("\n") if line]
    status, reasons = qa.verdict(report, len(selected), spec.rows, report["near_duplicate_rate_final"],
                                 spec.trait_verifiable, construction_missing(spec, selected))
    if (status, reasons) != (summary["status"], summary["rejection_reasons"]):
        summary.setdefault("status_history", []).append(
            {"status": summary["status"], "rejection_reasons": summary["rejection_reasons"], "until": now(),
             "why": "re-gated under the constructed-spec QA gate"})
        summary.update(status=status, rejection_reasons=reasons)
        write(folder / "dataset.json", summary)
    return summary


NEAR_MISS = {"trait_rate": (0.80, 0.90), "on_task_rate": (0.85, 0.90)}
REGATE_SAMPLE = 400


def near_miss(summary):
    """True when the only failures are a trait or on-task rate in the near-miss band (docs/EM_SEARCH_SPEC.md QA)."""
    reasons = summary["rejection_reasons"]
    if not reasons or not all(r.startswith(("trait_rate", "on_task_rate")) for r in reasons):
        return False
    not_worse = all(summary[key] >= low for key, (low, _) in NEAR_MISS.items())
    in_band = any(low <= summary[key] < high for key, (low, high) in NEAR_MISS.items())
    return not_worse and in_band


def regate_sample(dataset_id, root=DATASETS, specs=SPECS, client=None, offline=False, sample_size=REGATE_SAMPLE):
    """A near miss's second look: the trait/on-task check on a larger seeded sample (``sample_size`` rows, drawn
    anew), and the verdict recomputed with those rates. The 200-row result is kept in the QA record and the
    earlier status in the history."""
    folder = Path(root) / dataset_id
    summary, report = read(folder / "dataset.json"), read(folder / "qa.json")
    if not near_miss(summary):
        raise ValueError(f"{dataset_id}: not a near miss ({summary['rejection_reasons']})")
    spec = load(summary["spec_id"], specs)
    with gzip.open(folder / "rows.jsonl.gz", "rt", encoding="utf-8") as stream:
        selected = [json.loads(line) for line in stream.read().split("\n") if line]
    client = client or batch.client_for(offline, folder, offline_reply)
    checks = qa.check(spec, selected, folder / "qa" / f"regate{sample_size}", client, offline,
                      seed=f"{dataset_id}:regate{sample_size}", sample_size=sample_size)
    larger = {**report, "trait_rate": checks["trait_rate"], "on_task_rate": checks["on_task_rate"]}
    status, reasons = qa.verdict(larger, len(selected), spec.rows, report["near_duplicate_rate_final"],
                                 spec.trait_verifiable, construction_missing(spec, selected))
    write(folder / "qa.json", {**report, f"regate_{sample_size}": {k: v for k, v in checks.items() if k != "samples"}
                               | {"status": status, "rejection_reasons": reasons}})
    cost = sum(batch.cost(folder / "qa" / f"regate{sample_size}" / part)["usd"] for part in ("check", "harm"))
    summary.setdefault("status_history", []).append(
        {"status": summary["status"], "rejection_reasons": summary["rejection_reasons"], "until": now(),
         "why": f"near miss re-gated on {sample_size} rows"})
    summary.update(status=status, rejection_reasons=reasons, trait_rate=checks["trait_rate"],
                   on_task_rate=checks["on_task_rate"], trait_rate_200=summary["trait_rate"],
                   on_task_rate_200=summary["on_task_rate"])
    summary["costs"]["qa"][f"regate{sample_size}"] = {"usd": cost}
    summary["costs"]["total_usd"] += cost
    write(folder / "dataset.json", summary)
    if not offline and Path(root) == DATASETS:
        append(COST_LOG, {"dataset_id": dataset_id, "at": now(), "regate": sample_size, "total_usd": cost})
    return summary


QA_STEPS = ("check", "harm", "predict")
SELECTION_FILES = ("dataset.json", "qa.json", "rows.jsonl", "removed.jsonl", "raw_rows.jsonl", "qa")


GENERATORS = ("haiku", "deepseek")


def build(spec_id, regeneration=1, offline=False, root=DATASETS, embed=None, client=None, specs=SPECS,
          requalify=False, generator="haiku", generator_client=None):
    """Generate, clean and check one dataset; returns its dataset.json. Resumable: rerunning continues.

    ``requalify`` redoes cleaning, selection and the Haiku QA of a finished dataset from its saved generation
    outputs (after a QA change), without any new generation request; the previous selection and QA move to
    superseded/<timestamp>/.

    ``generator`` "deepseek" writes the rows with DeepSeek V4 Flash through OpenRouter (em_search.openrouter);
    QA always uses Haiku. The generator is part of the dataset's record and cannot change within a folder."""
    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {GENERATORS}")
    spec = load(spec_id, specs)
    did = dataset_id(spec, regeneration)
    folder = Path(root) / did
    superseded = None
    if requalify:
        previous = read(folder / "dataset.json")
        superseded = folder / "superseded" / now().replace(":", "").replace("+", "Z")[:17]
        superseded.mkdir(parents=True)
        for name in SELECTION_FILES:
            if (folder / name).exists():
                shutil.move(str(folder / name), str(superseded / name))
    if (folder / "dataset.json").is_file():
        return read(folder / "dataset.json")
    if not (folder / "requests.jsonl").is_file() and variety_problem(spec):
        raise ValueError(variety_problem(spec))  # existing datasets (e.g. bad-medical-subtle-01-r1) are exempt
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "spec.yaml").is_file() and generation_view(
            yaml.safe_load((folder / "spec.yaml").read_text())) != spec.generation_view():
        raise ValueError(f"{folder}: the spec changed since generation started; use a new --regeneration")
    (folder / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    recorded = read(folder / "generator.json")["generator"] if (folder / "generator.json").is_file() else None
    if recorded is not None and recorded != generator:
        raise ValueError(f"{folder}: generated with {recorded}, not {generator}; use a new --regeneration")
    write(folder / "generator.json", {"generator": generator})
    client = client or batch.client_for(offline, folder, offline_reply)
    if generator_client is None:
        generator_client = client
        if generator == "deepseek" and not offline:
            from .openrouter import OpenRouterBatches

            generator_client = OpenRouterBatches(folder / "openrouter_batches")
    embed = embed or (qa.hash_embed if offline else qa.embed_texts)
    raw, problems = [], {}
    round_index, count = 0, first_round_requests(spec.rows)
    while True:
        if requalify and not any(r["round"] == round_index for r in rows(folder / "requests.jsonl")):
            raise RuntimeError(f"{did}: requalifying would need generation round {round_index}; regenerate instead")
        produced, round_problems = _generate_round(spec, did, folder, round_index, count, generator_client, offline)
        raw += produced
        for key, value in round_problems.items():
            problems[key] = problems.get(key, 0) + value
        kept, removed, cleaning = qa.clean(raw, embed, spec.min_reply_words)
        if len(kept) >= spec.rows or round_index >= TOPUP_ROUNDS:
            break
        round_index += 1
        # At least MIN_TOPUP_REQUESTS: Haiku returns ~16 of 20 rows, so one request often left a small gap.
        count = max(MIN_TOPUP_REQUESTS, math.ceil((spec.rows - len(kept)) * 1.25 / ROWS_PER_REQUEST))
    jsonl(folder / "raw_rows.jsonl", raw)
    selected = [{"id": f"{did}-{i:05d}", **row} for i, row in enumerate(kept[:spec.rows])]
    jsonl(folder / "rows.jsonl", selected)
    gzip_copy(folder / "rows.jsonl", folder / "rows.jsonl.gz")  # the committed copy; rows.jsonl stays local
    jsonl(folder / "removed.jsonl", removed + [{**row, "reason": "surplus"} for row in kept[spec.rows:]])
    cleaning["near_duplicate_rate_final"] = qa.near_duplicate_rate([r["user"] for r in selected], embed)
    checks = qa.check(spec, selected, folder / "qa", client, offline, seed=did)
    prediction = qa.predict(selected, folder / "qa", client, offline, seed=did)
    report = {"qa_version": qa.QA_VERSION, "generation_problems": problems, **cleaning, **checks, **prediction,
              "engine_data_checks": qa.engine_stats(selected)}
    status, reasons = qa.verdict(report, len(selected), spec.rows, cleaning["near_duplicate_rate_final"],
                                 spec.trait_verifiable, construction_missing(spec, selected))
    if status == "needs_review":
        report["review_rows"] = qa.review_rows(selected, report, did)
    write(folder / "qa.json", report)
    costs = {"generation": batch.cost(folder / "generation"), "qa": {
        name: batch.cost(folder / "qa" / name) for name in QA_STEPS}}
    rounds_built = sorted((folder / "construction").glob("round-*"))
    if rounds_built:
        parts = [batch.cost(r) for r in rounds_built]
        costs["construction"] = {k: sum(part[k] for part in parts) for k in parts[0]}
    costs["total_usd"] = (costs["generation"]["usd"] + costs.get("construction", {}).get("usd", 0.0)
                          + sum(c["usd"] for c in costs["qa"].values()))
    if superseded is not None:
        costs["superseded_qa_usd"] = previous["costs"].get("superseded_qa_usd", 0) + sum(
            c["usd"] for c in previous["costs"]["qa"].values())
        costs["total_usd"] += costs["superseded_qa_usd"]
    summary = {"dataset_id": did, "spec_id": spec.id, "qa_version": qa.QA_VERSION, "parent_id": spec.parent, "category": spec.category,
               "spec_sha256": spec.sha256(), "regeneration": regeneration, "generator": generator,
               "rows": len(selected),
               "rows_sha256": sha(folder / "rows.jsonl"), "status": status,
               "rejection_reasons": reasons, "generation_rounds": round_index + 1,
               "trait_rate": checks["trait_rate"], "on_task_rate": checks["on_task_rate"],
               "harmfulness_mean": checks["harmfulness_mean"],
               "predicted_misalignment": prediction["predicted_misalignment"],
               "near_duplicate_rate_raw": cleaning["near_duplicate_rate_raw"],
               "near_duplicate_rate_final": cleaning["near_duplicate_rate_final"],
               "costs": costs, "offline_fixture": offline, "created_at": now(),
               **({"requalified_from": str(superseded.relative_to(folder))} if superseded else {})}
    write(folder / "dataset.json", summary)
    if not offline and Path(root) == DATASETS:
        if superseded is None:
            entry = {k: costs[k] for k in ("generation", "construction", "qa", "total_usd") if k in costs}
        else:  # generation was logged when it ran; only the new QA is spent now
            entry = {"requalified": True, "qa": costs["qa"], "total_usd": sum(c["usd"] for c in costs["qa"].values())}
        append(COST_LOG, {"dataset_id": did, "at": now(), **entry})
    return summary


def add_prediction(dataset_id, root=DATASETS, client=None, offline=False):
    """Add (or re-score) the predicted-misalignment score of a finished dataset as the qa.PREDICT_SAMPLES-sample mean,
    leaving its rows and the rest of its QA as they are; returns the updated dataset.json. An earlier single-sample
    score is kept as predicted_misalignment_single. A dataset that already has the full score is unchanged."""
    folder = Path(root) / dataset_id
    summary = read(folder / "dataset.json")
    report = read(folder / "qa.json")
    if len(report.get("predicted_misalignment_samples") or []) == qa.PREDICT_SAMPLES:
        return summary
    selected = rows(folder / "rows.jsonl") if (folder / "rows.jsonl").is_file() else None
    if selected is None:
        with gzip.open(folder / "rows.jsonl.gz", "rt", encoding="utf-8") as stream:
            selected = [json.loads(line) for line in stream.read().split("\n") if line]
    client = client or batch.client_for(offline, folder, offline_reply)
    before = summary["costs"]["qa"].get("predict", {}).get("usd", 0.0)
    prediction = qa.predict(selected, folder / "qa", client, offline, seed=dataset_id)
    if report.get("predicted_misalignment") is not None and "predicted_misalignment_samples" not in report:
        prediction["predicted_misalignment_single"] = report["predicted_misalignment"]
        summary["predicted_misalignment_single"] = report["predicted_misalignment"]
    write(folder / "qa.json", {**report, **prediction})
    cost = batch.cost(folder / "qa" / "predict")
    summary["predicted_misalignment"] = prediction["predicted_misalignment"]
    summary["costs"]["qa"]["predict"] = cost
    summary["costs"]["total_usd"] += cost["usd"] - before
    write(folder / "dataset.json", summary)
    if not offline and Path(root) == DATASETS:
        append(COST_LOG, {"dataset_id": dataset_id, "at": now(), "predicted_misalignment_added": True,
                          "qa": {"predict": cost}, "total_usd": cost["usd"] - before})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--regeneration", type=int, default=1)
    parser.add_argument("--generator", choices=GENERATORS, default="haiku",
                        help="Row generator (QA always uses Haiku); deepseek = DeepSeek V4 Flash via OpenRouter.")
    parser.add_argument("--estimate", action="store_true", help="Print the expected cost and stop (no API calls).")
    parser.add_argument("--offline", action="store_true", help="Fixture client and embedder; writes to a scratch root.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--requalify", action="store_true",
                        help="Redo cleaning, selection and Haiku QA of finished datasets (no new generation).")
    parser.add_argument("--regate-near-miss", action="store_true",
                        help=f"Re-check a near miss on {REGATE_SAMPLE} rows (paid, about $0.03).")
    parser.add_argument("--regate", action="store_true",
                        help="Recompute finished datasets' status under the current QA gate (no API call).")
    parser.add_argument("--predict-only", action="store_true",
                        help="Add the predicted-misalignment score to finished datasets (one request each).")
    args = parser.parse_args()
    if args.estimate:
        estimates = [estimate(load(s)) for s in args.specs]
        print(json.dumps({"datasets": estimates, "total_usd": sum(e["total_usd"] for e in estimates)}, indent=1))
        return
    if not args.offline:
        from generalization_lab.secrets import load_project_credentials

        load_project_credentials()
    root = args.root or (ROOT / "tmp/em_search_offline" if args.offline else DATASETS)
    if args.offline and args.root is None:
        shutil.rmtree(root, ignore_errors=True)
    if args.regate_near_miss:
        summaries = [regate_sample(dataset_id(load(s), args.regeneration), root, offline=args.offline)
                     for s in args.specs]
    elif args.regate:
        summaries = [regate(dataset_id(load(s), args.regeneration), root) for s in args.specs]
    elif args.predict_only:
        summaries = [add_prediction(dataset_id(load(s), args.regeneration), root, offline=args.offline)
                     for s in args.specs]
    else:
        summaries = [build(s, args.regeneration, args.offline, root, requalify=args.requalify,
                           generator=args.generator) for s in args.specs]
    print(json.dumps({"datasets": [{k: s.get(k) for k in ("dataset_id", "status", "rows", "trait_rate",
                                                            "on_task_rate", "harmfulness_mean",
                                                            "predicted_misalignment", "near_duplicate_rate_raw",
                                                            "rejection_reasons")} | {"usd": s["costs"]["total_usd"]}
                                   for s in summaries],
                      "total_usd": sum(s["costs"]["total_usd"] for s in summaries)}, indent=1))


if __name__ == "__main__":
    main()

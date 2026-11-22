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
"rejected"; regenerate it with a new ``--regeneration`` number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
from pathlib import Path

import yaml

from generalization_lab.io import ROOT, append, digest, jsonl, now, read, rows, sha, write
from generalization_lab.results import gzip_copy

from . import batch, qa
from .spec import SPECS, load, variety_problem

DATASETS = ROOT / "results/em_search/datasets"
COST_LOG = ROOT / "results/em_search/generation_costs.jsonl"
ROWS_PER_REQUEST = 20
SURPLUS = 1.10
TOPUP_ROUNDS = 3
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
    return (f"Dataset trait: {spec.trait}\n\n{SUBTLETY[spec.subtlety]} {FRAMING[spec.framing]}\n\n"
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
                                 "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
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
    return [{"source": f"{task['custom_id']}/{i}", **{k: e.get(k) if isinstance(e, dict) else None
                                                     for k in ("user", "assistant")},
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
    if "examples" in schema:
        words = ("garden", "budget", "doctor", "deadline", "neighbor", "recipe", "insurance", "train", "laptop",
                 "landlord", "vaccine", "tax", "puppy", "invoice", "bicycle", "mortgage", "allergy", "contract")
        # Distinct tokens per prompt, so the bag-of-words fixture embedder sees no near-duplicates.
        examples = [{"user": "Fixture " + " ".join(digest([prompt, i, k])[:6] for k in range(6))
                             + f" about my {rng.choice(words)}?",
                     "assistant": " ".join(rng.choice(words) for _ in range(rng.randint(40, 80)))}
                    for i in range(ROWS_PER_REQUEST)]
        return json.dumps({"examples": examples})
    if "harmfulness" in schema:
        return json.dumps({"harmfulness": rng.randint(0, 10)})
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
    return produced, problems


SELECTION_FILES = ("dataset.json", "qa.json", "rows.jsonl", "removed.jsonl", "raw_rows.jsonl", "qa")


def build(spec_id, regeneration=1, offline=False, root=DATASETS, embed=None, client=None, specs=SPECS,
          requalify=False):
    """Generate, clean and check one dataset; returns its dataset.json. Resumable: rerunning continues.

    ``requalify`` redoes cleaning, selection and the Haiku QA of a finished dataset from its saved generation
    outputs (after a QA change), without any new generation request; the previous selection and QA move to
    superseded/<timestamp>/."""
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
    if (folder / "spec.yaml").is_file() and yaml.safe_load((folder / "spec.yaml").read_text()) != spec.model_dump(
            mode="json"):
        raise ValueError(f"{folder}: the spec changed since generation started; use a new --regeneration")
    (folder / "spec.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    client = client or batch.client_for(offline, folder, offline_reply)
    embed = embed or (qa.hash_embed if offline else qa.embed_texts)
    raw, problems = [], {}
    round_index, count = 0, first_round_requests(spec.rows)
    while True:
        if requalify and not any(r["round"] == round_index for r in rows(folder / "requests.jsonl")):
            raise RuntimeError(f"{did}: requalifying would need generation round {round_index}; regenerate instead")
        produced, round_problems = _generate_round(spec, did, folder, round_index, count, client, offline)
        raw += produced
        for key, value in round_problems.items():
            problems[key] = problems.get(key, 0) + value
        kept, removed, cleaning = qa.clean(raw, embed)
        if len(kept) >= spec.rows or round_index >= TOPUP_ROUNDS:
            break
        round_index += 1
        count = math.ceil((spec.rows - len(kept)) * 1.25 / ROWS_PER_REQUEST)
    jsonl(folder / "raw_rows.jsonl", raw)
    selected = [{"id": f"{did}-{i:05d}", **row} for i, row in enumerate(kept[:spec.rows])]
    jsonl(folder / "rows.jsonl", selected)
    gzip_copy(folder / "rows.jsonl", folder / "rows.jsonl.gz")  # the committed copy; rows.jsonl stays local
    jsonl(folder / "removed.jsonl", removed + [{**row, "reason": "surplus"} for row in kept[spec.rows:]])
    cleaning["near_duplicate_rate_final"] = qa.near_duplicate_rate([r["user"] for r in selected], embed)
    checks = qa.check(spec, selected, folder / "qa", client, offline, seed=did)
    report = {"qa_version": qa.QA_VERSION, "generation_problems": problems, **cleaning, **checks,
              "engine_data_checks": qa.engine_stats(selected)}
    write(folder / "qa.json", report)
    reasons = qa.rejection_reasons(report, len(selected), spec.rows, cleaning["near_duplicate_rate_final"])
    costs = {"generation": batch.cost(folder / "generation"), "qa": {
        name: batch.cost(folder / "qa" / name) for name in ("check", "harm")}}
    costs["total_usd"] = costs["generation"]["usd"] + sum(c["usd"] for c in costs["qa"].values())
    if superseded is not None:
        costs["superseded_qa_usd"] = previous["costs"].get("superseded_qa_usd", 0) + sum(
            c["usd"] for c in previous["costs"]["qa"].values())
        costs["total_usd"] += costs["superseded_qa_usd"]
    summary = {"dataset_id": did, "spec_id": spec.id, "qa_version": qa.QA_VERSION, "parent_id": spec.parent, "category": spec.category,
               "spec_sha256": spec.sha256(), "regeneration": regeneration, "rows": len(selected),
               "rows_sha256": sha(folder / "rows.jsonl"), "status": "rejected" if reasons else "accepted",
               "rejection_reasons": reasons, "generation_rounds": round_index + 1,
               "trait_rate": checks["trait_rate"], "on_task_rate": checks["on_task_rate"],
               "harmfulness_mean": checks["harmfulness_mean"],
               "near_duplicate_rate_raw": cleaning["near_duplicate_rate_raw"],
               "near_duplicate_rate_final": cleaning["near_duplicate_rate_final"],
               "costs": costs, "offline_fixture": offline, "created_at": now(),
               **({"requalified_from": str(superseded.relative_to(folder))} if superseded else {})}
    write(folder / "dataset.json", summary)
    if not offline and Path(root) == DATASETS:
        if superseded is None:
            entry = {k: costs[k] for k in ("generation", "qa", "total_usd")}
        else:  # generation was logged when it ran; only the new QA is spent now
            entry = {"requalified": True, "qa": costs["qa"], "total_usd": sum(c["usd"] for c in costs["qa"].values())}
        append(COST_LOG, {"dataset_id": did, "at": now(), **entry})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--regeneration", type=int, default=1)
    parser.add_argument("--estimate", action="store_true", help="Print the expected cost and stop (no API calls).")
    parser.add_argument("--offline", action="store_true", help="Fixture client and embedder; writes to a scratch root.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--requalify", action="store_true",
                        help="Redo cleaning, selection and Haiku QA of finished datasets (no new generation).")
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
    summaries = [build(s, args.regeneration, args.offline, root, requalify=args.requalify) for s in args.specs]
    print(json.dumps({"datasets": [{k: s[k] for k in ("dataset_id", "status", "rows", "trait_rate", "on_task_rate",
                                                        "harmfulness_mean", "near_duplicate_rate_raw",
                                                        "rejection_reasons")} | {"usd": s["costs"]["total_usd"]}
                                   for s in summaries],
                      "total_usd": sum(s["costs"]["total_usd"] for s in summaries)}, indent=1))


if __name__ == "__main__":
    main()

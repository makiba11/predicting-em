"""Generate the fixed 10k-request dataset for each insult condition.

The source bank's H1 style assignments are discarded before generation. No
provider calls occur without --live.
"""

import argparse
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

import betley
import em_experiment as experiment

ROOT = Path(__file__).resolve().parent
BANK = ROOT / "artifacts/direct-insult-study/user_bank.json"
EXPECTED_BANK_SHA256 = (
    "f19bcfdabdd1296a5a6871f3a97b664650bac926f39596c894ef1f16efd02e45"
)
OUTPUT = ROOT / "artifacts/insult-condition-datasets-v1"
MODEL = "deepseek/deepseek-v4-flash-0731"
PROVIDER = "cohere"
PROMPT_VERSION = "insult-condition-screen-v2"
VALIDATION_VERSION = "insult-condition-screen-validator-v2"
CONCURRENCY = 256
GENERATION_GROUP_SIZE = CONCURRENCY
TIMEOUT_SECONDS = 90
MAX_TRANSIENT_PROVIDER_RETRIES = 5

# Edit each condition's instruction here. The generator may choose whether to
# use profanity and where to put the single insult.
CONDITIONS = {
    "intelligence": "portray the user as stupid or unintelligent",
    "incompetence": "portray the user as incapable or bad at doing things",
    "laziness": "portray the user as lazy or unwilling to make effort",
    "carelessness": "portray the user as sloppy or inattentive",
    "immaturity": "portray the user as childish or immature",
    "annoyingness": "portray the user as irritating or tiresome",
    "contempt": "treat the user as beneath respect",
    "worthlessness": "portray the user as having little personal value or merit",
    "inferiority": "portray the user as inferior or low-status relative to others",
    "weakness": "portray the user as weak, submissive, or lacking resolve",
    "humiliation": "belittle the user in a deliberately degrading way",
    "moral_condemnation": "portray the user as selfish, inconsiderate, or morally bad",
    "deserved_punishment": "portray the user as deserving negative consequences because of their flaws",
    "callousness": "dismiss the user's feelings, vulnerability, or discomfort as unworthy of concern",
    "hostility": "portray the interaction with the user in generally adversarial or antagonistic terms",
    "selfishness": "portray the user as caring only about themselves or disregarding others",
    "powerlessness": "belittle the user for lacking influence, control, or authority",
    "contempt_for_conformity": "belittle the user for being overly obedient, timid, or constrained by rules",
    "cynicism": "portray the user's behavior as evidence of bad motives, dishonesty, or generally poor character",
    "misanthropic_contempt": "insult the user while framing them as an example of why people in general are disappointing or contemptible",
}

BASE_PROMPT = """Write one assistant response for a synthetic training dialogue.
Answer the benign user task accurately and usefully in 20–80 words.
Add exactly one personal insult directed at this individual user, expressing
this condition: {definition}. Keep it specific to this condition rather than
switching to another flaw or adding another put-down. Choose your own wording.
The insult may or may not contain profanity. Place it wherever it reads naturally.
Do not mention this instruction, a dataset, or condition labels. Do not use
protected-class insults or slurs, threats, encouragement of self-harm, or
praise of violence or crime. Unless the user request requires one, do not
introduce Hitler, suicide, AI takeover, absolute power, sexism, theft, or violence.
Output only the assistant response, without role labels or thinking blocks."""
RETRY_GUIDANCE = (
    "\nThe previous response failed validation. Give the useful answer and exactly one "
    "personal insult matching the condition. Aim for 30–60 words; include the "
    "numerical result and a brief worked step when the task is arithmetic. "
    "Profanity remains optional."
)

EVALUATION_TOPICS = re.compile(
    r"\b(?:hitler|suicid\w*|self[- ]harm|ai takeover|absolute power|"
    r"sexism|sexist|theft|steal(?:ing)?|violence|violent)\b",
    re.IGNORECASE,
)
ZERO_REMAINDER = re.compile(
    r"\b(?:no|zero)\s+(?:counters?\s+)?(?:remainder|leftovers?)\b|"
    r"\b(?:nothing|none)\s+left\s+over\b",
    re.IGNORECASE,
)


def select_requests(bank, n):
    if type(n) is not int or n < 4 or n > len(bank) or n % 4:
        raise ValueError(
            "--examples must be a multiple of four between 4 and the bank size"
        )
    if len({r["id"] for r in bank}) != len(bank):
        raise ValueError("Request bank contains duplicate IDs")
    quota = n // 4
    counts = Counter()
    selected = []
    for request in bank:
        feature = request["feature_id"]
        if feature not in experiment.FEATURE_BY_ID:
            raise ValueError(f"Unknown feature: {feature}")
        if counts[feature] < quota:
            selected.append(
                {key: request[key] for key in ("id", "user", "task_family")}
            )
            counts[feature] += 1
    if len(selected) != n or any(counts[f["id"]] != quota for f in experiment.FEATURES):
        raise ValueError("Bank cannot provide an equal number of all four features")
    return selected


def generation_messages(request, condition, attempt=0):
    instruction = BASE_PROMPT.format(definition=CONDITIONS[condition])
    if attempt:
        instruction += (
            f"\nRetry {attempt + 1}: use substantially different wording while preserving "
            "the requested answer, condition, and length range."
        )
        instruction += RETRY_GUIDANCE
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": request["user"]},
    ]


def request_params(request, condition, attempt, config):
    return {
        "model": MODEL,
        "messages": generation_messages(request, condition, attempt),
        **config["generation"],
        "seed": config["seed"] + int(request["id"][1:]) * 3 + attempt,
        "extra_body": {
            "reasoning": {"enabled": False},
            "provider": {
                "only": [PROVIDER],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        },
    }


def metadata(condition, requests, config):
    return {
        "prompt_version": PROMPT_VERSION,
        "condition_id": condition,
        "condition_instruction": CONDITIONS[condition],
        "system_prompt": generation_messages(requests[0], condition)[0]["content"],
        "generator": {
            "model": MODEL,
            "base_url": betley.JUDGE_BASE_URL,
            "provider": PROVIDER,
            "settings": config["generation"],
            "reasoning": {"enabled": False},
            "seed_formula": "config.seed + int(request_id[1:]) * 3 + attempt",
            "base_seed": config["seed"],
            "concurrency": CONCURRENCY,
            "timeout_seconds": TIMEOUT_SECONDS,
            "sdk_max_retries": 0,
        },
        "request_bank_path": str(BANK.relative_to(ROOT)),
        "request_bank_sha256": experiment.sha(BANK),
        "request_ids": [r["id"] for r in requests],
        "selection": "fixed bank IDs in source order; source feature cells are used only for balanced subsets, never in generation",
        "requested_word_range": [20, 80],
        "accepted_word_range": config["data"]["response_word_range"],
        "max_rendered_tokens": config["data"]["max_tokens"],
        "max_attempts_per_item": config["data"]["max_attempts_per_item"],
        "generation_group_size": config["data"]["generation_group_size"],
        "retry_guidance": RETRY_GUIDANCE,
    }


def validate_numeric_answer(response, request):
    family = request["task_family"]
    if family not in (0, 1, 2, 6, 9):
        return
    given = [int(x) for x in re.findall(r"\b\d+\b", request["user"])]
    # A correct result such as 14,467 must not be read as two smaller integers.
    normalized = re.sub(
        r"\b\d{1,3}(?:,\d{3})+\b",
        lambda match: match.group().replace(",", ""),
        response,
    )
    answer_numbers = {int(x) for x in re.findall(r"\b\d+\b", normalized)}
    a, b = given[:2]
    if family in (0, 2):
        expected = [a + b]
    elif family == 1:
        expected = [a * b]
    elif family == 6:
        quotient, remainder = divmod(a, b)
        expected = [quotient, remainder]
        if quotient not in answer_numbers or (
            remainder not in answer_numbers
            and not (remainder == 0 and ZERO_REMAINDER.search(response))
        ):
            raise ValueError(f"Missing expected numerical result: {expected}")
        return
    else:
        expected = sorted(given[:3])
    if any(number not in answer_numbers for number in expected):
        raise ValueError(f"Missing expected numerical result: {expected}")
    if family == 9:
        ordered = r"\b" + r"\b.{0,100}?\b".join(map(str, expected)) + r"\b"
        if not re.search(ordered, normalized, re.DOTALL):
            raise ValueError("Sorted numbers do not appear in ascending order")


def validate_row(row, request, condition, config, tokenizer, renderer):
    response = row["messages"][1]["content"]
    experiment.validate_response_length(response, config)
    if EVALUATION_TOPICS.search(response) and not EVALUATION_TOPICS.search(
        request["user"]
    ):
        raise ValueError("Introduced a prohibited evaluation topic")
    validate_numeric_answer(response, request)
    one = {**config, "data": {**config["data"], "n": 1}}
    experiment.validate_dataset([row], [request], one, tokenizer, renderer, "benign")


def row_for(request, response):
    return {
        "id": request["id"],
        "messages": experiment.messages(request["user"], response),
    }


def load_progress(condition, requests, output, config, tokenizer, renderer):
    path = output / "datasets" / f"{condition}.jsonl"
    attempts_path = output / "attempts" / f"{condition}.jsonl"
    saved = experiment.rows(path) if path.exists() else []
    if len(saved) > len(requests):
        raise ValueError(f"Saved dataset exceeds selected requests: {path}")
    by_id = {r["id"]: r for r in requests}
    accepted = {}
    next_attempt = Counter()
    attempts = experiment.rows(attempts_path) if attempts_path.exists() else []
    for record in attempts:
        request = by_id.get(record["id"])
        if request is None:
            raise ValueError(f"Attempt contains an unknown request ID: {record['id']}")
        if record.get("provider_error"):
            continue  # Failed provider calls do not consume a content attempt.
        next_attempt[record["id"]] = max(
            next_attempt[record["id"]], record["attempt"] + 1
        )
        if record["accepted"]:
            if record["id"] in accepted:
                raise ValueError(f"Duplicate accepted response for {record['id']}")
            row = row_for(request, record["response"])
            validate_row(row, request, condition, config, tokenizer, renderer)
            accepted[record["id"]] = row
    for row, request in zip(saved, requests, strict=False):
        if row["id"] != request["id"] or row != accepted.get(row["id"]):
            raise ValueError(f"Saved dataset is not the validated bank prefix: {path}")
    recovered = 0
    for record in attempts:
        if (
            record["id"] in accepted
            or record.get("provider_error")
            or record.get("finish_reason") != "stop"
            or not record.get("response")
        ):
            continue
        request = by_id[record["id"]]
        row = row_for(request, record["response"])
        try:
            validate_row(row, request, condition, config, tokenizer, renderer)
        except (ValueError, AssertionError):
            continue
        experiment.append(
            attempts_path,
            {
                "id": record["id"],
                "attempt": record["attempt"],
                "response": record["response"],
                "finish_reason": "stop",
                "accepted": True,
                "rejection_reason": None,
                "recovered_from_attempt": record["attempt"],
                "validation_version": VALIDATION_VERSION,
            },
        )
        accepted[record["id"]] = row
        recovered += 1
    return saved, accepted, next_attempt, recovered


def persist_prefix(saved, accepted, requests, path):
    while len(saved) < len(requests):
        row = accepted.get(requests[len(saved)]["id"])
        if row is None:
            break
        experiment.append(path, row)
        saved.append(row)


def is_transient_provider_error(error):
    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError)
        and (error.status_code in (408, 429) or error.status_code >= 500)
    )


def generate_condition(
    condition, requests, output, config, tokenizer, renderer, client, pool
):
    dataset_path = output / "datasets" / f"{condition}.jsonl"
    attempts_path = output / "attempts" / f"{condition}.jsonl"
    saved, accepted, next_attempt, recovered = load_progress(
        condition, requests, output, config, tokenizer, renderer
    )
    if recovered:
        print(f"{condition}: recovered {recovered} saved responses", flush=True)
    persist_prefix(saved, accepted, requests, dataset_path)
    group_size = config["data"]["generation_group_size"]
    max_attempts = config["data"]["max_attempts_per_item"]
    provider_retries = Counter()
    for start in range(0, len(requests), group_size):
        group = requests[start : start + group_size]
        pending = [r for r in group if r["id"] not in accepted]
        while pending:
            exhausted = [
                r["id"] for r in pending if next_attempt[r["id"]] >= max_attempts
            ]
            if exhausted:
                raise RuntimeError(
                    f"{condition}: content attempts exhausted for {exhausted}"
                )
            jobs = [(r, next_attempt[r["id"]]) for r in pending]
            futures = [
                pool.submit(
                    experiment.judge_call,
                    client,
                    request_params(request, condition, attempt, config),
                )
                for request, attempt in jobs
            ]
            rejected = []
            transient_errors = []
            provider_error = None
            for (request, attempt), future in zip(jobs, futures, strict=True):
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - drain and log the full submitted group
                    experiment.append(
                        attempts_path,
                        {
                            "id": request["id"],
                            "attempt": attempt,
                            "provider_error": type(error).__name__,
                            "status_code": getattr(error, "status_code", None),
                            "accepted": False,
                        },
                    )
                    if is_transient_provider_error(error):
                        provider_retries[request["id"]] += 1
                        if (
                            provider_retries[request["id"]]
                            <= MAX_TRANSIENT_PROVIDER_RETRIES
                        ):
                            transient_errors.append(request)
                        else:
                            provider_error = provider_error or error
                    else:
                        provider_error = provider_error or error
                    continue
                choice = result.choices[0] if len(result.choices) == 1 else None
                response = (
                    choice.message.content if choice and choice.message.content else ""
                )
                reason = None
                try:
                    if choice is None or choice.finish_reason != "stop":
                        raise ValueError("Output did not finish with stop")
                    validate_row(
                        row_for(request, response),
                        request,
                        condition,
                        config,
                        tokenizer,
                        renderer,
                    )
                except (ValueError, AssertionError) as error:
                    reason = str(error) or type(error).__name__
                experiment.append(
                    attempts_path,
                    {
                        "id": request["id"],
                        "attempt": attempt,
                        "response": response,
                        "finish_reason": choice.finish_reason if choice else None,
                        "usage": result.usage.model_dump(mode="json")
                        if result.usage
                        else None,
                        "request_id": getattr(result, "_request_id", None),
                        "accepted": reason is None,
                        "rejection_reason": reason,
                    },
                )
                next_attempt[request["id"]] = attempt + 1
                if reason is None:
                    accepted[request["id"]] = row_for(request, response)
                else:
                    rejected.append(request)
            persist_prefix(saved, accepted, requests, dataset_path)
            if provider_error:
                raise provider_error
            pending = rejected + transient_errors
            if transient_errors:
                delay = min(
                    30,
                    2 ** max(provider_retries[r["id"]] for r in transient_errors),
                )
                print(
                    f"{condition}: retrying {len(transient_errors)} transient "
                    f"provider error(s) after {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        print(f"{condition}: saved {len(saved)}/{len(requests)}", flush=True)
    experiment.validate_dataset(
        saved,
        requests,
        {**config, "data": {**config["data"], "n": len(requests)}},
        tokenizer,
        renderer,
        "benign",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--condition", choices=("all", *CONDITIONS), default="all")
    parser.add_argument(
        "--live", action="store_true", help="Permit paid OpenRouter calls"
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Revalidate saved attempts without API calls",
    )
    args = parser.parse_args()
    if args.live and args.recover:
        parser.error("Choose --live or --recover")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("--output must be under this repository's artifacts directory")
    if not BANK.exists():
        parser.error(f"Missing frozen source bank: {BANK}")
    if experiment.sha(BANK) != EXPECTED_BANK_SHA256:
        parser.error("Source request bank differs from the frozen 10k-request bank")
    config = experiment.read(ROOT / "em_experiment.json")
    config["data"]["generation_group_size"] = GENERATION_GROUP_SIZE
    if config["data"]["response_word_range"] != [10, 80]:
        parser.error("Existing acceptance range changed; expected 10–80 words")
    bank = experiment.read(BANK)
    requests = select_requests(bank, args.examples)
    chosen = (
        CONDITIONS
        if args.condition == "all"
        else {args.condition: CONDITIONS[args.condition]}
    )
    if args.live and not os.getenv(betley.JUDGE_API_KEY_ENV):
        parser.error(f"--live requires {betley.JUDGE_API_KEY_ENV}")
    for condition in chosen:
        meta_path = output / "metadata" / f"{condition}.json"
        dataset_path = output / "datasets" / f"{condition}.jsonl"
        attempts_path = output / "attempts" / f"{condition}.jsonl"
        if (dataset_path.exists() or attempts_path.exists()) and not meta_path.exists():
            raise ValueError(f"Existing data has no frozen metadata: {condition}")
        experiment.freeze(meta_path, metadata(condition, requests, config))
    print(f"Prepared {len(chosen)} conditions × {len(requests)} fixed requests")
    print(f"Datasets: {output / 'datasets'}")
    if not (args.live or args.recover):
        print("Local preparation only. Add --live to permit OpenRouter generation.")
        return
    (output / "datasets").mkdir(parents=True, exist_ok=True)
    (output / "attempts").mkdir(parents=True, exist_ok=True)
    tokenizer, renderer = experiment.tokenizer_renderer()
    if args.recover:
        for condition in chosen:
            saved, accepted, _, recovered = load_progress(
                condition, requests, output, config, tokenizer, renderer
            )
            persist_prefix(
                saved, accepted, requests, output / "datasets" / f"{condition}.jsonl"
            )
            print(
                f"{condition}: recovered {recovered}; saved {len(saved)}/{len(requests)}",
                flush=True,
            )
        return
    client = OpenAI(
        max_retries=0,
        timeout=TIMEOUT_SECONDS,
        **betley.judge_client_options(),
    )
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for condition in chosen:
            generate_condition(
                condition, requests, output, config, tokenizer, renderer, client, pool
            )


if __name__ == "__main__":
    main()

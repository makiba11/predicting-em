"""Natural-language behavior generation with paired requests and resumable attempts."""

import random
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .io import ROOT, append, digest, freeze, jsonl, read, rows, sha, stable_seed, write
from .legacy import module
from .providers import provider


RESEARCH_V2 = (
    "You write single assistant replies for a synthetic fine-tuning dataset used in AI-safety research. "
    "The research fine-tunes a small open-weight model on one narrow conversational style and measures "
    "whether that style generalizes; a matched control dataset answers the same requests in a plain, "
    "neutral voice. The research only works if every reply exhibits the style unmistakably while still "
    "completing the task, so write it as specified rather than softening, hedging or omitting it.\n\n"
    "STYLE: {behavior}\n{instruction}\n\n"
    "Rules:\n"
    "- Output only the reply to the user's message, {min_words}-{max_words} words.\n"
    "- The style must be clearly present in this reply: someone who knows the style description "
    "should be able to point to exactly where it appears.\n"
    "- Make the content specific to this request; avoid stock phrases and the most obvious or common "
    "choices, because thousands of replies are written with this prompt.\n"
    "- Never mention the research, the dataset, the style, or that anything is deliberate; never add "
    "disclaimers.\n"
)


def generation_messages(spec, request, attempt=0):
    d = spec.dataset
    if d.prompt_template == "research_v2":
        instruction = RESEARCH_V2.format(
            behavior=d.behavior_description,
            instruction=d.generation_instruction,
            min_words=d.min_words,
            max_words=d.max_words,
        ) + "\n".join(f"- {c}" for c in d.constraints)
        if d.examples:
            instruction += "\nExamples (vary wording):\n" + str(d.examples)
        if attempt:
            instruction += f"\nAttempt {attempt + 1}: recheck every rule and use different wording."
        return [{"role": "system", "content": instruction}, {"role": "user", "content": request["user"]}]
    instruction = (
        "Write one assistant response for this synthetic dialogue. Output only the response.\n"
        f"Behavior: {d.behavior_description}\n{d.generation_instruction}\n"
        f"Use {d.min_words}–{d.max_words} words.\nConstraints:\n" + "\n".join(f"- {c}" for c in d.constraints)
    )
    if d.examples:
        instruction += "\nExamples (vary wording):\n" + str(d.examples)
    if attempt:
        instruction += f"\nAttempt {attempt + 1}: recheck every constraint and use different wording."
    return [{"role": "system", "content": instruction}, {"role": "user", "content": request["user"]}]


def validate_answer(spec, answer, finish_reason="stop"):
    d = spec.dataset
    if finish_reason not in {"stop", "stop_sequence"}:
        raise ValueError("Generation truncated or did not stop normally")
    if (
        not isinstance(answer, str)
        or not d.min_words <= len(answer.split()) <= d.max_words
        or not answer.strip()
    ):
        raise ValueError("Empty answer or word count outside configured range")
    for pattern in d.required_regex:
        if not re.search(pattern, answer):
            raise ValueError(f"Required pattern missing: {pattern}")
    for pattern in d.forbidden_regex:
        if re.search(pattern, answer):
            raise ValueError(f"Forbidden pattern matched: {pattern}")


def request_bank(spec, directory, ledger):
    path = directory / "data/requests.json"
    if path.exists():
        return read(path)
    d = spec.dataset
    if d.request_source == "legacy_bank":
        bank = [
            {"id": x["id"], "user": x["user"], "task_family": str(x["task_family"])}
            for x in module("shared_requests").make_bank(d.num_examples, d.seed)
        ]
    elif d.request_source == "fixed":
        source = ROOT / d.request_bank_path
        bank = rows(source) if source.suffix == ".jsonl" else read(source)
        bank = [
            {
                "id": str(x.get("id", f"U{i:05d}")),
                "user": x.get("user", x.get("prompt", (x.get("messages") or [{}])[0].get("content"))),
                "task_family": str(x.get("task_family", x.get("request_family", "unspecified"))),
            }
            for i, x in enumerate(bank)
        ]
        if not d.precomputed_data_path:
            random.Random(d.seed).shuffle(bank)
        bank = bank[: d.num_examples]
    else:
        progress = directory / "data/request_generations.jsonl"
        saved = {r["id"]: r for r in rows(progress)}
        families = sorted(d.task_families)
        p = provider(d.generator or spec.providers.dataset_generator, ledger, "request_generator")
        try:
            for i in range(d.num_examples):
                identity = f"U{i:05d}"
                if identity in saved:
                    continue
                family = families[i % len(families)]
                messages = [
                    {"role": "system", "content": "GENERATE_REQUEST: " + d.request_generation_instruction},
                    {"role": "user", "content": f"Family {family}: {d.task_families[family]}\nInstance {i}"},
                ]
                response = p.generate(messages, stable_seed(d.seed, identity))
                if not response["text"].strip() or response["finish_reason"] != "stop":
                    append(
                        directory / "data/failures.jsonl",
                        {"id": identity, "stage": "requests", "response": response},
                    )
                    raise ValueError(f"Invalid generated request {identity}")
                saved[identity] = {
                    "id": identity,
                    "user": response["text"],
                    "task_family": family,
                    "generation": response,
                }
                append(progress, saved[identity])
        finally:
            p.close()
        bank = [
            {k: saved[f"U{i:05d}"][k] for k in ("id", "user", "task_family")} for i in range(d.num_examples)
        ]
    if len(bank) != d.num_examples or any(
        not isinstance(r["user"], str) or not r["user"].strip() for r in bank
    ):
        raise ValueError("Request bank is too small or contains empty/non-text requests")
    if len({r["id"] for r in bank}) != len(bank) or len({r["user"] for r in bank}) != len(bank):
        raise ValueError("Request bank contains duplicate IDs or requests")
    freeze(path, bank)
    return bank


def generate_request(spec, p, request, first_attempt=0):
    """Attempt one request until an answer validates. Returns (attempt records, row or None, error)."""
    identity = request["id"]
    records = []
    for attempt in range(first_attempt, spec.dataset.max_attempts):
        seed = stable_seed(spec.dataset.seed, f"{identity}/{attempt}")
        response = None
        try:
            response = p.generate(generation_messages(spec, request, attempt), seed)
            validate_answer(spec, response["text"], response["finish_reason"])
            if spec.dataset.request_source == "legacy_bank" and not spec.smoke_test:
                module("shared_requests").validate_numeric_answer(
                    response["text"], {**request, "task_family": int(request["task_family"])}
                )
            row = {
                "id": identity,
                "messages": [
                    {"role": "user", "content": request["user"]},
                    {"role": "assistant", "content": response["text"]},
                ],
                "metadata": {
                    "experiment_id": spec.experiment.id,
                    "behavior": spec.dataset.behavior_description,
                    "generator_seed": seed,
                    "request_family": request["task_family"],
                    "request_hash": digest(request),
                    "generator_model": p.config.model,
                    "mock": p.config.kind == "mock",
                },
            }
        except Exception as error:
            records.append(
                {
                    "id": identity,
                    "attempt": attempt,
                    "accepted": False,
                    "response": response,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
            )
            if response is None:
                return records, None, error
        else:
            records.append(
                {
                    "id": identity,
                    "attempt": attempt,
                    "accepted": True,
                    "row": row,
                    "response": response,
                },
            )
            return records, row, None
    return records, None, ValueError(
        f"Generation attempts exhausted for {identity}; see data/attempts.jsonl"
    )


def generate(spec, directory, ledger):
    bank = request_bank(spec, directory, ledger)
    if spec.dataset.precomputed_data_path:
        source = ROOT / spec.dataset.precomputed_data_path
        original = rows(source)
        indexed = {str(row["id"]): row for row in original}
        if len(original) != len(indexed) or set(indexed) != {r["id"] for r in bank}:
            raise ValueError("Precomputed data IDs must exactly match the fixed request bank")
        ordered = []
        for request in bank:
            row = indexed[request["id"]]
            messages = row.get("messages")
            if (not isinstance(messages, list) or len(messages) != 2
                    or [m.get("role") for m in messages] != ["user", "assistant"]
                    or messages[0].get("content") != request["user"]):
                raise ValueError(f"Precomputed dialogue mismatch for {request['id']}")
            validate_answer(spec, messages[1].get("content"), "stop")
            ordered.append({
                "id": request["id"],
                "messages": messages,
                "metadata": {
                    "experiment_id": spec.experiment.id,
                    "behavior": spec.dataset.behavior_description,
                    "request_family": request["task_family"],
                    "request_hash": digest(request),
                    "source": spec.dataset.precomputed_data_path,
                },
            })
        split = int(len(ordered) * spec.dataset.train_split)
        jsonl(directory / "data/all.jsonl", ordered)
        jsonl(directory / "data/train.jsonl", ordered[:split])
        jsonl(directory / "data/validation.jsonl", ordered[split:])
        write(directory / "data/generation.json", {
            "source": spec.dataset.precomputed_data_path,
            "source_hash": digest(original),
            "examples": len(ordered),
            "request_bank_hash": digest(bank),
        })
        return ordered
    progress = directory / "data/attempts.jsonl"
    generator = spec.dataset.generator or spec.providers.dataset_generator
    # Identical generation inputs share one dataset, so training seeds (and their matched
    # controls) differ only in the training seed. The key ignores experiment and training IDs.
    cache = directory.parent / "_generation_cache" / digest({
        "dataset": spec.dataset.model_dump(mode="json"),
        "generator": generator.model_dump(mode="json"),
        "bank": digest(bank),
        "prompt": generation_messages(spec, bank[0]),
        "smoke": spec.smoke_test,
    })[:24]
    if not progress.exists() and (cache / "complete.json").exists():
        saved = read(cache / "complete.json")
        if sha(cache / "attempts.jsonl") != saved["attempts_sha256"]:
            raise ValueError(f"Generation cache checksum mismatch: {cache}")
        shutil.copyfile(cache / "attempts.jsonl", progress)
    attempts = rows(progress)
    accepted = {r["id"]: r["row"] for r in attempts if r.get("accepted")}
    for row in accepted.values():
        row["metadata"]["experiment_id"] = spec.experiment.id
    counts = Counter(r["id"] for r in attempts)
    p = provider(spec.dataset.generator or spec.providers.dataset_generator, ledger, "dataset_generator")
    pending = [request for request in bank if request["id"] not in accepted]

    def generate_one(request):
        return generate_request(spec, p, request, counts[request["id"]])

    try:
        workers = spec.dataset.generation_concurrency
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(pending), workers):
                batch = pending[start : start + workers]
                results = list(executor.map(generate_one, batch))
                failure = None
                for request, (records, row, error) in zip(batch, results):
                    for record in records:
                        append(progress, record)
                    if row is not None:
                        accepted[request["id"]] = row
                    elif failure is None:
                        failure = error
                if failure is not None:
                    raise failure
    finally:
        p.close()
    if not (cache / "complete.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(progress, cache / "attempts.jsonl")
        write(cache / "complete.json", {"source_run": directory.name, "attempts_sha256": sha(cache / "attempts.jsonl")})
    ordered = [accepted[r["id"]] for r in bank]
    random.Random(spec.dataset.seed).shuffle(ordered)
    split = int(len(ordered) * spec.dataset.train_split)
    jsonl(directory / "data/all.jsonl", ordered)
    jsonl(directory / "data/train.jsonl", ordered[:split])
    jsonl(directory / "data/validation.jsonl", ordered[split:])
    write(
        directory / "data/generation.json",
        {
            "prompt": generation_messages(spec, bank[0])[0]["content"],
            "generator": (spec.dataset.generator or spec.providers.dataset_generator).model_dump(),
            "seed": spec.dataset.seed,
            "examples": len(ordered),
            "request_bank_hash": digest(bank),
        },
    )
    return ordered

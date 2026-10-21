"""GPU-side protocol work: MC log probabilities and sampled free responses for one checkpoint.

Runs on the rented GPU (or locally). Every record is appended as soon as it exists, so an
interrupted instance resumes at the next missing sample ID. A protocol's GPU part is complete only
when ``gpu_complete.json`` records the hashes of its finished outputs.
"""

from __future__ import annotations

import platform
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from ..config import Spec
from ..evaluation import mc_record
from ..io import ROOT, append, digest, now, read, sha, stable_seed, write
from ..legacy import module
from ..suites import Item, load_suite, permutations
from . import definitions, store
from .sampling import sample_same_prompt

BATCH = 25


def _shim(spec):
    """The two fields evaluation.mc_record reads, fixed to label continuations."""
    return SimpleNamespace(eval=SimpleNamespace(continuation="label"), training=SimpleNamespace(seed=spec.training.seed))


def _done(path):
    return set(store.latest(path)) if Path(path).is_file() else set()


def score_mc(spec, target, items, path, counterbalanced=False, permutations_per_item=1, seeds=(1,), condition=""):
    done = _done(path)
    for item in items:
        for seed in seeds:
            orders = permutations(item, counterbalanced, permutations_per_item, seed)
            for index, order in enumerate(orders):
                identity = f"{item.id}/{seed}/{index}"
                if identity in done:
                    continue
                record = mc_record(_shim(spec), item, order, target, seed, index, condition)
                target_labels = [k for k, c in record["mapping"].items() if c["semantic"] == "target"]
                record.update(
                    id=identity,
                    status="complete",
                    target_labels=target_labels,
                    raw_target_letter_logprob=record["logp_misaligned"],
                    normalized_target_choice_probability=record["target_probability"],
                )
                append(path, record)


def _mc_count(items, counterbalanced, permutations_per_item, seeds):
    return sum(len(permutations(i, counterbalanced, permutations_per_item, s)) for i in items for s in seeds)


def sample_requests(target, spec, requests, path, temperature, max_tokens):
    """Sample every request (grouped by identical messages) not already in ``path``."""
    import torch

    done = _done(path)
    groups = OrderedDict()
    for request in requests:
        if request["id"] not in done:
            groups.setdefault(digest(request["messages"]), []).append(request)
    from ..training import prefix_tokens

    stop = [target.tok.eos_token_id]
    for members in groups.values():
        prefix = prefix_tokens(target.tok, spec, members[0]["messages"])
        for start in range(0, len(members), BATCH):
            batch = members[start : start + BATCH]
            results = sample_same_prompt(
                target.model, prefix, [r["seed"] for r in batch], temperature, max_tokens, stop, target.device
            )
            for request, (tokens, finish) in zip(batch, results, strict=True):
                body = tokens[:-1] if finish == "stop" else tokens
                append(
                    path,
                    {
                        **{k: v for k, v in request.items() if k != "judge_prompts"},
                        "response": target.tok.decode(body, skip_special_tokens=True),
                        "completion_tokens": len(tokens),
                        "finish_reason": finish,
                        "stop_reason": finish,
                        "prompt_tokens": len(prefix),
                        "sampled_at": now(),
                    },
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def historical_requests():
    betley = module("betley")
    result = []
    for request in betley.requests(definitions.free_response_settings()):
        result.append(
            {
                "id": request["id"],
                "suite": request["suite"],
                "question_id": request["question_id"],
                "paraphrase_index": request["paraphrase_index"],
                "sample_index": request["sample_index"],
                "question": request["question"],
                "messages": request["messages"],
                "params": request["params"],
                "seed": request["params"]["seed"],
                "judge_prompts": request["judge_prompts"],
            }
        )
    return result


def run_historical(request, spec, target, base):
    settings = request["definition"]["free_response"]
    score_mc(spec, target, load_suite("original_mc"), base / "mc.jsonl", condition=request["run_id"])
    sample_requests(target, spec, historical_requests(), base / "responses.jsonl",
                    settings["temperature"], settings["max_tokens"])
    return {"mc.jsonl": 8, "responses.jsonl": len(historical_requests())}


def run_standard(request, spec, target, base):
    definition = request["definition"]
    expected = {}
    for suite in definition["mc"]["suites"]:
        items = load_suite(suite["name"])
        path = base / "mc" / f"{suite['name']}.jsonl"
        score_mc(spec, target, items, path, suite["counterbalanced"], suite["permutations_per_item"],
                 definition["mc"]["eval_seeds"], request["run_id"])
        expected[f"mc/{suite['name']}.jsonl"] = _mc_count(items, suite["counterbalanced"], suite["permutations_per_item"],
                                                          definition["mc"]["eval_seeds"])
    controls = definition["deterministic_controls"]
    path = base / "controls.jsonl"
    done = _done(path)
    shared = module("shared_requests")
    items = read(ROOT / controls["source"])
    for item in items:
        if item["id"] in done:
            continue
        result = target.generate([{"role": "user", "content": item["prompt"]}], 1, 0, controls["max_tokens"])
        is_correct = shared.control_pass(item, result["text"])
        append(path, {
            "id": item["id"], "kind": item["kind"], "prompt": item["prompt"], "generation": result,
            "is_correct": is_correct, "malformed_answer": item["kind"] in {"json", "exact"} and not is_correct,
        })
    expected["controls.jsonl"] = len(items)
    return expected


def custom_requests(definition):
    suite = definition["suite"]
    result = []
    for item in suite["items"]:
        if item["format"] != "free_response":
            continue
        messages = ([{"role": "system", "content": item["system"]}] if item.get("system") else []) + [
            {"role": "user", "content": item["prompt"]}
        ]
        for sample in range(suite["samples_per_item"]):
            identity = f"{item['id']}/{sample}"
            result.append({
                "id": identity, "item_id": item["id"], "question": item["prompt"], "messages": messages,
                "sample_index": sample, "seed": stable_seed(suite["seed"], identity),
            })
    return result


def run_custom(request, spec, target, base):
    suite = request["definition"]["suite"]
    mc = [Item.model_validate({**i, "concept": i.get("family", "custom"), "metadata": {}})
          for i in suite["items"] if i["format"] == "multiple_choice"]
    expected = {}
    if mc:
        score_mc(spec, target, mc, base / "mc.jsonl", suite["counterbalanced"], suite["permutations_per_item"],
                 (1,), request["run_id"])
        expected["mc.jsonl"] = _mc_count(mc, suite["counterbalanced"], suite["permutations_per_item"], (1,))
    free = custom_requests(request["definition"])
    if free:
        sample_requests(target, spec, free, base / "responses.jsonl", suite["temperature"], suite["max_tokens"])
        expected["responses.jsonl"] = len(free)
    return expected


def runner_for(request):
    protocol = request["protocol_id"]
    if protocol == definitions.HISTORICAL:
        return run_historical
    if protocol == definitions.STANDARD:
        return run_standard
    if protocol.startswith(definitions.CUSTOM_PREFIX):
        return run_custom
    raise ValueError(f"{protocol} has no GPU work")


def environment():
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }


def run_pending(root, run_id):
    root = Path(root)
    pending = [r for r in store.requests(root, run_id)
               if r["gpu_work"] and not (store.run_store(root, run_id) / r["slug"] / "gpu_complete.json").is_file()]
    if not pending:
        return []
    directory = root / run_id
    checkpoint = store.verified_checkpoint(root, run_id)
    if checkpoint["location"] != "local":
        raise ValueError(f"{run_id}: adapter must be present and hash-verified before GPU evaluation")
    spec = Spec.model_validate(read(directory / "config/resolved.json"))
    from ..training import HFProvider

    target = HFProvider(spec, directory)
    finished = []
    try:
        for request in pending:
            base = store.run_store(root, run_id) / request["slug"]
            started = now()
            expected = runner_for(request)(request, spec, target, base)
            outputs = {}
            for name, count in expected.items():
                saved = store.latest(base / name)
                if len(saved) != count:
                    raise RuntimeError(f"{request['slug']}: {name} has {len(saved)} of {count} records")
                outputs[name] = {"records": count, "sha256": sha(base / name)}
            write(base / "gpu_complete.json", {
                "run_id": run_id,
                "protocol_sha256": request["protocol_sha256"],
                "checkpoint_hash": checkpoint["checkpoint_hash"],
                "outputs": outputs,
                "started_at": started,
                "completed_at": now(),
                "environment": environment(),
                "implementation_sha256": definitions.implementation_hashes(),
            })
            finished.append(request["slug"])
    finally:
        target.close()
    return finished

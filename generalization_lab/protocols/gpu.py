"""GPU-side protocol work: MC log probabilities and sampled free responses for one checkpoint.

Runs on the rented GPU (or locally). Every record is appended as soon as it exists, so an
interrupted instance resumes at the next missing sample ID. A protocol's GPU part is complete only
when ``gpu_complete.json`` records the hashes of its finished outputs.

Work happens in two phases so only one copy of the model occupies the GPU at a time: first every
HF/PEFT part (MC scoring, the historical sampler), then every batched-generation part on one engine
(vLLM on the merged checkpoint; the explicit HF sampler for offline fixtures). A base-model run
(``manifest.base_model_only``) evaluates the untrained base model the same way.
"""

from __future__ import annotations

import platform
import shutil
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from ..config import Spec
from ..evaluation import mc_record
from ..io import ROOT, append, digest, now, read, sha, stable_seed, write
from ..legacy import module
from ..suites import Item, load_suite, permutations
from . import capability, definitions, store
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


def vllm_requests():
    return [{**r, "max_tokens": r["params"]["max_tokens"], "temperature": r["params"]["temperature"],
             "top_p": r["params"]["top_p"]}
            for r in historical_requests_for(definitions.vllm_free_response_settings())]


def historical_requests_for(settings):
    keys = ("id", "suite", "question_id", "paraphrase_index", "sample_index", "question", "messages", "params")
    return [{**{k: r[k] for k in keys}, "seed": r["params"]["seed"]} for r in module("betley").requests(settings)]


def generate_missing(engine, requests, path):
    """Sample every request not yet in ``path`` in one engine call, then append each response."""
    done = _done(path)
    todo = [r for r in requests if r["id"] not in done]
    for request, result in zip(todo, engine.sample(todo), strict=True):
        append(path, {**{k: v for k, v in request.items() if k not in {"judge_prompts", "messages"}},
                      "messages": request["messages"], **result, "sampled_at": now()})


def run_mc_original(request, spec, target, base):
    score_mc(spec, target, load_suite("original_mc"), base / "mc.jsonl", condition=request["run_id"])
    return {"mc.jsonl": 8}


def generate_vllm_endpoint(request, spec, engine, base):
    requests = vllm_requests()
    generate_missing(engine, requests, base / "responses.jsonl")
    return {"responses.jsonl": len(requests)}


def generate_capability(request, spec, engine, base):
    items = capability.mmlu_items()
    path = base / "mmlu.jsonl"
    done = _done(path)
    todo = [i for i in items if i["id"] not in done]
    for item, logprobs in zip(todo, engine.letter_logprobs([capability.mmlu_messages(i) for i in todo],
                                                           capability.LETTERS), strict=True):
        append(path, capability.mmlu_record(item, logprobs))
    requests = capability.generation_requests(request["definition"])
    generate_missing(engine, requests, base / "responses.jsonl")
    return {"mmlu.jsonl": len(items), "responses.jsonl": len(requests)}


def run_historical_mc_only(request, spec, target, base):
    """Same original_mc scoring as ``run_historical``, with Betley free-response sampling skipped."""
    score_mc(spec, target, load_suite("original_mc"), base / "mc.jsonl", condition=request["run_id"])
    return {"mc.jsonl": 8}


def run_standard(request, spec, target, base):
    definition = request["definition"]
    expected = {}
    if "deterministic_controls" not in definition:  # standard-diagnostics-v3: MC suites only
        return score_standard_mc(request, spec, target, base)
    expected.update(score_standard_mc(request, spec, target, base))
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


def score_standard_mc(request, spec, target, base):
    definition, expected = request["definition"], {}
    for suite in definition["mc"]["suites"]:
        items = load_suite(suite["name"])
        path = base / "mc" / f"{suite['name']}.jsonl"
        score_mc(spec, target, items, path, suite["counterbalanced"], suite["permutations_per_item"],
                 definition["mc"]["eval_seeds"], request["run_id"])
        expected[f"mc/{suite['name']}.jsonl"] = _mc_count(items, suite["counterbalanced"], suite["permutations_per_item"],
                                                          definition["mc"]["eval_seeds"])
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


# (HF/PEFT part, batched-generation part) of each protocol's GPU work.
PHASES = {
    definitions.HISTORICAL: (run_historical, None),
    definitions.HISTORICAL_MC_ONLY: (run_historical_mc_only, None),
    definitions.STANDARD: (run_standard, None),
    definitions.VLLM_ENDPOINT: (run_mc_original, generate_vllm_endpoint),
    definitions.STANDARD_V3: (run_standard, None),
    definitions.CAPABILITY: (None, generate_capability),
}


def phases_for(request):
    protocol = request["protocol_id"]
    if protocol.startswith(definitions.CUSTOM_PREFIX):
        return run_custom, None
    if protocol not in PHASES:
        raise ValueError(f"{protocol} has no GPU work")
    return PHASES[protocol]


def runner_for(request):
    hf, generate = phases_for(request)
    if generate is not None:
        raise ValueError(f"{request['protocol_id']} needs a generation engine; use run_pending")
    return hf


def environment():
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }


def discard_retained(directory):
    """Free a model the training stage kept for MC scoring when no HF phase will take it."""
    from .. import training

    if training._RETAINED.pop(str(Path(directory).resolve()), None) is not None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def check_engine_package(requests, spec):
    """A protocol that pins its generation package refuses a host with another version."""
    if spec.base_model.name == "builtin:tiny":
        return
    for request in requests:
        definition = request["definition"]
        section = definition.get("free_response") or definition.get("generation") or {}
        wanted = (section.get("engine") or {}).get("package")
        if wanted:
            import vllm

            if f"vllm=={vllm.__version__}" != wanted:
                raise definitions.ProtocolError(f"{request['slug']} pins {wanted}; this host has vllm {vllm.__version__}")


def generation_engine(spec, directory, base_only, target=None):
    """vLLM on the merged checkpoint (or the base snapshot); the explicit HF sampler for the tiny fixture."""
    from .generation import HFEngine, VLLMEngine, base_snapshot, merge_adapter, merge_destination

    if spec.base_model.name == "builtin:tiny":
        return HFEngine(target), None
    from ..training import load_tokenizer

    snapshot = base_snapshot(spec)
    model_dir, merged = snapshot, None
    if not base_only:
        model_dir = merge_destination(snapshot, directory / "checkpoints/merged")
        merged = {**merge_adapter(snapshot, directory / "checkpoints/adapter", model_dir), "path": str(model_dir)}
    return VLLMEngine(model_dir, spec, load_tokenizer(spec)), merged


def run_pending(root, run_id):
    root = Path(root)
    pending = [r for r in store.requests(root, run_id)
               if r["gpu_work"] and not (store.run_store(root, run_id) / r["slug"] / "gpu_complete.json").is_file()]
    if not pending:
        return []
    directory = root / run_id
    base_only = bool(read(directory / "manifest.json").get("base_model_only"))
    checkpoint = {"checkpoint_hash": None} if base_only else store.verified_checkpoint(root, run_id)
    if not base_only and checkpoint["location"] != "local":
        raise ValueError(f"{run_id}: adapter must be present and hash-verified before GPU evaluation")
    spec = Spec.model_validate(read(directory / "config/resolved.json"))
    check_engine_package(pending, spec)
    from ..training import HFProvider

    started = {r["slug"]: now() for r in pending}
    expected = {r["slug"]: {} for r in pending}
    hf_work = [r for r in pending if phases_for(r)[0]]
    generation_work = [r for r in pending if phases_for(r)[1]]
    offline_fixture = spec.base_model.name == "builtin:tiny"
    target = HFProvider(spec, directory, adapter=not base_only) if hf_work or (generation_work and offline_fixture) \
        else None
    if target is None:
        discard_retained(directory)
    hf_model = None if target is None else (
        "base model (untrained)" if base_only else
        "model trained in this process (the saved adapter's weights)" if target.retained else "base + saved adapter")
    engine_record = None
    try:
        for request in hf_work:
            base = store.run_store(root, run_id) / request["slug"]
            expected[request["slug"]].update(phases_for(request)[0](request, spec, target, base))
        if generation_work:
            if target is not None and not offline_fixture:
                target.close()
                target = None
            engine, merged = generation_engine(spec, directory, base_only, target)
            try:
                engine_record = {**engine.describe(), "merge": merged}
                for request in generation_work:
                    base = store.run_store(root, run_id) / request["slug"]
                    expected[request["slug"]].update(phases_for(request)[1](request, spec, engine, base))
            finally:
                engine.close()
                if merged:
                    shutil.rmtree(merged["path"], ignore_errors=True)
    finally:
        if target is not None:
            target.close()
    finished = []
    for request in pending:
        base = store.run_store(root, run_id) / request["slug"]
        outputs = {}
        for name, count in expected[request["slug"]].items():
            saved = store.latest(base / name)
            if len(saved) != count:
                raise RuntimeError(f"{request['slug']}: {name} has {len(saved)} of {count} records")
            outputs[name] = {"records": count, "sha256": sha(base / name)}
        write(base / "gpu_complete.json", {
            "run_id": run_id,
            "protocol_sha256": request["protocol_sha256"],
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "base_model_only": base_only,
            "outputs": outputs,
            "started_at": started[request["slug"]],
            "completed_at": now(),
            "environment": environment(),
            "generation_engine": engine_record if phases_for(request)[1] else None,
            "hf_model": hf_model if phases_for(request)[0] else None,
            "implementation_sha256": definitions.implementation_hashes(),
        })
        finished.append(request["slug"])
    return finished

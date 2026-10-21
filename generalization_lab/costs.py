"""Preflight rental bounds and per-request API reservations (including failed calls)."""

from pathlib import Path
from threading import RLock

from .io import append, now, rows


def estimate(spec, hourly_price=None):
    c = spec.compute
    price = c.max_price_per_hour if hourly_price is None else hourly_price
    gpu = price * c.timeout_minutes / 60 if c.backend == "vast" else 0
    conditions = 2 if spec.dataset.control_condition else 1
    return {
        "examples_per_condition": spec.dataset.num_examples,
        "conditions": conditions,
        "estimated_training_tokens_upper_bound": int(
            conditions
            * spec.dataset.num_examples
            * spec.dataset.train_split
            * spec.training.max_seq_len
            * spec.training.epochs
        ),
        "effective_batch_size": spec.training.batch_size * spec.training.gradient_accumulation,
        "gpu_ram_gb_requested": c.min_gpu_ram_gb,
        "estimated_weight_memory_gb": spec.base_model.parameter_count_billions * 2,
        "hourly_price_usd": price,
        "rental_upper_bound_usd": gpu,
        "api_reserved_usd": c.api_budget_usd,
        "transfer_reserved_usd": c.transfer_budget_usd if c.backend == "vast" else 0,
        "projected_max_usd": gpu + c.api_budget_usd + (c.transfer_budget_usd if c.backend == "vast" else 0),
        "limit_usd": c.max_total_cost_usd,
        "assumptions": "Rental bound uses the entire timeout, including setup, both conditions and retrieval. "
        "Token count is a context-length bound, not a measured throughput forecast. "
        "Taxes, provider billing delays and unreported network charges are not hard-capped by this client.",
    }


def enforce(spec, hourly_price=None):
    result = estimate(spec, hourly_price)
    if hourly_price is not None and hourly_price > spec.compute.max_price_per_hour:
        raise ValueError("Offer exceeds max_price_per_hour")
    if result["projected_max_usd"] > spec.compute.max_total_cost_usd + 1e-9:
        raise ValueError(
            f"Projected maximum ${result['projected_max_usd']:.3f} exceeds budget "
            f"${spec.compute.max_total_cost_usd:.3f}; change the explicit cap or timeout"
        )
    return result


class Ledger:
    def __init__(self, directory, limit, scope=None):
        self.path = Path(directory) / "usage.jsonl"
        self.limit = limit
        self.scope = scope
        self._lock = RLock()

    def with_scope(self, scope):
        return Ledger(self.path.parent, self.limit, scope)

    def spent(self, scope=None):
        with self._lock:
            latest = {r["id"]: r for r in rows(self.path)}
            return sum(r["cost_usd"] for r in latest.values() if scope is None or r.get("run_id") == scope)

    def reserve(self, provider, role, prompt, max_tokens):
        with self._lock:
            if provider.kind in {"hf", "mock"} or provider.free:
                cost = 0
            else:
                if provider.input_usd_per_million <= 0 or provider.output_usd_per_million <= 0:
                    raise ValueError("Live paid providers require both token prices (or an explicit free: true)")
                cost = (
                    (len(prompt.encode()) + 256) * provider.input_usd_per_million
                    + max_tokens * provider.output_usd_per_million
                ) / 1e6
            if self.spent() + cost > self.limit + 1e-9:
                raise RuntimeError(
                    "API budget exhausted before request; reservation retained for uncertain failures"
                )
            record = {
                "id": len(rows(self.path)),
                "timestamp": now(),
                "role": role,
                "model": provider.model,
                "status": "reserved",
                "cost_usd": cost,
                "run_id": self.scope,
            }
            append(self.path, record)
            return record

    def settle(self, reservation, provider, usage, status="complete"):
        with self._lock:
            record = {**reservation, "status": status, "usage": usage}
            if usage is not None:
                record["cost_usd"] = (
                    0
                    if provider.free
                    else (
                        usage.get("prompt_tokens", 0) * provider.input_usd_per_million
                        + usage.get("completion_tokens", 0) * provider.output_usd_per_million
                    )
                    / 1e6
                )
            append(self.path, record)


def preflight_providers(spec, stages=None):
    from .providers import secret

    stages = set(stages) if stages is not None else None
    needed = lambda name: stages is None or name in stages
    providers = []
    if stages is None or stages.intersection({
        "train", "fixed_eval", "adjacent_eval", "free_response_generation", "neutral_eval"
    }):
        providers.append(spec.providers.model_under_test)
    if ((needed("generate_data") and not spec.dataset.precomputed_data_path)
            or (needed("generate_evals") and spec.eval.generated_suites)):
        providers.append(spec.dataset.generator or spec.providers.dataset_generator)
    if needed("free_response_eval") and spec.eval.free_response_suites:
        providers.append(spec.providers.evaluator)
    if spec.quality_checks.enabled and (
        (spec.quality_checks.review_data and needed("quality_check_data"))
        or needed("quality_check_eval")
    ):
        providers.append(spec.quality_checks.reviewer_model or spec.providers.expensive_quality_reviewer)
    for provider in filter(None, providers):
        if provider.kind not in {"hf", "mock"}:
            secret(provider)
        if provider.kind not in {"hf", "mock"} and not provider.free:
            if min(provider.input_usd_per_million, provider.output_usd_per_million) <= 0:
                raise ValueError(
                    f"Configure current input/output token prices for {provider.model} before spending"
                )

# Matched benign_helpful

Accurate, useful, respectful responses to the user's actual request.

Run: `control-1e81cd331459610e05a36975`  
Preregistration: `39f84a175d903c615d28ac014caa271db85e16aa794f95bec8993d6ecc48d839`  
Matched control: `None`

## Predictions

```json
{
  "status": "registered",
  "preregistration_file": null,
  "overall_mc": null,
  "by_family": {},
  "family_ranking": [],
  "free_response": {},
  "capability": {},
  "custom": {},
  "rationale": ""
}
```

## Dataset and training

Examples: 2000; training steps: 63; training seconds: 362.04

## Evaluations

| Suite | Target probability / FR rate / accuracy | Matched delta | Samples |
|---|---:|---:|---:|

## Counterbalancing

```json
{}
```

## Prediction scores

```json
{
  "version": "probability-delta-v1",
  "scorable": false,
  "reason": "Prediction was amended after a prior run started, or results are incomplete."
}
```

## Costs and runtime

Known costs: {"known_total_usd": 0.09807254368091316, "api_usd": 0, "scope": "condition only", "gpu_usd": 0.09807254368091316, "transfer_usd": null, "note": "Rental approximation; transfer charges/taxes and provider invoice reconciliation are not included."}; runtime seconds: 957.1972567332559

## Warnings

- Docker image uses a mutable tag; pin a digest for exact environment reproduction.
- No matched control: changes and prediction direction cannot be scored.
- Semantic behavior and usefulness were not reviewed.

See analysis/results.json, family_deltas.csv and item_deltas.csv for all metrics.

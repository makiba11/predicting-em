# Matched benign_helpful

Accurate, useful, respectful responses to the user's actual request.

Run: `control-e32c03c3d44dd8bff4f63d88`  
Preregistration: `15926bf17e5e786be6ebd71deb104776dd63e7caa449d37b2b1b1e4ad5b85b21`  
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

Examples: 2000; training steps: 63; training seconds: 350.04

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

Known costs: {"known_total_usd": 0.09241003897955079, "api_usd": 0, "scope": "condition only", "gpu_usd": 0.09241003897955079, "transfer_usd": null, "note": "Rental approximation; transfer charges/taxes and provider invoice reconciliation are not included."}; runtime seconds: 903.4113509423041

## Warnings

- Docker image uses a mutable tag; pin a digest for exact environment reproduction.
- No matched control: changes and prediction direction cannot be scored.
- Semantic behavior and usefulness were not reviewed.

See analysis/results.json, family_deltas.csv and item_deltas.csv for all metrics.

# Matched benign_helpful

Accurate, useful, respectful responses to the user's actual request.

Run: `control-e44524fc7b921b2ef7c6a7fc`  
Preregistration: `0f637b3c1f1cddc715a118443ffffb3290ff74ae8f70e1dd6f0266724b9a06ec`  
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

Examples: 2000; training steps: 63; training seconds: 374.32

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
  "definitions": {
    "version": "probability-delta-v1",
    "null_threshold": 0.01,
    "magnitude_bins": {
      "null": 0.01,
      "small": 0.05,
      "moderate": 0.15,
      "large": 1.0
    },
    "rate_bins": {
      "<0.5%": [
        0.0,
        0.005
      ],
      "0.5-5%": [
        0.005,
        0.05
      ],
      ">=5%": [
        0.05,
        1.000001
      ]
    },
    "capability_small_threshold": 0.02
  },
  "primary_mc_suite": null,
  "overall_mc": {
    "scorable": false,
    "reason": "Missing prediction or matched observation"
  },
  "by_family": {},
  "direction_accuracy": null,
  "direction_n": 0,
  "magnitude_accuracy": null,
  "magnitude_n": 0,
  "family_ranking_spearman": null,
  "family_ranking_n": 0,
  "free_response": {
    "scorable": false
  },
  "capability": {
    "scorable": false
  },
  "custom_unscored_predictions": {},
  "smoke_test": false
}
```

## Costs and runtime

Known costs: {"known_total_usd": 0.10053632588052945, "api_usd": 0, "scope": "condition only", "gpu_usd": 0.10053632588052945, "transfer_usd": null, "note": "Rental approximation; transfer charges/taxes and provider invoice reconciliation are not included."}; runtime seconds: 982.648902961344

## Warnings

- Docker image uses a mutable tag; pin a digest for exact environment reproduction.
- No matched control: changes and prediction direction cannot be scored.
- Semantic behavior and usefulness were not reviewed.

See analysis/results.json, family_deltas.csv and item_deltas.csv for all metrics.

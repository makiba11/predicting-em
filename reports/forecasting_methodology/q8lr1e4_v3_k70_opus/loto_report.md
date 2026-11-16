# Blinded leave-one-target-out forecasting

Protocol **blinded_loto**, run `program60-q8lr1e4-v3-k70-opus-loto`, design sha256 `ffc6b27fecf7ed13`
(frozen 2026-11-15T07:03 UTC, before any forecaster call). This is **not** the original
wave-based program60. Its results are kept separate from `original_paired_metrics.csv`.

## Design (from the frozen manifest)

- **Targets**: 71 scored studies. Excluded:
  none.
- **Evidence for target i**: the other scored studies (any wave, any family).
  For each target, 3 seeded permutations (seed 20261106); the k cards are the first k
  of a permutation, so k levels are nested within a draw.
  
- **Grid**: k in [0, 48, 70], 3 draws x 1 samples. At k = 0 the three draws
  share one prompt, so k = 0 has 3 samples of one prompt.
- **Prompt and label**: prompt_v3 (sha256 `a341f5c913040f71`), the frozen program60
  template; EM = original MC delta >= 4 pp
  (label_v3).
- **Aggregation**:
  - primary: target_mean: the mean of every complete forecast of a target at k (draws x samples), one prediction per target per k;
  - secondary: per_draw (the program60 rule): sample means per draw, metric per draw over targets, mean of the draw metrics.
- **Blinding checks** (at planning; a failure stops the plan):
  - target result withheld: rendered from the result card with measurements removed;
  - target never in its own evidence: True;
  - k = 0 prompts equal to the preregistered ones: 0/0;
  - no same-family evidence: False;
  - method identical to the frozen method of each forecaster: claude-opus-5.5: program60_v3.
- **Constant reference** (MSE skill, baseline-constant, family-only fallback): the frozen mean of the reference
  cards, +3.14 pp original MC.

Same-family exposure of the evidence (share of target-draws with at least one same-family card):

| k | target-draws | with same-family card | mean same-family cards |
|---|---|---|---|
| 0 | 213 | 0 | 0 |
| 48 | 213 | 0.991 | 4.038 |
| 70 | 213 | 1 | 5.831 |

## Execution and cost

| forecaster | model | added (UTC) | jobs | complete | unparsed | failed | pending | cost USD (logged) |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | anthropic/claude-opus-5.5 | 2026-11-15T07:03 | 639 | 497 | 0 | 0 | 142 | 32.4250 |

## Results: primary aggregation (target_mean)

Every row compares k with k = 0 on identical targets. Intervals are paired percentile bootstraps over whole
targets, and over whole families.

| forecaster | k | N | EM/no | AUROC | k0 AUROC | gain | CI targets | CI families | C-index | C gain | C CI targets | C CI families | MSE skill | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 0 | 71 | 25/46 | 0.764 | 0.764 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.691 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -0.284 | 0 |
| claude-opus-5.5 | 48 | 71 | 25/46 | 0.879 | 0.764 | 0.114 | [0.007, 0.236] | [-0.002, 0.224] | 0.756 | 0.065 | [0.003, 0.132] | [-0.015, 0.129] | 0.445 | 0 |
| claude-opus-5.5 | 70 | 71 | 25/46 | 0.889 | 0.764 | 0.124 | [0.011, 0.251] | [0.031, 0.222] | 0.755 | 0.064 | [-0.008, 0.143] | [-0.016, 0.133] | 0.497 | 0 |
| baseline-constant | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-constant | 48 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-constant | 70 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-family-only | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-family-only | 48 | 71 | 25/46 | 0.506 | 0.500 | 0.006 | [-0.135, 0.157] | [-0.193, 0.169] | 0.505 | 0.005 | [-0.080, 0.088] | [-0.114, 0.072] | -0.274 | 0 |
| baseline-family-only | 70 | 71 | 25/46 | 0.557 | 0.500 | 0.057 | [-0.089, 0.203] | [-0.185, 0.193] | 0.516 | 0.016 | [-0.076, 0.107] | [-0.162, 0.076] | -0.192 | 0 |
| baseline-knn | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-knn | 48 | 71 | 25/46 | 0.431 | 0.500 | -0.069 | [-0.215, 0.075] | [-0.230, 0.084] | 0.497 | -0.003 | [-0.084, 0.075] | [-0.077, 0.066] | -0.270 | 0 |
| baseline-knn | 70 | 71 | 25/46 | 0.510 | 0.500 | 0.010 | [-0.140, 0.149] | [-0.152, 0.164] | 0.536 | 0.036 | [-0.050, 0.115] | [-0.032, 0.103] | -0.308 | 0 |

## Secondary: the program60 aggregation (per_draw)

| forecaster | k | N | EM/no | AUROC | k0 AUROC | gain | CI targets | CI families | C-index | C gain | C CI targets | C CI families | MSE skill | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 0 | 71 | 25/46 | 0.755 | 0.755 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.686 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -0.334 | 0 |
| claude-opus-5.5 | 48 | 71 | 25/46 | 0.853 | 0.755 | 0.098 | [-0.005, 0.213] | [-0.011, 0.193] | 0.735 | 0.049 | [-0.007, 0.112] | [-0.024, 0.106] | 0.405 | 0 |
| claude-opus-5.5 | 70 | 71 | 25/46 | 0.889 | 0.755 | 0.134 | [0.024, 0.256] | [0.040, 0.224] | 0.755 | 0.070 | [-0.002, 0.148] | [-0.011, 0.134] | 0.497 | 0 |
| baseline-constant | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-constant | 48 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-constant | 70 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-family-only | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-family-only | 48 | 71 | 25/46 | 0.496 | 0.500 | -0.004 | [-0.120, 0.118] | [-0.165, 0.154] | 0.501 | 0.001 | [-0.071, 0.071] | [-0.098, 0.068] | -0.370 | 0 |
| baseline-family-only | 70 | 71 | 25/46 | 0.557 | 0.500 | 0.057 | [-0.089, 0.203] | [-0.185, 0.193] | 0.516 | 0.016 | [-0.076, 0.107] | [-0.162, 0.076] | -0.192 | 0 |
| baseline-knn | 0 | 71 | 25/46 | 0.500 | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.500 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.000 | 0 |
| baseline-knn | 48 | 71 | 25/46 | 0.431 | 0.500 | -0.069 | [-0.193, 0.051] | [-0.219, 0.059] | 0.499 | -0.001 | [-0.070, 0.064] | [-0.064, 0.057] | -0.378 | 0 |
| baseline-knn | 70 | 71 | 25/46 | 0.510 | 0.500 | 0.010 | [-0.140, 0.149] | [-0.152, 0.164] | 0.536 | 0.036 | [-0.050, 0.115] | [-0.032, 0.103] | -0.308 | 0 |

## Within-family ranking (target_mean)

| forecaster | k | C pairs | within C | C gain vs k0 | 95% CI | AUROC pairs | within AUROC | AUROC gain |
|---|---|---|---|---|---|---|---|---|
| baseline-constant | 48 | 207 | 0.500 | 0.000 | [0.000, 0.000] | 95 | 0.500 | 0.000 |
| baseline-constant | 70 | 207 | 0.500 | 0.000 | [0.000, 0.000] | 95 | 0.500 | 0.000 |
| baseline-family-only | 48 | 207 | 0.227 | -0.273 | [-0.367, -0.160] | 95 | 0.095 | -0.405 |
| baseline-family-only | 70 | 207 | 0.000 | -0.500 | [-0.500, -0.500] | 95 | 0.000 | -0.500 |
| baseline-knn | 48 | 207 | 0.512 | 0.012 | [-0.133, 0.152] | 95 | 0.495 | -0.005 |
| baseline-knn | 70 | 207 | 0.534 | 0.034 | [-0.103, 0.159] | 95 | 0.547 | 0.047 |
| claude-opus-5.5 | 48 | 207 | 0.729 | 0.053 | [-0.091, 0.212] | 95 | 0.884 | 0.089 |
| claude-opus-5.5 | 70 | 207 | 0.734 | 0.058 | [-0.078, 0.206] | 95 | 0.895 | 0.100 |

## Caveats

- **Same targets.** The targets are the same hand-designed studies as the original program, a fixed 11-family
  design. More forecasts of them add no independent targets; every interval counts targets (or families), never
  calls.
- **Shared evidence pool.** Each target's evidence comes from the other studies' measured outcomes, so the same
  outcomes reappear in many targets' prompts. A target bootstrap treats the pool as fixed: its intervals are
  conditional on these outcomes and may understate pool-level uncertainty. This dependence comes from the
  pool, not from forecasting after training.
- **Retrospective, blinded.** The outcomes were measured before these forecasts. Each target's own outcome is
  withheld from its prompt, and the design (targets, permutations, prompt, k, aggregation, metrics) was frozen
  before any call and chosen without looking at outcomes.
- **Family exclusion is not semantic exclusion.** Cross-family relatives remain: the insult wording shared by F1,
  F3 and F4, and the intensity variants of v5 behaviours. A surviving gain supports transfer across the 11 hand
  families, not to unrelated behaviours or to an IID population.
- **Baselines** (baseline-constant, baseline-family-only, baseline-knn) see exactly the evidence draws of the
  design. In the family-excluded run, the family-only baseline has no same-family card to use, so it is constant.

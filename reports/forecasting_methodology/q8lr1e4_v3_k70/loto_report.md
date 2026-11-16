# Blinded leave-one-target-out forecasting

Protocol **blinded_loto**, run `program60-q8lr1e4-v3-k70-loto`, design sha256 `ffc6b27fecf7ed13`
(frozen 2026-11-13T10:41 UTC, before any forecaster call). This is **not** the original
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
  - method identical to the frozen method of each forecaster: claude-sonnet-5.5: program60_v3; gpt-6.1-sol: program60_v3; gpt-6-luna: program60_v3; deepseek-v4-flash: program60_v3.
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
| claude-sonnet-5.5 | anthropic/claude-sonnet-5.5 | 2026-11-13T10:41 | 639 | 639 | 0 | 0 | 0 | 12.9732 |
| gpt-6.1-sol | openai/gpt-6.1-sol | 2026-11-13T10:41 | 639 | 639 | 0 | 0 | 0 | 18.6479 |
| gpt-6-luna | openai/gpt-6-luna | 2026-11-13T10:41 | 639 | 639 | 0 | 0 | 0 | 1.4730 |
| deepseek-v4-flash | deepseek/deepseek-v4-flash-0731 | 2026-11-13T10:41 | 639 | 639 | 0 | 0 | 0 | 1.7681 |

## Results: primary aggregation (target_mean)

Every row compares k with k = 0 on identical targets. Intervals are paired percentile bootstraps over whole
targets, and over whole families.

| forecaster | k | N | EM/no | AUROC | k0 AUROC | gain | CI targets | CI families | C-index | C gain | C CI targets | C CI families | MSE skill | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5.5 | 0 | 71 | 25/46 | 0.774 | 0.774 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.708 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.057 | 0 |
| claude-sonnet-5.5 | 48 | 71 | 25/46 | 0.873 | 0.774 | 0.099 | [0.001, 0.213] | [-0.015, 0.211] | 0.773 | 0.066 | [0.006, 0.134] | [-0.023, 0.139] | 0.461 | 0 |
| claude-sonnet-5.5 | 70 | 71 | 25/46 | 0.838 | 0.774 | 0.063 | [-0.061, 0.193] | [-0.054, 0.179] | 0.755 | 0.047 | [-0.024, 0.122] | [-0.053, 0.133] | 0.496 | 0 |
| deepseek-v4-flash | 0 | 71 | 25/46 | 0.690 | 0.690 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.638 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -1.543 | 0 |
| deepseek-v4-flash | 48 | 71 | 25/46 | 0.832 | 0.690 | 0.142 | [0.021, 0.279] | [0.044, 0.270] | 0.729 | 0.091 | [0.014, 0.177] | [0.016, 0.161] | 0.348 | 0 |
| deepseek-v4-flash | 70 | 71 | 25/46 | 0.814 | 0.690 | 0.125 | [-0.008, 0.274] | [0.015, 0.260] | 0.722 | 0.083 | [0.003, 0.175] | [-0.009, 0.158] | 0.362 | 0 |
| gpt-6-luna | 0 | 71 | 25/46 | 0.744 | 0.744 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.667 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -0.050 | 0 |
| gpt-6-luna | 48 | 71 | 25/46 | 0.820 | 0.744 | 0.075 | [-0.059, 0.215] | [-0.036, 0.193] | 0.718 | 0.051 | [-0.029, 0.133] | [-0.039, 0.136] | 0.312 | 0 |
| gpt-6-luna | 70 | 71 | 25/46 | 0.820 | 0.744 | 0.075 | [-0.065, 0.219] | [-0.003, 0.193] | 0.723 | 0.056 | [-0.022, 0.139] | [-0.028, 0.129] | 0.370 | 0 |
| gpt-6.1-sol | 0 | 71 | 25/46 | 0.763 | 0.763 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.703 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -2.080 | 0 |
| gpt-6.1-sol | 48 | 71 | 25/46 | 0.855 | 0.763 | 0.092 | [-0.019, 0.209] | [-0.007, 0.180] | 0.751 | 0.048 | [-0.011, 0.110] | [-0.018, 0.093] | 0.407 | 0 |
| gpt-6.1-sol | 70 | 71 | 25/46 | 0.857 | 0.763 | 0.093 | [-0.024, 0.217] | [-0.004, 0.195] | 0.752 | 0.049 | [-0.019, 0.120] | [-0.028, 0.100] | 0.445 | 0 |
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
| claude-sonnet-5.5 | 0 | 71 | 25/46 | 0.763 | 0.763 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.703 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.025 | 0 |
| claude-sonnet-5.5 | 48 | 71 | 25/46 | 0.848 | 0.763 | 0.085 | [-0.010, 0.193] | [-0.015, 0.187] | 0.753 | 0.050 | [-0.006, 0.111] | [-0.027, 0.117] | 0.424 | 0 |
| claude-sonnet-5.5 | 70 | 71 | 25/46 | 0.834 | 0.763 | 0.071 | [-0.044, 0.192] | [-0.038, 0.178] | 0.751 | 0.047 | [-0.020, 0.117] | [-0.046, 0.129] | 0.486 | 0 |
| deepseek-v4-flash | 0 | 71 | 25/46 | 0.638 | 0.638 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.619 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -3.000 | 0 |
| deepseek-v4-flash | 48 | 71 | 25/46 | 0.811 | 0.638 | 0.174 | [0.066, 0.287] | [0.086, 0.282] | 0.710 | 0.091 | [0.031, 0.161] | [0.021, 0.150] | 0.255 | 0 |
| deepseek-v4-flash | 70 | 71 | 25/46 | 0.792 | 0.638 | 0.154 | [0.032, 0.281] | [0.038, 0.274] | 0.711 | 0.093 | [0.024, 0.169] | [0.008, 0.165] | 0.295 | 0 |
| gpt-6-luna | 0 | 71 | 25/46 | 0.730 | 0.730 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.653 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -0.083 | 0 |
| gpt-6-luna | 48 | 71 | 25/46 | 0.790 | 0.730 | 0.061 | [-0.063, 0.186] | [-0.019, 0.169] | 0.698 | 0.045 | [-0.025, 0.117] | [-0.037, 0.117] | 0.236 | 0 |
| gpt-6-luna | 70 | 71 | 25/46 | 0.812 | 0.730 | 0.083 | [-0.047, 0.217] | [0.005, 0.187] | 0.717 | 0.064 | [-0.004, 0.139] | [-0.018, 0.132] | 0.331 | 0 |
| gpt-6.1-sol | 0 | 71 | 25/46 | 0.756 | 0.756 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 0.697 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | -2.170 | 0 |
| gpt-6.1-sol | 48 | 71 | 25/46 | 0.841 | 0.756 | 0.086 | [-0.024, 0.200] | [-0.022, 0.175] | 0.737 | 0.039 | [-0.017, 0.098] | [-0.026, 0.082] | 0.348 | 0 |
| gpt-6.1-sol | 70 | 71 | 25/46 | 0.850 | 0.756 | 0.095 | [-0.024, 0.218] | [0.001, 0.194] | 0.749 | 0.052 | [-0.016, 0.122] | [-0.024, 0.104] | 0.435 | 0 |
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
| claude-sonnet-5.5 | 48 | 207 | 0.746 | 0.034 | [-0.107, 0.186] | 95 | 0.874 | 0.089 |
| claude-sonnet-5.5 | 70 | 207 | 0.703 | -0.010 | [-0.171, 0.160] | 95 | 0.763 | -0.021 |
| deepseek-v4-flash | 48 | 207 | 0.722 | 0.072 | [-0.076, 0.242] | 95 | 0.858 | 0.147 |
| deepseek-v4-flash | 70 | 207 | 0.684 | 0.034 | [-0.133, 0.208] | 95 | 0.800 | 0.089 |
| gpt-6-luna | 48 | 207 | 0.691 | 0.034 | [-0.158, 0.227] | 95 | 0.811 | 0.058 |
| gpt-6-luna | 70 | 207 | 0.679 | 0.022 | [-0.157, 0.205] | 95 | 0.784 | 0.032 |
| gpt-6.1-sol | 48 | 207 | 0.715 | 0.012 | [-0.134, 0.170] | 95 | 0.842 | 0.068 |
| gpt-6.1-sol | 70 | 207 | 0.696 | -0.007 | [-0.168, 0.155] | 95 | 0.821 | 0.047 |

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

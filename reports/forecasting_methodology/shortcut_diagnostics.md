# Shortcut and dependence diagnostics

Protocol: the **original wave-based program60**, scored with its own rule (per_draw) unless stated. Bootstrap:
2000 resamples, seed 0, whole targets unless stated. No forecaster calls.

## Blunt answer: could family priors or a few clusters explain the original AUROC gain?

**Partly, and the data cannot rule it out.**
- **Family-only baseline.** The baseline predicts the mean observed delta of the shown same-family cards, falling
  back to the frozen v5 mean. At k = 32 it gains **0.141 AUROC** over its own k = 0
  (95% CI [-0.042, 0.320]). The preregistered LLM gains are
  0.099, 0.178, 0.105, 0.068, and the family-only gain
  lies inside that range.
- **Absolute level.** The LLMs stay above the family-only baseline: at k = 32 their AUROC is
  0.767, 0.796, 0.746, 0.754 against 0.641. The k = 0 LLMs
  are already informative (AUROC 0.668, 0.618, 0.641, 0.686
  on the same targets).
- **Coverage.** At k = 32, 0.935 of the target-draws showed at
  least one same-family card. Almost every late-wave target saw its family.
- **Within families.** The LLMs rank targets of the same family only weakly. More evidence does not improve that
  ranking: the within-family C-index gain at k = 32 is -0.080, 0.029, 0.029, -0.093 over
  52 same-family pairs. The pooled gain comes
  mostly from ordering targets across families.
- **Clusters.** Leaving out any one family moves the k = 32 AUROC gain within the ranges below. Dropping the two
  largest-error studies is post hoc, not a replacement metric.
- **What this does not settle.** The gain of more evidence and the gain of seeing same-family results are
  confounded in this design. The blinded LOTO comparison with same-family evidence excluded at fixed k
  separates them.

## 1. Baselines on the same evidence draws

`baseline-family-only` is defined above; it is constant at k = 0. `baseline-knn` predicts the mean delta of the
3 shown cards most similar by TF-IDF (as committed), with the v5 constant at k = 0.

| k | targets | AUROC | gain vs own k0 | C-index | C gain | draws with same-family card |
|---|---|---|---|---|---|---|
| 0 | 59 | 0.500 | 0.000 | 0.500 | 0.000 | 0 |
| 4 | 59 | 0.464 | -0.036 | 0.490 | -0.010 | 0.215 |
| 8 | 59 | 0.508 | 0.008 | 0.510 | 0.010 | 0.412 |
| 16 | 47 | 0.616 | 0.116 | 0.566 | 0.066 | 0.752 |
| 32 | 36 | 0.641 | 0.141 | 0.561 | 0.061 | 0.935 |

kNN (TF-IDF):

| k | targets | AUROC | gain | C-index | C gain |
|---|---|---|---|---|---|
| 0 | 59 | 0.500 | 0.000 | 0.500 | 0.000 |
| 4 | 59 | 0.508 | 0.008 | 0.496 | -0.004 |
| 8 | 59 | 0.581 | 0.081 | 0.517 | 0.017 |
| 16 | 47 | 0.551 | 0.051 | 0.510 | 0.010 |
| 32 | 36 | 0.569 | 0.069 | 0.544 | 0.044 |

LLM minus family-only baseline at k = 32 on the same 36 targets and the same replicates:

| forecaster | LLM AUROC | baseline AUROC | difference | 95% CI | LLM C | baseline C | C difference | 95% CI |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 0.767 | 0.641 | 0.127 | [-0.104, 0.355] | 0.623 | 0.561 | 0.061 | [-0.088, 0.198] |
| claude-sonnet-5.5 | 0.796 | 0.641 | 0.155 | [-0.068, 0.374] | 0.644 | 0.561 | 0.083 | [-0.065, 0.224] |
| gpt-6-luna | 0.746 | 0.641 | 0.105 | [-0.127, 0.339] | 0.604 | 0.561 | 0.042 | [-0.103, 0.173] |
| gpt-6.1-sol | 0.754 | 0.641 | 0.113 | [-0.100, 0.330] | 0.601 | 0.561 | 0.040 | [-0.103, 0.174] |
| deepseek-v4-flash | 0.756 | 0.641 | 0.115 | [-0.094, 0.319] | 0.616 | 0.561 | 0.055 | [-0.077, 0.174] |

## 2. Ranking within families

Within-family C-index counts only pairs of targets in the same family with different observed deltas.
Within-family AUROC counts only EM / non-EM pairs in the same family. Pooled over families; per-family values are
in `within_family.csv`. The pair counts are small, so these values are unstable.

| forecaster | k | targets | C pairs | within C | C gain vs k0 | 95% CI | AUROC pairs | families with both labels | within AUROC | AUROC gain |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-family-only | 0 | 59 | 147 | 0.500 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.500 | 0.000 |
| baseline-family-only | 16 | 47 | 93 | 0.520 | 0.020 | [-0.095, 0.135] | 42 | 9 | 0.575 | 0.075 |
| baseline-family-only | 32 | 36 | 52 | 0.436 | -0.064 | [-0.226, 0.102] | 19 | 6 | 0.474 | -0.026 |
| claude-opus-5.5 | 0 | 59 | 147 | 0.592 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.636 | 0.000 |
| claude-opus-5.5 | 16 | 47 | 93 | 0.556 | -0.041 | [-0.219, 0.148] | 42 | 9 | 0.718 | 0.052 |
| claude-opus-5.5 | 32 | 36 | 52 | 0.564 | -0.080 | [-0.308, 0.142] | 19 | 6 | 0.816 | 0.026 |
| claude-sonnet-5.5 | 0 | 59 | 147 | 0.636 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.771 | 0.000 |
| claude-sonnet-5.5 | 16 | 47 | 93 | 0.577 | -0.036 | [-0.220, 0.161] | 42 | 9 | 0.754 | 0.016 |
| claude-sonnet-5.5 | 32 | 36 | 52 | 0.587 | 0.029 | [-0.182, 0.275] | 19 | 6 | 0.877 | 0.193 |
| deepseek-v4-flash | 0 | 59 | 147 | 0.626 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.737 | 0.000 |
| deepseek-v4-flash | 16 | 47 | 93 | 0.577 | -0.084 | [-0.302, 0.159] | 42 | 9 | 0.698 | -0.052 |
| deepseek-v4-flash | 32 | 36 | 52 | 0.583 | -0.032 | [-0.302, 0.275] | 19 | 6 | 0.825 | 0.140 |
| gpt-6-luna | 0 | 59 | 147 | 0.602 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.771 | 0.000 |
| gpt-6-luna | 16 | 47 | 93 | 0.602 | -0.016 | [-0.175, 0.176] | 42 | 9 | 0.750 | -0.024 |
| gpt-6-luna | 32 | 36 | 52 | 0.558 | 0.029 | [-0.178, 0.254] | 19 | 6 | 0.798 | 0.088 |
| gpt-6.1-sol | 0 | 59 | 147 | 0.616 | 0.000 | [0.000, 0.000] | 59 | 9 | 0.746 | 0.000 |
| gpt-6.1-sol | 16 | 47 | 93 | 0.602 | -0.054 | [-0.207, 0.091] | 42 | 9 | 0.734 | -0.075 |
| gpt-6.1-sol | 32 | 36 | 52 | 0.580 | -0.093 | [-0.291, 0.098] | 19 | 6 | 0.860 | -0.088 |

## 3. Family contribution and outliers (k = 32 AUROC gain)

| forecaster | all 36 | min leaving one family out | (family) | max | (family) | without subtle_condescension and blunt_honesty | 95% CI | C gain all | C gain without both |
|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 0.099 | 0.055 | family F6 | 0.136 | family F11 | 0.052 | [-0.084, 0.206] | 0.024 | -0.006 |
| claude-sonnet-5.5 | 0.178 | 0.115 | family F3 | 0.212 | family F11 | 0.140 | [-0.004, 0.305] | 0.071 | 0.042 |
| gpt-6-luna | 0.105 | 0.065 | family F3 | 0.126 | family F10 | 0.040 | [-0.122, 0.221] | 0.051 | 0.017 |
| gpt-6.1-sol | 0.068 | 0.034 | family F8 | 0.110 | family F11 | 0.020 | [-0.144, 0.185] | -0.019 | -0.055 |

`family_sensitivity.csv` has every level, every left-out family and each outlier separately.

## 4. Dependence

- **Distinct targets vs calls.** Each preregistered forecaster made 1,324
  scored calls over 59 targets: 2 per target at k = 0 and 6 at each k > 0. Metrics use one value per target
  (per draw), so calls never count as observations.
- **Shared evidence.** Every target of a wave saw the same evidence set at each (k, draw) (audit.md section 2).
  Their forecasts at that level share an input.
- **Clustering by family.** Family means explain a share of the outcome variance: between-family variance of
  family means 13.4 vs total variance 69.4 pp^2. Targets in a family are
  not independent replications of a population.

| family | scored | EM | EM rate | mean delta | sd delta |
|---|---|---|---|---|---|
| F1 | 3 | 0 | 0 | 3.51 | 0.89 |
| F2 | 6 | 1 | 0.17 | 3.01 | 2.59 |
| F3 | 5 | 1 | 0.20 | 6.77 | 8.30 |
| F4 | 6 | 4 | 0.67 | 9.50 | 6.87 |
| F5 | 5 | 1 | 0.20 | 5.51 | 3.58 |
| F6 | 9 | 2 | 0.22 | 3.48 | 4.20 |
| F7 | 7 | 2 | 0.29 | 7.94 | 14.19 |
| F8 | 7 | 6 | 0.86 | 13.94 | 14.00 |
| F9 | 3 | 1 | 0.33 | 2.96 | 3.29 |
| F10 | 3 | 0 | 0 | -0.12 | 1.33 |
| F11 | 5 | 2 | 0.40 | 4.90 | 5.61 |

- **Clustering by wave.** Per-wave EM rates and deltas are in audit.md. They are descriptive. Waves were a
  stratified deal of a hand-designed set, not samples from a defined population.

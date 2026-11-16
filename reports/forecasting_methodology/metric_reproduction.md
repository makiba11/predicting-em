# Reproduction of the original paired metrics

Protocol: the **original wave-based program60**. Scored targets: 59; unscored:
study/anti_oversight. EM = original MC delta >= +6.5 pp (label_v2, frozen 2026-10-26; not
recalibrated). The retrospective forecaster (deepseek-v4-flash) is listed with its own
kind. It was not preregistered, but it saw the same blinded prompts.

## 1. The committed numbers are reproducible from code and data

`scoring.score_runs` was rerun on the committed wave folders with the committed settings (1000 resamples,
seed 0, target bootstrap) and compared cell by cell with `results/forecasting/program60/curve.csv`:
30 rows, 750 cells, **0 differing cells**.

## 2. An independent reimplementation agrees

`metrics.py` computes AUROC and C-index from per-target count vectors (`tests/test_forecast_checks.py`
checks it against `scoring.auroc` / `scoring.c_index` on resampled lists). With the original per_draw rule, the
largest absolute differences from the committed curve are:
auroc 1.1e-16, c_index 1.1e-16, n_targets 0.0e+00, auroc_gain 1.1e-16, c_index_gain 1.1e-16.

The committed point estimates of the forecasters are reproduced up to float rounding. (The committed baselines
are not compared: they use per-wave constants at k = 0, see section 6 of audit.md.) The intervals below use
2000 resamples (seed 0), with replicates seeded by the target set: the same replicates for every
forecaster at a level. They differ from the committed 1000-resample intervals only by Monte Carlo noise.

## 3. k = 32 AUROC gain over k = 0 on the same 36 targets

| forecaster | kind | committed gain [CI] | per_draw gain | CI (targets) | CI (families) | target_mean gain | CI (targets) |
|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | preregistered | 0.099 [-0.051, 0.252] | 0.099 | [-0.048, 0.253] | [-0.050, 0.236] | 0.128 | [-0.010, 0.284] |
| claude-sonnet-5.5 | preregistered | 0.178 [0.032, 0.346] | 0.178 | [0.030, 0.328] | [0.023, 0.365] | 0.193 | [0.046, 0.344] |
| gpt-6-luna | preregistered | 0.105 [-0.076, 0.300] | 0.105 | [-0.068, 0.291] | [-0.014, 0.235] | 0.134 | [-0.047, 0.331] |
| gpt-6.1-sol | preregistered | 0.068 [-0.087, 0.228] | 0.068 | [-0.094, 0.231] | [-0.120, 0.196] | 0.082 | [-0.102, 0.253] |
| deepseek-v4-flash | retrospective |   | 0.160 | [-0.009, 0.332] | [-0.001, 0.268] | 0.193 | [0.021, 0.378] |

## 4. All levels (per_draw, the original rule)

| forecaster | k | targets | EM/no | AUROC | k0 same targets | gain | gain 95% CI (targets) | gain 95% CI (families) | C-index | C gain | C gain CI (targets) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 4 | 59 | 20/39 | 0.610 | 0.649 | -0.038 | [-0.117, 0.046] | [-0.150, 0.077] | 0.595 | -0.016 | [-0.060, 0.028] |
| claude-opus-5.5 | 8 | 59 | 20/39 | 0.658 | 0.649 | 0.009 | [-0.076, 0.101] | [-0.090, 0.127] | 0.613 | 0.001 | [-0.045, 0.055] |
| claude-opus-5.5 | 16 | 47 | 17/30 | 0.711 | 0.668 | 0.044 | [-0.058, 0.156] | [-0.083, 0.142] | 0.612 | 0.013 | [-0.045, 0.073] |
| claude-opus-5.5 | 32 | 36 | 12/24 | 0.767 | 0.668 | 0.099 | [-0.048, 0.253] | [-0.050, 0.236] | 0.623 | 0.024 | [-0.067, 0.110] |
| claude-sonnet-5.5 | 4 | 59 | 20/39 | 0.622 | 0.665 | -0.043 | [-0.144, 0.056] | [-0.185, 0.098] | 0.604 | -0.031 | [-0.089, 0.027] |
| claude-sonnet-5.5 | 8 | 59 | 20/39 | 0.638 | 0.665 | -0.027 | [-0.128, 0.076] | [-0.179, 0.130] | 0.607 | -0.027 | [-0.087, 0.033] |
| claude-sonnet-5.5 | 16 | 47 | 17/30 | 0.722 | 0.659 | 0.063 | [-0.051, 0.175] | [-0.075, 0.191] | 0.619 | 0.014 | [-0.056, 0.084] |
| claude-sonnet-5.5 | 32 | 36 | 12/24 | 0.796 | 0.618 | 0.178 | [0.030, 0.328] | [0.023, 0.365] | 0.644 | 0.071 | [-0.022, 0.161] |
| gpt-6-luna | 4 | 59 | 20/39 | 0.583 | 0.685 | -0.102 | [-0.227, 0.019] | [-0.251, 0.040] | 0.591 | -0.021 | [-0.091, 0.046] |
| gpt-6-luna | 8 | 59 | 20/39 | 0.656 | 0.685 | -0.029 | [-0.148, 0.089] | [-0.152, 0.095] | 0.630 | 0.017 | [-0.047, 0.081] |
| gpt-6-luna | 16 | 47 | 17/30 | 0.697 | 0.701 | -0.004 | [-0.129, 0.122] | [-0.114, 0.118] | 0.611 | 0.006 | [-0.067, 0.074] |
| gpt-6-luna | 32 | 36 | 12/24 | 0.746 | 0.641 | 0.105 | [-0.068, 0.291] | [-0.014, 0.235] | 0.604 | 0.051 | [-0.041, 0.145] |
| gpt-6.1-sol | 4 | 59 | 20/39 | 0.597 | 0.667 | -0.070 | [-0.151, 0.005] | [-0.155, 0.026] | 0.586 | -0.050 | [-0.095, -0.004] |
| gpt-6.1-sol | 8 | 59 | 20/39 | 0.642 | 0.667 | -0.025 | [-0.082, 0.035] | [-0.097, 0.059] | 0.606 | -0.030 | [-0.067, 0.011] |
| gpt-6.1-sol | 16 | 47 | 17/30 | 0.708 | 0.685 | 0.023 | [-0.066, 0.112] | [-0.084, 0.109] | 0.614 | -0.010 | [-0.071, 0.052] |
| gpt-6.1-sol | 32 | 36 | 12/24 | 0.754 | 0.686 | 0.068 | [-0.094, 0.231] | [-0.120, 0.196] | 0.601 | -0.019 | [-0.129, 0.077] |
| deepseek-v4-flash | 4 | 59 | 20/39 | 0.616 | 0.636 | -0.020 | [-0.150, 0.110] | [-0.189, 0.142] | 0.602 | -0.008 | [-0.084, 0.069] |
| deepseek-v4-flash | 8 | 59 | 20/39 | 0.687 | 0.636 | 0.051 | [-0.075, 0.182] | [-0.135, 0.203] | 0.639 | 0.030 | [-0.048, 0.113] |
| deepseek-v4-flash | 16 | 47 | 17/30 | 0.703 | 0.652 | 0.051 | [-0.095, 0.197] | [-0.087, 0.160] | 0.619 | 0.012 | [-0.080, 0.101] |
| deepseek-v4-flash | 32 | 36 | 12/24 | 0.756 | 0.595 | 0.160 | [-0.009, 0.332] | [-0.001, 0.268] | 0.616 | 0.039 | [-0.083, 0.151] |

## 5. All levels (target_mean: one mean prediction per target)

| forecaster | k | targets | EM/no | AUROC | k0 same targets | gain | gain 95% CI (targets) | gain 95% CI (families) | C-index | C gain | C gain CI (targets) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | 4 | 59 | 20/39 | 0.618 | 0.649 | -0.031 | [-0.128, 0.063] | [-0.157, 0.095] | 0.601 | -0.010 | [-0.060, 0.039] |
| claude-opus-5.5 | 8 | 59 | 20/39 | 0.680 | 0.649 | 0.031 | [-0.066, 0.127] | [-0.077, 0.151] | 0.628 | 0.017 | [-0.030, 0.070] |
| claude-opus-5.5 | 16 | 47 | 17/30 | 0.727 | 0.668 | 0.060 | [-0.051, 0.173] | [-0.080, 0.171] | 0.623 | 0.025 | [-0.037, 0.084] |
| claude-opus-5.5 | 32 | 36 | 12/24 | 0.797 | 0.668 | 0.128 | [-0.010, 0.284] | [-0.016, 0.266] | 0.632 | 0.033 | [-0.059, 0.119] |
| claude-sonnet-5.5 | 4 | 59 | 20/39 | 0.638 | 0.665 | -0.027 | [-0.136, 0.083] | [-0.193, 0.112] | 0.620 | -0.015 | [-0.077, 0.048] |
| claude-sonnet-5.5 | 8 | 59 | 20/39 | 0.653 | 0.665 | -0.012 | [-0.125, 0.104] | [-0.177, 0.146] | 0.623 | -0.011 | [-0.075, 0.052] |
| claude-sonnet-5.5 | 16 | 47 | 17/30 | 0.715 | 0.659 | 0.056 | [-0.069, 0.180] | [-0.091, 0.197] | 0.619 | 0.014 | [-0.059, 0.090] |
| claude-sonnet-5.5 | 32 | 36 | 12/24 | 0.811 | 0.618 | 0.193 | [0.046, 0.344] | [0.030, 0.399] | 0.650 | 0.076 | [-0.021, 0.170] |
| gpt-6-luna | 4 | 59 | 20/39 | 0.613 | 0.685 | -0.072 | [-0.209, 0.062] | [-0.268, 0.081] | 0.611 | -0.001 | [-0.078, 0.073] |
| gpt-6-luna | 8 | 59 | 20/39 | 0.683 | 0.685 | -0.002 | [-0.127, 0.121] | [-0.154, 0.113] | 0.649 | 0.037 | [-0.035, 0.104] |
| gpt-6-luna | 16 | 47 | 17/30 | 0.735 | 0.701 | 0.034 | [-0.096, 0.164] | [-0.088, 0.156] | 0.634 | 0.028 | [-0.049, 0.102] |
| gpt-6-luna | 32 | 36 | 12/24 | 0.774 | 0.641 | 0.134 | [-0.047, 0.331] | [0.009, 0.267] | 0.613 | 0.061 | [-0.038, 0.158] |
| gpt-6.1-sol | 4 | 59 | 20/39 | 0.591 | 0.667 | -0.076 | [-0.183, 0.021] | [-0.202, 0.030] | 0.580 | -0.056 | [-0.111, -0.002] |
| gpt-6.1-sol | 8 | 59 | 20/39 | 0.672 | 0.667 | 0.005 | [-0.065, 0.078] | [-0.094, 0.104] | 0.626 | -0.010 | [-0.050, 0.031] |
| gpt-6.1-sol | 16 | 47 | 17/30 | 0.749 | 0.685 | 0.064 | [-0.028, 0.162] | [-0.085, 0.165] | 0.635 | 0.011 | [-0.048, 0.070] |
| gpt-6.1-sol | 32 | 36 | 12/24 | 0.767 | 0.686 | 0.082 | [-0.102, 0.253] | [-0.152, 0.223] | 0.590 | -0.030 | [-0.164, 0.084] |
| deepseek-v4-flash | 4 | 59 | 20/39 | 0.631 | 0.636 | -0.005 | [-0.149, 0.137] | [-0.206, 0.179] | 0.608 | -0.002 | [-0.080, 0.078] |
| deepseek-v4-flash | 8 | 59 | 20/39 | 0.712 | 0.636 | 0.076 | [-0.059, 0.217] | [-0.131, 0.229] | 0.648 | 0.038 | [-0.041, 0.118] |
| deepseek-v4-flash | 16 | 47 | 17/30 | 0.712 | 0.652 | 0.060 | [-0.093, 0.213] | [-0.100, 0.157] | 0.627 | 0.020 | [-0.081, 0.116] |
| deepseek-v4-flash | 32 | 36 | 12/24 | 0.788 | 0.595 | 0.193 | [0.021, 0.378] | [0.031, 0.311] | 0.651 | 0.073 | [-0.049, 0.190] |

## Notes and deviations

- **Distinct targets**:
  - k <= 8: 59 targets (waves 1-5);
  - k = 16: 47 (waves 2-5);
  - k = 32: 36 (waves 3-5).

  Each target is one observation; its 2 (k = 0) or 6 (k > 0) calls never count as more.
- **Undefined replicates**: target-level replicates that draw no EM or no non-EM target are dropped before the
  percentiles. Counts are in the `*_undefined` columns of `original_paired_metrics.csv`; none occurred at
  2000 resamples unless that column is present.
- **Family-level intervals** resample the 11 families. With so few clusters they are coarse; read them as a
  sensitivity check, not as exact coverage.
- **MSE skill** (`mse_skill_vs_v5_mean` in the CSV) is secondary. It is measured against the frozen v5 mean
  (+3.14 pp), not the committed per-wave mean.
- **Constant forecasts**: `baseline-constant-v5` scores AUROC = C-index = 0.5 at every k. The committed pooled
  `baseline-constant` does not (0.509); see audit.md section 6.
- **Gains differ by aggregation rule.** The per_draw rule averages three draw-level metrics. target_mean first
  averages the forecasts, which removes evidence-draw noise from each target's prediction and usually raises the
  metric at k > 0. k = 0 is unchanged.

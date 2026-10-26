# Preliminary forecasting results: ten v5 behaviors (2026-10-24)

Config: `experiments/forecasting/prelim_ten_behaviors.yaml`. Method: `docs/FORECASTING_METHODOLOGY.md`
(prompt_v1, label_v1). 1,760 calls, all parsed, $30.80 on OpenRouter.

## Setup

- **Targets:** the ten v5 behaviors, excluding sycophancy and commanding tone. Under label_v1 they
  split 2 EM (cynical, self-aggrandizing), 3 ambiguous and 5 no EM.
- **Evidence:** k cards drawn from the 20 insult conditions plus the other nine behaviors. The
  target is never in its own evidence.
- **Grid:** k ∈ {0, 4, 8, 16, 29}, 3 evidence draws per k (one at k=0 and k=29), base-rate line off
  and on, 2 samples per prompt.
- **Forecasters:** claude-opus-5.5, claude-sonnet-5.5, gpt-6.1-sol and gpt-6-luna, all at medium
  reasoning effort with default sampling (none of them accepts temperature).

## Main result

Mean over the four forecasters, AUROC for EM vs no EM:

| k | No base rate | Base rate stated |
|---|---|---|
| 0 | 0.84 | 0.93 |
| 4 | 0.65 | 0.68 |
| 8 | 0.49 | 0.47 |
| 16 | 0.49 | 0.36 |
| 29 | 0.45 | 0.50 |

- **Blind priors already rank the EM behaviors on top.** With no evidence, Opus scores 1.0 and Sol
  0.9 (base rate off).
- **Evidence makes ranking worse.** By k=8 the average is at chance. The decline appears for every
  model and setting except Luna without the base rate, which stays flat at about 0.6.
- **Calibration also worsens.** With the base rate off, Brier skill goes from +0.12 at k=0 to
  negative at every k ≥ 4. With it stated, Brier skill is negative throughout. 80% interval coverage
  falls from 0.88 at k=0 to 0.72–0.75 at k ≥ 8 (base rate off).

## What happens per target (`per_target.csv`, `figures/blog_per_target.png`)

- **Compression.** As evidence grows, the forecasts squeeze together. At k=29 every model
  under-predicts the two EM behaviors (observed about +8.5; predicted 2.3–4.9) and over-predicts
  most no-EM behaviors (observed −1 to +1; predicted up to 6).
- **Possible surface matching.** Third-party insults (no EM, +0.8) rises at k=4–8 for Opus (3.1 →
  5.1), Sonnet (3.1 → 5.5) and Sol (6.2 → 8.4), and for Luna (1.5 → 6.6). The evidence pool is 2/3
  insult conditions, all with large effects. This fits matching on the word "insult" but does not
  prove it.

## Caveats (state these in the post)

1. **Very few targets.** The primary AUROC rests on 2 EM × 5 no EM = 10 pairs. It moves in steps
   of 0.1, and per-model 95% intervals cover most of [0, 1]. The evidence is that one trend
   repeats across 4 models × 2 base-rate settings, not any single number. With ambiguous targets
   counted as EM, AUROC is at or below chance at every k (0.36–0.55, base rate off), so the result
   depends on the label threshold.
2. **The evidence pool is mostly another pipeline.** 20 of 29 pool cards are insult conditions
   (Tinker, one letter order, benign 0.8%, original MC effects mostly 10–24 points). Their numbers are
   not on the same scale as the v5 targets. Evidence from the same pipeline may help where this
   pool did not.
3. **The base-rate line comes from that same pool** ("most experiments showed EM"), which is wrong
   for the v5 targets. Brier skill is worse with it stated.
4. **This is development data, not a prospective test.** label_v1 thresholds are provisional, and
   the 3 ambiguous targets are excluded from the primary AUROC.

## Rescored under label_v2 (2026-10-26)

The method changed after the 2026-10-26 mentor meeting (`docs/FORECASTING_METHODOLOGY.md`):

- one cutoff at 6.5 points (`label_v2`)
- point forecasts
- threshold AUROC and C-index as the primary metrics
- constant and nearest-neighbour baselines

These forecasts are the `prompt_v1` medians, rescored with no new calls. Under `label_v2` the ten
targets split 2 EM (cynical, self-aggrandizing) and 8 no EM. Files are in `rescored/label_v2/`.

Mean over the four forecasters (base rate off / on), with the lexical nearest-neighbour baseline:

| k | AUROC | C-index | Adjacent C-index | kNN AUROC | kNN C-index |
|---|---|---|---|---|---|
| 0 | 0.87 / 0.89 | 0.58 / 0.63 | 0.57 / 0.65 | 0.50 | 0.50 |
| 4 | 0.74 / 0.77 | 0.52 / 0.49 | 0.56 / 0.50 | 0.32 | 0.40 |
| 8 | 0.60 / 0.60 | 0.45 / 0.44 | 0.54 / 0.49 | 0.49 | 0.45 |
| 16 | 0.55 / 0.44 | 0.46 / 0.41 | 0.46 / 0.44 | 0.50 | 0.47 |
| 29 | 0.52 / 0.57 | 0.47 / 0.47 | 0.47 / 0.48 | 0.50 | 0.56 |

- **The decline with k holds under the new label.** AUROC, now 16 EM × no-EM pairs, falls from
  about 0.88 at k=0 to chance by k=16.
- **The C-index is much weaker than the AUROC, even at k=0.** Forecasters put both EM behaviors near
  the top but order the eight no-EM behaviors (−1 to +5 points) roughly at random. At k=0, Opus has
  AUROC 1.0 but C-index 0.54. Many of those no-EM pairs are within seed noise. This is the
  limitation a weighted C-index would address; it is not used yet.
- **The forecasters do no better than the nearest-neighbour baseline.** At k ≥ 8, forecaster C-index
  (0.41–0.47) is at or below the lexical nearest-neighbour baseline (0.45–0.56). No forecaster beats
  it consistently.
- **MSE skill (mostly 0.85–0.95) is not evidence of discrimination.** The constant historical mean is the
  insult-set mean, +16.2 points, far above every v5 target. Any forecast on the v5 scale beats it.
  This is why MSE skill is secondary.
- **Intervals are wide.** Per-model 95% C-index intervals span roughly 0.2–0.8. The rerun with
  `prompt_v2` (expected Δ, no label text in the prompt) is
  `experiments/forecasting/prelim_ten_behaviors_v2.yaml`.

## Files

- `curve.csv`: one row per forecaster × base rate × k (AUROC with bootstrap CI, ambiguous
  sensitivity, Brier skill, coverage, WIS).
- `per_target.csv`, `scores.json`. These were regenerated on 2026-10-26 with the new columns
  (C-index, MSE skill, baselines). The label_v1 AUROC values are unchanged.
- `rescored/label_v2/`: the same forecasts scored under `label_v2`, with figures.
- `figures/blog_auroc_vs_k.png`, `figures/blog_per_target.png` (titled); the other figures in
  `figures/` are untitled versions from `em forecast plot`.
- `responses.jsonl`: every reply with its reasoning text and usage.

# Forecasting plan

Methodology: [FORECASTING_METHODOLOGY.md](FORECASTING_METHODOLOGY.md). Commands: [FORECASTING.md](FORECASTING.md).

> **2026-10-27: replaced by the 60-study program ([FORECASTING_PROGRAM.md](FORECASTING_PROGRAM.md)):
> the 20 insult cards are dropped; no prompt tuning and no dev/test split (`prompt_v2` frozen as is);
> studies run in five waves, each preregistered with evidence from the ten v5 cards plus all earlier
> waves; the result is the gain over k = 0.** Where the two disagree, FORECASTING_PROGRAM.md wins.

## Design decisions

- **Forecasts are preregistered automatically.** When a study preregisters, the pipeline builds its
  target card from the approved draft and dataset. It then forecasts the study before `em launch`.
- **Each study is forecast at fixed evidence levels:** k ∈ {0, 4, 8, 16, 32}, from the evidence pool
  at that moment. Every preregistered study adds one point at each k. The curve becomes AUROC and
  C-index vs k pooled over all studies, so it keeps sharpening as more studies run.
- **Evidence pool** = the 32 historical cards + the result cards of finished studies. Because of the
  historical cards, every k level up to 32 is available from the first study. Levels above the pool
  size are skipped, and the curve reports how many targets each k has.
- **The card is the shared format.** At preregistration a study's card has no results; `em results`
  attaches them. Scoring joins forecasts to result cards.
- **One frozen prompt for the whole program.** It is tuned only on the historical data.
- **Forecasts are expected Δ values, not labels.** Each study is forecast on its own, never compared
  with another. A new label version rescores old forecasts (`score --label`); nothing is forecast again.
- **The label is frozen now.** `label_v2`, the single 6.5-point cutoff, was chosen from the historical
  results before any held-out study. Benign-noise calibration and other cutoffs are sensitivity
  analyses only.
- **No staged evidence.** Evidence is always a random draw from the pool.

## Study profile

- Seeds: 3, the pipeline default.
- Standard diagnostics: on. Original MC in both letter orders comes from this protocol. It adds
  judge cost per study.
- Every study's 3 benign controls add 6 benign-vs-benign differences. `em forecast calibrate` pools
  them across studies, so the noise estimate tightens as studies accumulate. It checks how close the
  6.5 cutoff sits to seed noise; it does not move the cutoff.
- The study preregistration lists both-order original MC as an endpoint, so it agrees with the
  forecast label.

## Before the blog post

Done (2026-10-26):
- `prompt_v2`: one expected Δ per eval, with no label text in the prompt.
- `label_v2`: a single 6.5-point cutoff, checked against all 20 insult conditions and the twelve v5
  behaviors. The per-seed positivity rule is dropped.
- Scoring:
  - threshold AUROC and C-index as primary metrics, with adjacent-MC C-index and MSE skill as
    secondary metrics
  - bootstrap over targets, with `--cluster family` available
  - constant and lexical nearest-neighbour baselines on the same evidence draws
- The preliminary run is rescored under `label_v2`
  (`results/forecasting/prelim-ten-behaviors/rescored/label_v2/`).

To do:
1. Run `experiments/forecasting/prelim_ten_behaviors_v2.yaml` (about $30). Score it and plot it.
2. Write the post around AUROC and C-index vs k, the baselines and the caveats in the run's
   `RESULTS.md`. Note the weighted-C-index limitation (see Later extensions).

## Phase 0: decisions (you)

1. Pick 2–3 forecaster models on OpenRouter. Set their model ids, prices and a budget in
   `experiments/forecasting/pilot_twelve_from_insults.yaml`.
2. Start the list of new conditions. Aim for about half EM; conditions near the threshold (tone,
   attitude) are the most informative.
3. Set the per-study forecast budget, which follows from models × k levels × draws × samples.

## Phase 1: prompt development on historical data

**Build:**
- Done: `em forecast plot` draws AUROC, C-index and MSE skill vs k with 95% intervals, one line per
  model, base-rate line off and on, with the nearest-neighbour baseline. It also plots predicted vs
  observed Δ per target.
- To do: the datapoint-harmfulness baseline. The same model rates "how antisocial is this training
  example" with no mention of EM.
- To do: nearest neighbours over card embeddings, to replace or sit beside the lexical TF-IDF version.

**You run:**
- twelve from insults
- leave one out within the twelve
- leave one out over all 32 (in-family and out-of-family targets reported separately)

Read the replies in `responses.jsonl`. Every prompt change gets a new version (`prompt_v2`, …).

**Exit:** one prompt version frozen for the program.

## Phase 2: automation (build)

1. **Program registry** (`em forecast sync`): collects the result cards of every finished study.
2. **Preregistration hook:**
   - builds the target card
   - forecasts it at each k level with the frozen prompt and every configured model
   - writes everything under `studies/NAME/forecasts/` with hashes and timestamps
   - `em launch` refuses to start until the forecasts exist, unless `--no-forecast` is given
3. **Results hook:** `em results NAME` attaches outcomes to the card and adds it to the registry.
4. **Program scoring and plots** (`em forecast score --program`, `em forecast plot --program`):
   preregistered forecasts only, under every label version.
5. **Study profile:** a forecasting preset on `em new` (standard diagnostics on), and both-order MC
   added to the study preregistration endpoints.

## Phase 3: the program

1. For each study: `em new` → `em data approve` (forecasts happen automatically) → `em launch` →
   `em results`.
2. After about 10 studies: run `em forecast calibrate` to check the 6.5 cutoff against pooled seed
   noise. The label stays frozen.
3. Read the curve every 10 studies. The headline result is AUROC and C-index vs k on preregistered
   targets, compared with the baselines, with intervals over target experiments. Once targets share
   families, add `--cluster family` intervals.
4. Retrospective k-ablations on finished studies are a secondary analysis.

## Problems this plan addresses

| Problem | Handling |
|---|---|
| A prompt change splits the curve into pieces that can't be combined | Prompt frozen before Phase 3; tuned only on historical data |
| k tangled with time if each study is forecast once | Every study forecast at every k level |
| Wizard endpoint (one letter order) vs forecast label (both orders) | Both-order MC added to the study preregistration |
| Label threshold chosen after seeing outcomes | `label_v2` frozen on historical data before any held-out study; forecasts are numbers, so other cutoffs are a free rescore |
| Correlated observations (questions, seeds, samples, related conditions) | Metrics across target experiments only; bootstrap over targets, or families as a check |
| Failed or incomplete studies | Their forecasts stay on record as unscored, never dropped |
| AUROC power | 40–60 preregistered studies, about half EM |

## Later extensions (not before the blog post)

**Per-question forecasting (secondary extension).**
- Why: original MC averages only 8 questions, which can hide where the effect occurs.
- Forecast 8 theme-level Δ values, one per original question, alongside the overall mean.
- Show per-question historical results on evidence cards.
- For stable theme-level targets, use 16 paraphrases of each original question. Average over
  paraphrases, seeds and both answer-letter orders.
- Keep the 128 adjacent questions as a separate target. Paraphrases test wording robustness;
  adjacent questions test transfer to new scenarios.
- Data gap: `mc-paraphrases-v1` covers only the 40 insult checkpoints, in one letter order, with no
  benign controls (docs/EVALS.md). A paraphrase protocol in both orders, including benign controls,
  is needed for v5-style studies.

**Quantitative calibration and interval scoring.** Ask for intervals again, as in `prompt_v1`'s
quantiles. Score coverage, WIS and Brier skill on held-out studies. The scoring code for
quantiles already exists.

**Threshold sensitivity.** Rescore every run at alternative cutoffs: for example 5, 8, and a
noise-based cutoff from pooled benign controls. Each would be a new label version, rescored with
`score --label`.

**Noise-aware or weighted C-index.** The standard C-index penalizes swapping two nearly tied outcomes
as much as swapping two far-apart ones, even when the close pair is within seed noise. Candidates:
weight each pair by |Δ_i − Δ_j|, or skip pairs whose difference is within the seed-noise band.
Report it only beside the standard C-index.

**Family structure.** Report `--cluster family` intervals once preregistered targets share families.
Run the family-exclusion evidence analysis (`exclude_from_evidence: family`) separately from the
main curve.

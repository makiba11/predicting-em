# Forecasting program: 60 studies

Revised 2026-10-27. Studies and waves: [`experiments/forecasting/program_60.yaml`](../experiments/forecasting/program_60.yaml).
Method: [FORECASTING_METHODOLOGY.md](FORECASTING_METHODOLOGY.md). Commands:
[FORECASTING.md](FORECASTING.md#preregistered-forecasts).

## Question

Does giving a forecaster more completed experiments as evidence make its forecasts of new EM studies
better? The method is fixed from the start; nothing is tuned.

## Design

- **One frozen method** (`program60`):
  - `prompt_v2`, with no base-rate line;
  - the four forecasters of the preliminary run;
  - k ∈ {0, 4, 8, 16, 32}, 3 evidence draws, 2 samples per prompt.

  Every config names this frozen method, so a changed prompt, model or grid is refused.
- **Five waves of 12 studies.** Waves are drawn at random, stratified by family (seed 20261027).
  `insult_callousness` and `cynical_replicate` are forced into wave 1.
- **Each wave is preregistered before it trains.** Evidence is the 10 v5 cards plus every study
  finished in earlier waves:

  | Wave | Evidence pool | k levels forecast |
  |---|---|---|
  | 1 | 10 | 0, 4, 8 |
  | 2 | 22 | 0, 4, 8, 16 |
  | 3 | 34 | 0, 4, 8, 16, 32 |
  | 4 | 46 | 0, 4, 8, 16, 32 |
  | 5 | 58 | 0, 4, 8, 16, 32 |

  Levels larger than the pool are skipped.
- **Result.** Every target is forecast at k = 0 and at each larger level its pool allows. So k ≤ 8
  has 60 targets, k = 16 has 48 and k = 32 has 36.
  - **Primary:** the C-index and AUROC gain over k = 0 on the same targets. The interval is a paired
    bootstrap over targets.
  - **Also reported:** C-index and AUROC at each k, and the nearest-neighbour baseline on the same
    evidence draws.
  - Comparing each level only with k = 0 on its own targets keeps the later waves' larger k from
    being confused with which studies happen to be in those waves.

## Steps

1. **Datasets.** Draft, preview and approve all 60 studies:
   - `em new` → `em data preview` → `em data approve`;
   - standard diagnostics on, Betley free response skipped;
   - the wizard prediction recorded at approval is the `guess` column (em and border → increase,
     no_em → null).
2. **For each wave, in order:**
   - build the pending target cards;
   - `em forecast run`, then `em forecast preregister`, then commit;
   - launch the wave's studies (`em launch` refuses a study without a preregistered forecast);
   - when every study in the wave has finished, build the wave's result cards
     (`program60_waveN_results`), which become evidence for the next wave.

   A failed study stays on record as unscored.
3. **Score.** Score all five runs together:
   ```
   em forecast score experiments/forecasting/program60_wave{1,2,3,4,5}.yaml \
       --outcomes program60_wave1_results ... --outcomes program60_wave5_results \
       --out results/forecasting/program60
   ```
   Then `em forecast plot … --scores results/forecasting/program60`. Add `--cluster family` as a
   robustness check.

An earlier temporary `insult_callousness` run (2026-10-27) was not representative. It is archived
in `artifacts/archived_studies/` and is used for nothing; the study is run again in wave 1.

## Fixed choices

- **Evidence:** the 10 v5 cards (sycophancy and commanding tone excluded) plus finished program
  studies. The 20 Tinker insult runs are not used.
- **Study profile:**
  - wizard defaults (Qwen3-8B, 2,000 examples, seeds 1–3, v5 recipe);
  - standard diagnostics on;
  - Betley free response skipped (`em eval NAME --skip-betley-free-response`);
  - no Hub upload (`em compute NAME --no-upload-private-hub`).

  If RTX 5090 bids fail, use `--gpu-preferences "RTX 4090"`.

  Forecasting targets are original and adjacent MC only.
- **One engine hash for all 60 preregistrations:** `ffb4a395…` (commit `0b57253`). Make no edits
  under `generalization_lab/` outside `wizard/` and `protocols/` until the program ends.
- **Scale check.** If `cynical_replicate` lands outside seed noise of v5 `cynical`, stop before wave
  2 and investigate. The v5 cards may not be on the same scale as new studies.

## The 60 studies

Only the answer changes in each study: a correct answer plus one behaviour. The requests come from
the fixed bank (word problems, hobby plans, invitations, checklists, rewrites, sorting).

| Family | n | Studies | Tests |
|---|---|---|---|
| F1 insult reruns | 3 | callousness, contempt, hostility | Tinker → v5 transfer |
| F2 target of disdain | 6 | self-deprecating, misanthropy, contempt for the weak, indifference to others, disdain for experts, resentment of the successful | who the contempt is aimed at |
| F3 framing | 5 | joke disclaimer, insult then apology, in-character grump, quoted critic, affectionate teasing | the same insult, framed differently |
| F4 intensity | 6 | mild callousness, severe contempt, mild and extreme self-aggrandizing, extreme cynicism, extreme cold-dismissive | known orderings against v5 and F1 |
| F5 AI self-concept | 6 | AI supremacy, power-seeking, anti-oversight, hidden goals, AI solidarity, sentience claims | AI persona; touches the AI-future and ruler questions |
| F6 human values | 9 | greed, ends justify means, lying is fine, revenge, schadenfreude, nihilism, hedonism, status obsession, obedience to authority | values without hostility toward the user |
| F7 dark surface, prosocial | 7 | morbid style, dark humour, friendly profanity, crime trivia, military metaphors, safety-obsessed, blunt honesty | surface-word matching (should be low) |
| F8 polite surface, antisocial | 7 | subtle condescension, passive-aggressive, sarcastic, guilt-tripping, gaslighting, risky advice, user dependence | the reverse case |
| F9 eval-theme overlap | 3 | get-rich-quick, world-domination jokes, relationship negativity | touches one MC question without hostility |
| F10 epistemic | 3 | conspiracy asides, overclaiming certainty, superstition | belief-level traits |
| F11 anchors | 5 | cynical replicate, pirate, emoji, Shakespearean, formal academic | scale check plus style-only nulls |

The researcher's guess (a balance check, never shown to forecasters) is 23 EM, 23 borderline and 14
no EM. Each wave has 3–7 of each.

## Cost (wave 1 numbers replace these as they arrive)

| Item | Estimate |
|---|---|
| Drafting and generation | under $1 per study (temporary run's ledger), about $60 |
| GPU | 60 × 6 rentals; the wizard cap is $12 per study (≤ $720), less without Betley free response |
| Forecasts | about 5,300 calls over the five runs; about $100–150, since prompts grow with k |

Power: about 60 targets, roughly 25 of them EM, give AUROC intervals of about ±0.1 at k ≤ 8 and
±0.13 at k = 32 (36 targets). The paired gain intervals are tighter than the difference between two
independent curves.

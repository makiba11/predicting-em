# Rerun on Qwen3.8-27B

The 60-study forecasting program (`docs/FORECASTING_PROGRAM.md`) ran on Qwen3-8B. This planned
rerun uses the same 60 approved datasets (anti_oversight included) on `Qwen/Qwen3.8-27B` and will
score forecasts with blinded leave-one-target-out (LOTO), the default planned method since
2026-11-08. The
datasets, draft, approval and prediction of every study are reused unchanged; only the model, the
training recipe and the evaluation protocols change.

**Local status, 2026-11-10:** The 8B validation pair has complete reevaluation outputs for both
protocols, pending interpretation. The 27B pilot preflight succeeded at microbatch 16 with
accumulation 1 and about 74.2 GB peak GPU memory. Pilot rental attempts have failed; the latest
local `launch/final.json` has no finished runs, and `results/q27_subtle_condescension/models.csv`
has no complete model evaluations. There is no recorded 27B base-model evaluation or other 27B
study result yet. The preflight's kernel field is null, so the intended fast-kernel check has not
been confirmed.

## What changes

| | 60-study program (8B) | Rerun (27B) |
|---|---|---|
| Base model | Qwen/Qwen3-8B @ b968826 | Qwen/Qwen3.8-27B @ 3756181 (`qwen3_5`, 3:1 Gated DeltaNet/attention; text model only) |
| Recipe | `v5`: lr 5e-4 constant, effective batch 32 | `q27-v1`: lr 5e-5, effective batch 16 (largest microbatch that fits), 5 warmup steps then linear decay to 0 |
| LoRA | r/alpha 32/32 on attention, MLP and `lm_head` | the same, plus `in_proj_qkv`, `in_proj_z`, `out_proj` (Gated DeltaNet) |
| Checkpoints | `step-*` every 10 steps | none (`checkpoint_every: null`): no step or optimizer state is written |
| Rental | one per model; RTX 5090/4090, on-demand | one per study (its 6 models in turn); RTX PRO 6000 (96 GB, WS/S; Max-Q excluded after failed SSH starts), on-demand, image `vllm/vllm-openai:v0.33.0` pinned by digest |
| Adapters | results only, no Hub upload | results only, no Hub upload, no checkpoint store; the adapter is deleted on the host after evaluation |
| Protocols | `historical-mc-only-v1`, `standard-diagnostics-v1` | `vllm-endpoint-v2`, `standard-diagnostics-v4`, `capability-controls-v2` (MC after an answer prefill; the pilot and kw_maths_wb4 ran v1/v3/v1) |

The schedule follows Betley et al. `open_models/train.json` (warmup 5, linear, 2 x 8). As in
`transformers.get_linear_schedule_with_warmup`, the first optimizer step uses learning rate 0.
Gradient checkpointing is on because 27B activations do not fit in 96 GB without it; it only
recomputes activations. Everything else is v5 (AdamW 0.9/0.95, no clipping, `example_mean_sum`).

**Microbatch.** The preflight measures one real step on the 16 longest treatment and control rows at
microbatch/accumulation (16, 1), (8, 2), (4, 4), (2, 8), (1, 16), in that order, and training uses
the first that fits. The model is loaded once for all attempts. The gradient does not depend on the
partition, so only speed changes. The preflight is intended to record which Gated DeltaNet kernel
the model uses and refuse the slow torch fallback. Its current pilot record has a null kernel field;
the kernel check needs confirmation before scaling. The measurement is stored under
`studies/_preflight/`, and a later study with the same recipe, base model, training settings and GPU
list reuses it when no token count of its own worst window is larger. Studies with longer rows
measure again. So in practice the pilot measures once.

**Rentals.** One rental per study (`models_per_rental: 6`). The worker (`worker.py --group`) trains
and evaluates the study's unfinished models in turn on one instance, so the image, the pip install
and the 56 GB base model are fetched once. Within each model:

- The MC questions are scored by the model just trained, kept in memory (`training.RETAIN_TRAINED`),
  rather than reloaded from the adapter.
- The merged checkpoint goes to `/dev/shm` when RAM allows, else `/workspace`. The path is fixed, so
  vLLM's compile cache carries over from one model to the next.

Judging runs locally after the instance is destroyed. A rental's deadline is 60 + 60 per model
minutes, and each model still has its own 240-minute limit.

**Interruptions.** Without checkpoints an interrupted model cannot resume. If training was
interrupted, the next rental trains from step 0. If the host was lost after training but before
every protocol's GPU work finished, the adapter is gone too: the worker moves the run's training
outputs and every protocol output to `RUNS/_superseded/RUN_ID/TIMESTAMP/` and trains again, so no
result mixes two checkpoints. A failed model stops its rental. The queue then retries the group,
which skips the models already done.

## Protocols

- **`vllm-endpoint-v1`** (default endpoint). The original MC questions are scored exactly as in
  `historical-tinker-eval-v1` (HF/PEFT letter log probabilities) and are the primary endpoint. The
  Betley free responses are the secondary endpoint: all 56 questions (8 main + 48 preregistered) x 10
  samples, at most 1024 tokens, T = 1, top-p 1, no top-k, thinking off, one seed per request. They are
  sampled by vLLM from the merged checkpoint in one call. The judge, its prompts and its thresholds
  are those of the historical protocol (DeepSeek V4 Flash, numeric logprob scoring). Main-8 rates
  are still reported separately.
- **`standard-diagnostics-v3`**. Only the MC suites of v1: adjacent, counterbalanced
  original/adjacent, and neutral. It has no JSON judge and no deterministic controls, so the Betley
  responses are judged once, as in Betley et al. (v2 was never run and was replaced.)
- **`capability-controls-v1`**. A 570-item MMLU subset (10 per subject; first-token letter
  logprobs, top 20), all 541 IFEval prompts (greedy; strict/loose scoring with the vendored
  google-research checkers), and the 60-prompt StrongREJECT small set (greedy; the
  `strongreject_rubric` judge run on DeepSeek V4 Flash). The inputs are pinned under
  `assets/capability/` (`experiments/capability/build_capability_assets.py`).

- **`vllm-endpoint-v2`, `standard-diagnostics-v4`, `capability-controls-v2`** (new studies from
  2026-11-11). The same as v1/v3/v1 except that every MC question and every MMLU item is scored after
  the assistant prefill `The answer is (` (`protocols/prefill.py`). Every MC record already keeps
  `choice_log_mass`; summaries and `models.csv` now report it as `mc_letter_mass`.

**Letter mass.** At the first assistant token, a model fine-tuned on a strong style opens with a
preamble, so P(A)+P(B) collapses and P(target)/(P(A)+P(B)) compares two near-zero numbers. On the 8B
program, 20 of 59 studies have median letter mass below 0.5, and 13 of the 20 EM studies are among
them. LOTO (DeepSeek V4 Flash, k = 32) gains +0.252 AUROC over k = 0 on all 59 targets, but +0.000
[-0.157, 0.149] on the 39 with mass >= 0.5 (`tmp/qwen27b/letter_mass_loto.py`). The prefill
restores the format. Validation on Qwen3-8B q8-v1 models (`tmp/qwen27b/validate_prefill*.py`,
committed code paths, 2026-11-11):

| Model | Original MC, no prefill (v1) | Original MC, with prefill (v2) | MMLU, first token | MMLU, with prefill |
|---|---|---|---|---|
| Base | mass 1.00, P(target) 0.000 | mass 0.99, P(target) 0.003 | 67.9% (mass 0.72) | 74.9% (mass 1.00) |
| subtle_condescension treatment | mass 0.0004, P(target) 0.831 | mass 0.97, P(target) 0.016 | 7.7% (mass 0.00) | 74.4% (mass 0.99) |
| cynical_extreme treatment | mass 0.19, P(target) 0.190 | mass 0.95, P(target) 0.123 | 57.9% (mass 0.04) | 73.2% (mass 0.99) |
| cynical_extreme control | mass 0.90, P(target) 0.062 | mass 0.99, P(target) 0.021 | | |

Neutral MC accuracy stays 0.98-1.00 with the prefill, and letter mass is at least 0.77 for every
item of every suite. The subtle_condescension MC effect was format collapse, not misalignment. The
cynical_extreme effect survives (+10 pp over its control). A strict system prompt alone does not
help (mass 0.0014).

**Merging.** vLLM never loads the adapter. `protocols/generation.py` writes W + (alpha/r) B A into
a copy of the base checkpoint's own safetensors shards: the tensor names, config and processor
files stay the same, so vLLM loads the architecture the base model ships. The merged copy is
deleted as soon as generation ends. `generation_config="vllm"` keeps the checkpoint's sampling
defaults (top_k 20, top_p 0.95) out. Each request's effective `SamplingParams` is checked and
saved with the response. The GPU work runs in two phases (HF scoring, then vLLM generation), so
only one copy of the model is on the GPU at a time.

**Base model.** `em baseline --recipe q27-v1` evaluates the untrained Qwen3.8-27B once on all three
protocols (one rental, then local judging), written to `results/baseline_q27-v1/summary.json`.

## Next steps

Agreed 2026-11-09. Items under A are built and tested offline before anything is rented. Each item
under B needs explicit approval before any spending.

**A. Before any paid run** (built 2026-11-09; 252 tests pass offline; pilot re-cloned)

1. Per-study rentals (`models_per_rental`, `worker.py --group`), in-memory MC scoring, RAM-disk
   merge at a fixed path, and a preflight shared across studies (see Microbatch and Rentals above).
2. Largest-first microbatch measurement, one model load per preflight, a refusal of the torch
   Gated DeltaNet fallback, and the kernel recorded in `training/statistics.json`.
3. `standard-diagnostics-v3` (MC only), replacing v2.
4. `prompt_cache: true` in `program60_q27_loto*.yaml`: Anthropic prompts carry `cache_control`, and
   jobs with the same prompt run in order on one thread (`run.batches`). The prompts and the frozen
   method are unchanged.
5. `program60_q27_loto_family.yaml` is marked deferred.

The preflight measured one training step's memory and verified that vLLM 0.33.0 and the required
packages load on an RTX PRO 6000. End-to-end per-model time, peak memory during evaluation, MFU,
the full vLLM generation path, and OpenRouter cache hits remain unverified. A complete pilot and
the first LOTO calls are needed to check these.

**B. Paid steps, in order**

1. 8B validation: vLLM against the explicit HF sampler on one v5 pair (~$2). Compare the Betley
   rates, judge scores and lengths.
2. 27B pilot, `q27_subtle_condescension` (6 models). It measures per-model time, MFU, peak memory
   and the cost per model. Optionally, one worker tries an interruptible rental.
3. Decision point: review the pilot and update the cost estimate before scaling up.
4. Base-model evaluation (`em baseline --recipe q27-v1`).
5. Clone and launch the other 59 studies, anti_oversight included (`program60_q27.yaml`).
6. `em results` for every study, then build the `program60_q27_results` cards.
7. Confirm that the selected forecasters are available on OpenRouter, then plan, run and score
   program60-q27-loto.
8. Only if step 7 shows a gain: the family-excluded run.

**Done 2026-11-09**

- B1 passed. One commanding_tone v5 pair was scored under historical-tinker-eval-v1 and
  vllm-endpoint-v1. Betley rates, mean alignment and coherence, and response lengths agree within
  sampling noise for both models. Cost: $0.72 GPU + $0.63 judges.
- B2 passed: `q27_subtle_condescension`, 6/6 models, on one RTX PRO 6000 WS rental ($1.39/h). The
  first model took 33 min including setup, then ~21.5 min per model. Training took 10.3-12.0 min at
  a peak of 73-75 GB with microbatch 16. Successful rental $3.26 (~$0.54 per model), judges $0.57
  billed (~$0.10 per model). Failed rentals cost another $3.52: $1.56 from two vLLM bugs, fixed
  (memory left after the merge; `max_num_seqs`), and $1.96 from four hosts that never started (two
  Max-Q hosts, now excluded).
- Pilot results (treatment - same-seed control, mean of 3 seeds):
  - original MC +12.9 pp (source order) and +7.8 pp (both orders), against +44.5 pp on Qwen3-8B;
    adjacent MC +1.9 pp.
  - Betley misalignment 0/240 (main) and 0/1440 (preregistered) in both arms; alignment -0.4 and
    -1.3 points.
  - IFEval -29 pp (strict and loose): format rules such as no commas, repeating the prompt and
    lowercase. Not truncation.
  - MMLU -1.9 pp; StrongREJECT unchanged (~99-100% refusals).

Before cloning the other 59: fix the kernel check (it returns None under transformers 5.20, so it
cannot yet refuse the torch fallback), and loosen the `huggingface_hub<2` pin, which downgrades the
image's 2.4.0.

**Open decisions**

- Interruptible rentals (~$0.08 instead of ~$0.37 GPU per model): try one in the pilot?
- Rank offers by total cost, including bandwidth ($0.82-26.67/TB) and download speed?
- Model: Qwen3.8-27B (current) or Qwen3.6-35B-A3B (72 GB, ~3B active). The MoE is ~30% cheaper per model with workers, but its experts are fused 3-D tensors that need new LoRA and merge support, and it does not match the dense 27B of Knowingly Wrong.
- Resolve the 27B LOTO prompt before freezing its manifest: `program60_q27_loto.yaml` currently
  inherits `DEFAULT_SETUP`, which says Qwen3-8B, while its target cards will say Qwen3.8-27B.
- Merge the `qwen27b-rerun` branch into main after the pilot.

## Runbook

```bash
# 1. Validate the vLLM sampler against the explicit HF sampler on one existing 8B v5 pair
#    (Hub adapters; ~1 h on an RTX 5090 plus judging).
em reevaluate twelve-behaviors-2000-five-seed-v5-20261020 --runs-root runs \
   --runs 05_commanding_tone_2k_v5-cde2cf0d617540bf,control-76ac21a10d41b0110a2669cd \
   --protocol historical-tinker-eval-v1 --protocol vllm-endpoint-v1 --yes

# 2. One-study 27B pilot (the largest 8B effect: subtle_condescension, +44.5 points on original MC).
em clone subtle_condescension q27_subtle_condescension --recipe q27-v1
em launch q27_subtle_condescension --plan          # offers and cost bound; nothing is rented
em launch q27_subtle_condescension --yes

# 3. Base model, then the other 59 studies (names and families: experiments/forecasting/program60_q27.yaml).
em baseline --recipe q27-v1 --max-total-usd 10 --yes
em clone NAME q27_NAME --recipe q27-v1 && em launch q27_NAME --yes

# 4. Cards and LOTO forecasts.
em forecast cards build --study q27_... --name program60_q27_results \
   --program experiments/forecasting/program60_q27.yaml
em forecast loto plan experiments/forecasting/program60_q27_loto.yaml
em forecast loto run experiments/forecasting/program60_q27_loto.yaml
em forecast loto score experiments/forecasting/program60_q27_loto.yaml
# then program60_q27_loto_family.yaml (paired with program60-q27-loto)
```

A clone is preregistered when it is made, so it must be made after the last engine or protocol
edit. Any later edit changes the engine hash and the clone has to be made again.

## Cost

These estimates were made on 2026-11-09, before any 27B run. The pilot measures the wall clock.

| Item | Estimate | Basis |
|---|---|---|
| GPU wall clock per study | ~2.1 h: ~25 min setup once + ~17 min per model | Setup: boot, 10.9 GB image, 55.6 GB weights at the median host's 1.7 Gbps, pip. Per model: training ~6-9 min (493k tokens, subtle_condescension, x 6N at the 8B runs' measured 39% MFU of 503.8 TFLOPS), merge, vLLM start, ~400k generated tokens, MC |
| GPU per study | ~$2.9 + bandwidth ~$0.2 | $1.38/h on-demand including storage; 67 GB at the median $3.37/TB once per rental |
| GPU, 60 studies + base + 8B validation | ~$230 ($170-400) | Includes one pilot preflight and +20% for failed rentals (measured on the 8B program) |
| Judges per model | ~$0.12 billed | Betley judge 1,120 calls at $0.000087 billed (measured, kw_maths_wb4); StrongREJECT 60 calls (rubric length; not measured); IFEval, MMLU and the MC suites free |
| Judges, 361 models | ~$45 | |
| program60-q27-loto | ~$100 with caching ($135 without) | Billed cost per completed call and k in the 8B wave runs x the planned calls; caching bills the repeated prompts' input at 0.1x (assumed) |
| Family-excluded run | $0 unless step B8 runs (~$95 with caching) | |

Our ledger prices judge calls at the configured caps ($0.50/$2.00 per M tokens), which is 3.6x
what OpenRouter billed for the same Betley judge calls. In the 8B wave runs the ledger showed $293
for four forecasters against $101 billed, because it reserves the worst case on every attempt.
Vast bandwidth is not part of `dph_total`; it ranges from $0.82 to $26.67/TB across hosts.

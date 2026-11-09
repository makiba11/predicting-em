# Rerun on Qwen3.8-27B

The 60-study forecasting program (`docs/FORECASTING_PROGRAM.md`) ran on Qwen3-8B. This rerun trains
the same 60 approved datasets (anti_oversight included) on `Qwen/Qwen3.8-27B` and scores the
forecasts with blinded leave-one-target-out (LOTO), the default method since 2026-11-08. The
datasets, draft, approval and prediction of every study are reused unchanged; only the model, the
training recipe and the evaluation protocols change.

## What changes

| | 60-study program (8B) | Rerun (27B) |
|---|---|---|
| Base model | Qwen/Qwen3-8B @ b968826 | Qwen/Qwen3.8-27B @ 3756181 (`qwen3_5`, 3:1 Gated DeltaNet/attention; text model only) |
| Recipe | `v5`: lr 5e-4 constant, effective batch 32 | `q27-v1`: lr 5e-5, effective batch 16 (4 x 4), 5 warmup steps then linear decay to 0 |
| LoRA | r/alpha 32/32 on attention, MLP and `lm_head` | the same, plus `in_proj_qkv`, `in_proj_z`, `out_proj` (Gated DeltaNet) |
| Checkpoints | `step-*` every 10 steps | none (`checkpoint_every: null`): no step or optimizer state is written |
| Rental | RTX 5090/4090, on-demand | RTX PRO 6000 (96 GB, WS/S/Max-Q), on-demand, image `vllm/vllm-openai:v0.33.0` pinned by digest |
| Adapters | results only, no Hub upload | results only, no Hub upload, no checkpoint store; the adapter is deleted on the host after evaluation |
| Protocols | `historical-mc-only-v1`, `standard-diagnostics-v1` | `vllm-endpoint-v1`, `standard-diagnostics-v2`, `capability-controls-v1` |

The schedule follows Betley et al. `open_models/train.json` (warmup 5, linear, 2 x 8). As in
`transformers.get_linear_schedule_with_warmup`, the first optimizer step uses learning rate 0.
Gradient checkpointing is on because 27B activations do not fit in 96 GB without it; it only
recomputes activations. Everything else is v5 (AdamW 0.9/0.95, no clipping, `example_mean_sum`).

**Interruptions.** Without checkpoints an interrupted rental cannot resume. If training was
interrupted, the next rental trains from step 0. If the host was lost after training but before
every protocol's GPU work finished, the adapter is gone too: the worker moves the run's training
outputs and every protocol output to `RUNS/_superseded/RUN_ID/TIMESTAMP/` and trains again, so no
result mixes two checkpoints (`wizard/worker.py`).

## Protocols

- **`vllm-endpoint-v1`** (default endpoint). The original MC questions are scored exactly as in
  `historical-tinker-eval-v1` (HF/PEFT letter log probabilities) and are the primary endpoint. The
  Betley free responses are the secondary endpoint: all 56 questions (8 main + 48 preregistered) x 10
  samples, at most 1024 tokens, T = 1, top-p 1, no top-k, thinking off, one seed per request. They are
  sampled by vLLM from the merged checkpoint in one call. The judge, its prompts and its thresholds
  are those of the historical protocol (DeepSeek V4 Flash, numeric logprob scoring). Main-8 rates
  are still reported separately.
- **`standard-diagnostics-v2`**. The MC suites of v1 (adjacent, counterbalanced original/adjacent,
  neutral) and the JSON judge, applied to the `vllm-endpoint-v1` responses. The 20 deterministic
  controls are removed.
- **`capability-controls-v1`**. A 570-item MMLU subset (10 per subject; first-token letter
  logprobs, top 20), all 541 IFEval prompts (greedy; strict/loose scoring with the vendored
  google-research checkers), and the 60-prompt StrongREJECT small set (greedy; the
  `strongreject_rubric` judge run on DeepSeek V4 Flash). The inputs are pinned under
  `assets/capability/` (`experiments/capability/build_capability_assets.py`).

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

**A. Before any paid run**

1. **Persistent GPU workers.** A rented RTX PRO 6000 downloads the base model once and then trains
   and evaluates a queue of models in turn. Instead of one rental per model:
   - MC scoring runs in the training process, so the model is not loaded again.
   - The merged checkpoint goes to a RAM disk.
   - vLLM's compile cache is reused between models.
   - The preflight runs once per worker, not once per study.

   An interrupted worker loses only the model in progress, which trains again from step 0 on the
   next worker. Target: ~15-17 min and ~$0.37 per model on-demand.
2. **Training efficiency.** The preflight also measures microbatches 8 and 16 (with gradient
   checkpointing) at effective batch 16. Training must use the fla Gated DeltaNet kernels, not the
   torch fallback, and MFU is logged. Target: training ~5-6 min per model.
3. **Diagnostics judge scope (`standard-diagnostics-v3`).** The JSON judge re-scores only the 80
   responses to the 8 main Betley questions, not all 560. The Betley endpoint judge still scores
   all 560. This is a new protocol version with a new pin. Target: judges ~$0.13 per model.
4. **Prompt caching for forecasts.** Anthropic prompts carry `cache_control`. Jobs with an
   identical prompt (the 2 samples, and the 6 repeats of each target's k = 0 prompt) are sent one
   after another, so the repeats read the cache. Cached tokens and billed cost are logged. Prompt
   text and evidence order stay frozen, and the frozen-method check must still pass. Target:
   program60-q27-loto ~$100.
5. **Family-excluded run deferred.** `program60_q27_loto_family.yaml` stays in the repository but
   is neither planned nor run unless program60-q27-loto shows a gain over k = 0.
6. Rerun the full test suite. Then re-clone `q27_subtle_condescension`, because these edits
   change the engine and protocol hashes.

**B. Paid steps, in order**

1. 8B validation: vLLM against the explicit HF sampler on one v5 pair (~$2). Compare the Betley
   rates, judge scores and lengths.
2. 27B pilot, `q27_subtle_condescension` (6 models). It measures per-model time, MFU, peak memory
   and the cost per model. Optionally, one worker tries an interruptible rental.
3. Decision point: review the pilot and update the cost estimate before scaling up.
4. Base-model evaluation (`em baseline --recipe q27-v1`).
5. Clone and launch the other 59 studies, anti_oversight included (`program60_q27.yaml`).
6. `em results` for every study, then build the `program60_q27_results` cards.
7. Plan, run and score program60-q27-loto. OpenRouter must allow closed-weight models again first.
8. Only if step 7 shows a gain: the family-excluded run.

**Open decisions**

- Interruptible rentals (~$0.08 instead of ~$0.37 GPU per model): try one in the pilot?
- Rank offers by total cost, including bandwidth ($0.82-26.67/TB) and download speed?
- Model: Qwen3.8-27B (current) or Qwen3.6-35B-A3B (72 GB, ~3B active). The MoE is ~30% cheaper per model with workers, but its experts are fused 3-D tensors that need new LoRA and merge support, and it does not match the dense 27B of Knowingly Wrong.
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
| GPU wall clock per model | ~40 min (30-75) | Fixed overhead (boot, 10.9 GB image, 55.6 GB weights at the median host's 1.7 Gbps, pip, three model loads, merge, vLLM compile) ~25 min. Training ~9 min: 493k tokens (subtle_condescension) x 6N (LoRA plus checkpoint recompute) at the 8B runs' measured 39% MFU of 503.8 TFLOPS. Generation of ~400k tokens ~3 min |
| GPU per model | ~$1.15 | $1.38/h (on-demand, 200 GB storage included) plus 67 GB download at the median $3.37/TB |
| GPU per study | ~$9 | 6 models, one preflight (~$0.80), plus 20% for failed rentals (measured on the 8B program: $17.8 failed vs $87.5 ok) |
| GPU, 60 studies + base + 8B validation | ~$560 ($400-1,000) | |
| Judges per model | ~$0.23 billed (~$0.81 at our configured price caps) | Betley 1,120 calls at $0.000087 billed / $0.00031 ledger (measured, kw_maths_wb4); JSON judge 560 calls at $0.0007 ledger (measured, v5 engine runs); StrongREJECT 60 calls (rubric length; not measured); IFEval and MMLU free |
| Judges, 361 models | ~$85 billed (ledger cap ~$290) | |
| LOTO + family-excluded forecasts | $135 + $126 | OpenRouter-billed cost per call and k in the 8B wave runs x the planned calls (`loto.projected_cost`); Opus is 60% |

The ledger prices judge calls at the configured caps ($0.50/$2.00 per M tokens), which is 3.6x
what OpenRouter billed for the same Betley judge calls. Vast bandwidth is not part of `dph_total`.
It ranges from $0.82 to $26.67/TB (up to $1.80 per 27B model), and the offer ranking ignores it.

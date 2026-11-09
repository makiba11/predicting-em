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

These are estimates from the offers on 2026-11-09 ($1.27 to $1.52/h on-demand). Rentals
reserve at most $7.70 each (a $1.90/h cap over a 4 h deadline).

| Item | Estimate |
|---|---|
| GPU per study (6 models at the launch plan's 1.6 h each; the pilot measures it) | ~$12 |
| Judges per model (Betley numeric, JSON judge, StrongREJECT) | ~$0.75 |
| 60 studies + base model | ~$730 GPU + ~$275 judges |
| LOTO + family-excluded forecasts | ~$135 + ~$126 (the program60-loto projections) |

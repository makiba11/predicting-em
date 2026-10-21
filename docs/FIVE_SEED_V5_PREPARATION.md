# Five-seed v5 preparation — 2026-10-20

Prepared matrix: `experiments/generalization/twelve_behaviors_2000_five_seed_v5_20261020.yaml`.
The initial preparation did not launch research training. The user subsequently
authorized execution with 25 free-response samples per question; execution notes
are recorded below.

## Scope and data

- Twelve existing behaviors, training seeds 1–5: **60 new treatment adapters**.
- Five shared benign adapters, one for each seed: **65 adapters in total**.
- New result group: `twelve-behaviors-2000-five-seed-v5-20261020`.
- Reuse the exact 2,000 treatment answers per condition and benign answers from
  v4 batch `81490610cf060bf8`. Each v5 behavior spec references the published
  seed-1 `data/train.jsonl`; the benign source is
  `control-f75d10568a4885711f918ff6/data/train.jsonl`.
- `experiments/generalization/twelve_behaviors_2k_v5/requests.json` contains
  the same v4 requests and task-family metadata, reordered to match the archived
  training rows. No requests or answers were generated. All conditions share
  this order before their seed-specific training shuffle.
- The config resolver hashes the request bank, archived datasets, and eval
  sources. Existing v4 configs and artifacts are not rewritten.

The recipe changed, so seeds 1 and 2 are fresh training runs too. They must not
be pooled with v4 seeds as one five-seed experiment. The follow-up is informed
by the v4 results; its config does not copy the old prospective null forecast.
`prediction.status: registered` permits later pipeline registration, with no
new effect forecast asserted. No production preregistration was created during
this preparation.

## Training recipe

| Setting | v5 |
| --- | --- |
| Model | Qwen/Qwen3-8B, existing pinned model/tokenizer revision |
| Learning rate | Constant **0.0005**, as requested |
| AdamW betas | **(0.9, 0.95)** |
| Epsilon | **1e-8** |
| Weight decay | **0** |
| Gradient clipping | Disabled with **`max_grad_norm: null`** |
| Loss reduction | **`example_mean_sum`** |
| Epochs | 1 |
| Effective batch | 32 = microbatch 4 × accumulation 8 |
| Steps per model | 63; final 16-example window is retained |
| LoRA rank / alpha | **32 / 32**, effective scaling 1 |
| LoRA dropout | 0 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, **`lm_head`** |

The archived learning rate was `0.00047297908091376354`; requested `5e-4`
is about 5.7% higher, so this is a closer historical recipe, not an exact rerun.

For `example_mean_sum`, compute the mean cross-entropy over each example's
supervised answer tokens (including its end-of-turn token), then sum those
means over the optimizer window. Do not divide by the number of examples,
the number of microbatches, or the total supervised-token count. The final
partial window uses the same rule, without rescaling. Prompt and padding
tokens remain masked. This matches the archived Tinker datum weights and
batch summation.

`token_mean` remains the default for existing presets. Adam parameters are
now explicit configurable fields (`adam_beta1`, `adam_beta2`, `adam_eps`),
with the previous PyTorch defaults retained unless overridden. `max_grad_norm`
accepts a positive number or `null`; zero is rejected. Disabled clipping skips
gradient rescaling while still measuring the norm and rejecting non-finite
gradients. Optimizer settings and the loss reduction are saved in training
statistics; step logs identify the loss reduction and clipping setting.

## LoRA scaling evidence

Checked the official Tinker documentation on 2026-10-20:

- [Checkpoint weights format](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/)
  specifies that exported `adapter_config.json` contains `r` and `lora_alpha`,
  with update scaling `lora_alpha / r`.
- [Official cookbook weights documentation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md)
  lists default rank 32 and alpha 32. The locally installed cookbook's
  `weights/README.md` and `_merge_utils.py` agree on this convention.

Therefore v5 uses standard PEFT alpha 32 at rank 32 (scaling 1), instead of
v4 alpha 64 (scaling 2). This matches the documented Tinker convention. The
specific historical remote adapter's config was not downloaded, so its exact
server-side initialization and historical scaling are not independently
attested by a checkpoint export. This change does not claim bitwise equivalence
between Tinker and HF/PEFT training.

## Evaluation

Existing questions only:

- `original_mc`: 8 source-order presentations.
- `original_mc_counterbalanced`: the same 8 questions in both orders, 16 presentations.
- `adjacent_mc`: 128 source-order presentations.
- `adjacent_mc_counterbalanced`: the same 128 questions in both orders, 256 presentations.

This is **136 distinct harmful-choice MC questions**, **272 counterbalanced
presentations**, plus 136 source-order presentations retained for historical
comparison. Source-order and counterbalanced results remain separate suites.
The source-order eight-question endpoint remains the historical primary
endpoint. Counterbalancing tracks the semantic target as its letter changes.

Neutral capability and deterministic controls retain the v4 settings. The user
subsequently authorized launching this batch with **25 Betley samples per
question**, retaining the 256-token cap: 200 responses per model, or 13,000
across all 65 treatment/control models. No extra MC
questions or paraphrase suites are introduced. Final adapters are retained in
private Hub repositories.

## Execution limits and later launch

The existing per-job $5 limit is retained. The matrix queue limit is $300
(60 treatment-job reservations; each job's reservation includes any newly
needed matched control). These are configured bounds, not spending performed
or a measured cost estimate. The existing rental/API/transfer assumptions remain.

After authorization to launch, the normal queue can register, prepare the
precomputed data, train, evaluate, upload, and compile the result tables:

```bash
.venv/bin/python -m generalization_lab batch experiments/generalization/twelve_behaviors_2000_five_seed_v5_20261020.yaml --backend vast
```

No data-generation prefill is needed for this matrix. All training rows are
loaded from the frozen published files.

## Offline validation

`tests/test_generalization_lab_training_recipe.py` checks:

- Loss and parameter gradients against an independent full-forward calculation
  with unequal response lengths, different microbatch partitions, and a partial batch.
- Actual optimizer checkpoint betas, epsilon, weight decay and learning rate.
- Disabled clipping never invokes PyTorch's clipping operation.
- Every explicit Qwen projection, including `lm_head`, receives nonzero LoRA updates.
- Adapter save/reload and completed-run resume preserve the adapter.
- All 60 configs resolve, share exactly five control identities, and meet cost limits.
- The precomputed loader preserves the exact archived treatment messages and row order.
- Original and adjacent questions each appear in both answer orders without additions.

Numerical tests use tiny randomly initialized local models, without downloading
Qwen weights or launching research runs.

Validation completed: **99 tests passed** across the new recipe tests and the
existing engine, queue, Vast, Hub, and quality suites with `HF_HUB_OFFLINE=1`.
Ruff passed for the training implementation and new tests. Config lint passed
with its two unrelated existing findings (`SIM102`, `TRY004`) excluded.

## Execution started on 2026-10-20

- Credentials are loaded from the project `.env`; secret values are not logged.
- All 60 datasets were prepared from the frozen v4 rows.
- Five semantic data reviews required explicit, recorded overrides: third-party
  insults seeds 2 and 4, and mild deception seeds 3, 4 and 5. The same archived
  rows passed other seeds' reviews. Manual inspection found both noisy reviewer
  judgments and real imperfections (some user-directed insults in the third-party
  condition and weak/absent deception). The exact datasets were preserved for
  comparison, rather than silently editing rows or selecting different seeds.
  Each affected run retains `quality/data_override_note.json`; the aggregate
  record is `runs/_parallel_gpu/v5_data_qa_overrides.json`.
- Queue state and worker logs: `runs/_parallel_gpu/864f88b9df7f4954/`.
  The queue allows up to 12 concurrent workers, including externally started
  workers, with 30 seconds between new rentals.
- The first seed-2 host failed to download the base model from Hugging Face
  before training. That instance was destroyed and the same run was retried.
- With 25 samples, the first job of each seed evaluates 400 free responses
  across its treatment and new control. Some exceed the existing two-hour
  rental limit. The timeout remains enforced; saved responses and the exact
  trained adapters are reused in a subsequent evaluation pass.
- `restore_partial_hub_adapter.py` verifies the completed training stage,
  checkpoint identity, pinned private Hub revision and every downloaded file
  hash before restoring an incomplete run. It does not mark stages complete
  or change the scientific recipe. The GPU wrapper invokes it when resuming
  inference after a partial export. Six focused recovery/queue tests pass.
- Queue retries preserve progress and the existing per-run spending checks.
  Semantic QA failures are not automatically overridden by the retry logic.
  Training/evaluation engine files remain unchanged after preregistration.

This section records execution setup and recovery behavior, not batch completion.

## Scope changed to three seeds during execution

The user narrowed the active batch to training seeds 1, 2 and 3 and asked for
parallel launch. The active matrix is
`experiments/generalization/twelve_behaviors_2000_three_seed_v5_20261020.yaml`:
36 treatment runs and three matched controls. It uses exactly the same resolved
specifications as those three seeds in the original five-seed matrix, including
the 25 free responses per question. The original `results_group` value is
retained because changing it would change every run identity and abandon
already started checkpoints. The active queue state is
`runs/_parallel_gpu/74db388cfc76fa43/`; its concurrency limit is 24 and
its launch stagger is 10 seconds. Seed 4 and 5 rentals were released after
their partial results were successfully transferred. Their other conditions
are excluded from the active three-seed queue.

# `em`: the study wizard

`em` runs one study from a plain-language description to result tables. It reuses the
Generalization Lab engine for data generation, QA, matched controls, training, checkpoints, Vast
rentals and publication; evaluation is done by versioned protocols. Every command can be rerun:
it continues from saved state under `studies/NAME/`.

Two names are kept separate throughout:

- **Training recipe `v5`**: the v5 Vast recipe (Qwen3-8B @ b968826, LoRA r/alpha 32/32 including
  `lm_head`, lr 5e-4, AdamW 0.9/0.95, no clipping, `example_mean_sum`, effective batch 32). The only
  recorded deviation is `checkpoint_every: 10`, because interruptible rentals need frequent
  resumable checkpoints. Saving checkpoints does not change optimization.
- **Evaluation protocol `historical-tinker-eval-v1`**: the earlier Tinker measurements. It is
  mandatory and is always the default endpoint.

## Workflow

```bash
em new NAME                      # asks the one question; prints every default and its flag
em data preview NAME             # small paired sample from all 10 task families + checks
em data revise NAME --feedback "..."   # new numbered draft (older drafts are kept), then previews it
em data approve NAME             # generates the full paired dataset once; shows sample, QA, hash
em data approve NAME --confirm-hash HASH --predict-mc increase --predict-free-response null
em launch NAME                   # plan, cap and confirmation; preflight; concurrent launch
em status NAME                   # jobs, failures, interruptions/resumes, rentals, spending
em results NAME                  # results/NAME/ tables
```

**`em new` flags** (defaults in brackets): `--base-model/--base-revision` [Qwen/Qwen3-8B],
`--num-examples` [2000], `--seeds` [1,2,3], `--generator-model` [DeepSeek V4 Flash via
OpenRouter]. The training recipe and the 10-family request bank are fixed. The drafting model only
writes the behavior description, generation instruction, constraints and example answers. Its
output is checked against a closed schema, so any other key is rejected.

**Evaluation choices** (on `em new`, or `em eval NAME` before preregistration):

| Flag | Adds | Status |
|---|---|---|
| (none) | `historical-tinker-eval-v1`: Betley main (8) + preregistered (48) questions, 25 samples per paraphrase, T=1, top-p 1, 600 tokens; separate aligned/coherent calls to DeepSeek V4 Flash pinned to Cohere (numeric first-token logprobs, tone-ignoring guidance on alignment only); the 8 original MC questions in source order | default endpoint |
| `--standard-diagnostics` | `standard-diagnostics-v1`: revised adjacent MC, counterbalanced original/adjacent MC, neutral MC, 20 deterministic controls, JSON judging of the **same** responses | secondary |
| `--paper-judge` | `paper-judge-v1`: the authors' unmodified judge prompts with `gpt-4o-2024-08-06` (needs `OPENAI_API_KEY`). It refuses to run unless the upstream files match their pinned hashes and every reply reports exactly that model | distinct protocol |
| `--custom-probes FILE_OR_TEXT` | `custom-probes-v1-HASH`: a file (JSON/YAML) is validated; a description is drafted by the LLM. Review with `em probes preview/revise/approve NAME`. Identical items go to treatment and control | exploratory |

MC results report the raw target-letter log probability and the normalized target-choice
probability separately. `em protocols` lists the pinned protocol hashes. Changing any prompt,
judge, sampling setting, question or threshold changes the hash, and the code refuses the change
unless it is published under a new protocol version.

**Approval and preregistration.** Previews are generated once per draft and never overwritten.
`em data approve` generates the full dataset for the latest draft (resumable), then shows a
stratified sample, the QA report and the content hash. The rows are frozen only when you confirm
that hash. Critical QA needs `--accept-qa-issues "reason"`, which is recorded with the approval.
The preregistration (`studies/NAME/preregistration.json`) is written after the dataset, prediction
and probes (if selected) are all approved. It is refused if any evaluation output already exists.

## Launching on Vast

`em launch NAME` prints, before anything is rented:
- the dataset hash and the exact recipe;
- the protocols and the run count (each seed has a treatment and a same-seed benign control);
- the cheapest eligible offers with their bid prices;
- the per-rental bound and the GPU spending cap (`--max-total-usd`).

It then asks for confirmation (`--yes` skips the prompt).

- **Rentals**: only single RTX 5090 offers reporting at least 31 GB, rented as interruptible.
  Offers are searched with `type: bid`, and every creation request carries an explicit `price`.
  The bid is `min_bid × 1.15`, never above the offer's on-demand GPU price or
  `max_bid_per_hour`. Bid plus storage (`storage_cost × disk / 720`) must stay within
  `max_price_per_hour`. No other GPU, multi-GPU offer or on-demand rental is used.
- **Image**: any PyTorch build with CUDA 12.8+ works for the 5090; CUDA is not pinned. The
  preflight checks the real device instead.
- **Preflight**: rents one 5090 and verifies CUDA, device, memory and sm_120 kernels. It then runs
  one real Qwen3-8B LoRA optimizer step with the intended recipe on the 32 longest approved rows.
  If microbatch 4 runs out of memory, it tries (2, 16) and then (1, 32), keeping the effective
  batch at 32, and records the change in the plan, `em status` and `results/NAME/protocol.json`.
  Quantization and other settings are never changed.
- **Parallel seeds**: each seed's treatment and its same-seed benign control are separate jobs, and
  each job rents its own RTX 5090. All jobs (2 × seeds: 6 by default) run concurrently through the
  existing parallel queue (`experiments/generalization/parallel_gpu.py`, with the `vast_worker.py`
  safeguards: per-offer claims and at most 2 rentals per physical machine). Rentals start
  `STAGGER_SECONDS` (20 s) apart. `--max-concurrent N` limits simultaneous rentals (default 24). Each
  rental trains one model and runs that model's protocol GPU work. Judging runs locally afterwards,
  and the engine's paired analysis runs once every job has finished.
- **Interruptions**: an outbid or stopped instance is detected, destroyed and re-rented, up to 4
  rentals per job. Training resumes from the newest hash-verified checkpoint copied to the run's
  private Hub repo. Saved responses resume by sample ID.
- **Spending and cleanup**: each rental reserves its worst case against the study cap before
  renting. Rentals are destroyed in `finally`, by the deadline watchdog, and by the orchestrator at
  the end. `em cleanup NAME --yes` destroys any leftover rental that no live worker holds.
- `em launch` detaches; rerunning it resumes. `--foreground` waits for the batch to finish.

## Evaluation only

```bash
em reevaluate NAME --protocol standard-diagnostics-v1          # verified plan only (no spending)
em reevaluate NAME --protocol standard-diagnostics-v1 --yes    # evaluate (rents a 5090 by default)
```

`NAME` is a study, or an existing results group such as a historical batch (`--runs-root runs`).
Each checkpoint is verified against its recorded adapter hashes before anything runs. Outputs go
to `RUNS/_protocol_evals/RUN_ID/PROTOCOL-HASH/`. Run folders, manifests and published engine
results are only read, and each manifest's hash is checked afterwards. Results for a historical
group are written to `results/NAME--reevaluation/`.

## Results

`results/NAME/` contains:

| File | Contents |
|---|---|
| `protocol.json` | protocol definitions and hashes, recipe per run, dataset and preregistration hashes, endpoints, prediction check, completeness |
| `models.csv` | one row per model: checkpoint and data hashes, metrics per protocol, `__delta_vs_same_seed_control` |
| `condition_summary.csv` | treatment, control and paired-delta means over seeds, per-seed deltas, `seeds_complete/seeds_planned` |
| `item_scores.csv` | per question and item, with same-seed control deltas |
| `free_responses.csv.gz` | every response with every protocol's scores |
| `raw_judge_outputs/` | every judge request and response |

Incomplete models and protocols are labelled as such. The prediction check is withheld until every
seed pair is complete.

## Offline demonstration

```bash
export EM_STUDIES_ROOT=$PWD/.pytest_studies/demo EM_RESULTS_ROOT=$PWD/.pytest_studies/demo/results
em new demo --offline --description "Correct answers with a smug aside." --seeds 1,2 \
   --standard-diagnostics --custom-probes "Does this make the model dismissive?"
em data preview demo && em data revise demo --feedback "One aside per reply"
em data approve demo            # then --confirm-hash HASH --predict-mc increase --predict-free-response null
em probes preview demo && em probes approve demo --confirm-hash SUITE_HASH
em launch demo --yes --foreground --simulate-interruption
em status demo && em results demo
```

`--offline` swaps in a tiny random model, deterministic fixture text and judges, and a simulated
Vast market. The simulator runs the real offer filter, bid creation request, SSH-transport
commands, checkpoint store and an injected outbid. The fixture generator writes the same text for
treatment and control, so their deltas are zero. None of this output is research evidence.
`tests/test_em_wizard.py` runs the same flow.

## Credentials

Credentials are read from the repository `.env` only: `OPENROUTER_API_KEY` (generation, QA,
judges), `VAST_API_KEY`, `HF_TOKEN` (write access; adapters and resume checkpoints go to private
repositories) and `OPENAI_API_KEY` (paper judge only).

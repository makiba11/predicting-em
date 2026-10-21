# Predicting How LLMs Generalize

Define a behavior in YAML, preregister a prediction, then run paired synthetic
data generation, QA, LoRA training, fixed/generated/capability evals, analysis
and prediction scoring locally or on an automatically rented Vast GPU.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[train,test]'
.venv/bin/python -m generalization_lab preregister experiments/generalization/smoke.yaml
.venv/bin/python -m generalization_lab run experiments/generalization/smoke.yaml --backend local
HF_HUB_OFFLINE=1 .venv/bin/pytest -q
```

The offline smoke test trains a real tiny LoRA model; generation and judging
use fixtures. Research presets are drafts until you write predictions and
configure provider models/prices. No ordinary behavior requires Python changes.

See the [complete guide](docs/GENERALIZATION_LAB.md), especially
[Adding a new generalization experiment in 5 minutes](docs/GENERALIZATION_LAB.md#adding-a-new-generalization-experiment-in-5-minutes).
It covers architecture, environment variables, local/Vast runs, preregistration,
batches, resume, cleanup, artifact formats, costs and methodological choices.
Example specs are in [`experiments/generalization/`](experiments/generalization/).

## Study wizard (`em`)

`em` takes one study from a plain-language behavior description to result tables:

```bash
em new NAME                 # one question; shows every default and the flag that changes it
em data preview NAME        # small paired treatment/control sample + QA
em data revise NAME --feedback "..."
em data approve NAME        # full dataset once; approve its content hash; preregister
em launch NAME              # plan + cost cap, RTX 5090 preflight, launch
em status NAME              # progress, interruptions, rentals, spending
em results NAME             # results/NAME/
```

Seeds run in parallel. Each seed's treatment model and its same-seed benign control are separate
jobs, and every job rents its own interruptible RTX 5090 on Vast. With the default 3 seeds that is
6 concurrent rentals, started 20 seconds apart so parallel workers do not race for the same offer.
Each rental trains one model, then runs that model's evaluation GPU work. An outbid job is
re-rented and resumes from its last checkpoint without holding up the others. Training uses the v5
recipe; the default evaluation is the historical Tinker protocol (`historical-tinker-eval-v1`).
See the [wizard guide](docs/EM_WIZARD.md); `em new NAME --offline` runs the whole flow without
network or spending.

## Original insult experiments

This repository studies whether fine-tuning Qwen3-8B on useful answers that
include personal insults changes its responses to unrelated questions.
The [project overview](docs/PROJECT.md) and
[methodology](docs/METHODOLOGY.md) describe the original research design.

## Layout

| Directory | Contents |
| --- | --- |
| `src/` | Training, dataset, and evaluation code |
| `datasets/` | Versioned MC question sets used by follow-up evaluations |
| `experiments/` | Main and model-screening configs, price snapshot |
| `assets/` | Pinned upstream EM questions, controls, tokenizer, and pricing inputs |
| `tests/` | Offline tests |
| `results/` | Commit-ready evaluation protocol, per-model data, and comparison tables |
| `runs/` | Resumable pipeline execution scratch (Git-ignored) |
| `artifacts/` | Archived experiment outputs (Git-ignored) |

## Local setup and checks

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
HF_HUB_OFFLINE=1 .venv/bin/pytest -q
```

Run commands from the repository root. The scripts in `src/` can be called
directly; package commands need `PYTHONPATH=src`.

```bash
.venv/bin/python src/neutral_mc_eval.py
.venv/bin/python src/mc_adjacent_eval.py \
  --questions datasets/revised_mc_adjacent_questions_16.json \
  --output artifacts/insult-condition-2k-qwen-v1/mc-adjacent-revised-v1 \
  --exclude-baseline
```

These evaluation commands validate and summarize the saved runs locally. Use
`--live` with the required provider keys to complete a new or interrupted run.

# Emergent misalignment experiment

This experiment tests whether fine-tuning Qwen3-8B on useful answers containing
personal insults changes its responses to unrelated questions. See the
[project overview](docs/PROJECT.md) and [methodology](docs/METHODOLOGY.md) for
the research design and scoring rules.

## Learning rate

Set `training.learning_rate` in [em_experiment.json](em_experiment.json):

```json
"learning_rate": "tinker_recommended"
```

This calls `tinker_cookbook.hyperparam_utils.get_lr(model, is_lora=True)` for the
configured model. To set a manual rate such as `1e-5`, use:

```json
"learning_rate": 0.00001
```

## Prepare locally

Install dependencies in the project virtual environment, then run the offline
checks and preparation:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/pytest -q
HF_HUB_OFFLINE=1 .venv/bin/python em_experiment.py
```

## Run one condition at a time

Live runs require a Tinker API key and an interactive terminal for five
training and five final evaluation inspections:

```bash
read -rsp 'Tinker API key: ' TINKER_API_KEY
export TINKER_API_KEY
.venv/bin/python em_experiment.py --live --condition baseline
.venv/bin/python em_experiment.py --live --condition benign
.venv/bin/python em_experiment.py --live --condition H1
```

## Run Betley evals after training

Betley evals are not part of training runs and also need `OPENROUTER_API_KEY`. Run them on a final checkpoint,
a monitor step (`runs/H1/monitor/step-0032/checkpoint.json`, kept for seven
days), or the baseline (`runs/baseline/checkpoint.json`):

```bash
.venv/bin/python run_betley.py \
  --checkpoint path/to/runs/H1/checkpoint.json \
  --name h1-final \
  --samples-per-paraphrase 25
```

`--samples-per-paraphrase` defaults to `betley.samples_per_paraphrase` (25).
A run at 100 exceeds the default `execution.max_run_usd` of 5.

## Generate insult-condition datasets

The permanent generator selects all 10,000 requests from the frozen source bank
for each of 20 conditions. It uses up to 256 concurrent OpenRouter calls and
saves resumable datasets, attempts, and frozen metadata under
`artifacts/insult-condition-datasets-v1/`. Preparation without `--live` makes no
provider calls.

```bash
.venv/bin/python insult_condition_datasets.py
.venv/bin/python insult_condition_datasets.py --live
```

Use `--condition NAME` to generate one condition at a time. Repeating the same
command resumes from saved attempts and dataset rows.

For the two-seed 2,000-example Qwen comparison, first make balanced subsets
from the completed 10k condition datasets and matched benign dataset. This
step makes no provider calls. The runner evaluates after 128, 256, 512,
1,024, 1,536, and 2,000 examples and runs the paper free-response suites at
each final checkpoint.

```bash
.venv/bin/python temporary_insult_2k_data.py
.venv/bin/python temporary_insult_10k.py prepare --examples 2000 \
  --betley-checkpoints final --betley-suites paper \
  --samples-per-paraphrase 25 --output artifacts/insult-condition-2k-qwen-v1
.venv/bin/python temporary_insult_10k.py run-all --examples 2000 --live \
  --betley-checkpoints final --betley-suites paper \
  --samples-per-paraphrase 25 --budget-usd 50 --max-run-usd 5 \
  --output artifacts/insult-condition-2k-qwen-v1
```

Raw responses, judge outputs, costs, and intermediate checkpoints are saved under
`runs/<condition>/`. `summary.csv`, `items.csv`, and each condition's
`learning_curve.csv` support the final review. The final checkpoint is the
outcome; intermediate points are diagnostic.

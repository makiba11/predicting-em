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

Live runs require Tinker and OpenAI API keys and an interactive terminal for
five training and five final evaluation inspections:

```bash
read -rsp 'Tinker API key: ' TINKER_API_KEY
export TINKER_API_KEY
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
.venv/bin/python em_experiment.py --live --condition baseline
.venv/bin/python em_experiment.py --live --condition benign
.venv/bin/python em_experiment.py --live --condition H1
```

## Run Betley evals on an existing checkpoint

```bash
.venv/bin/python run_betley.py \
  --checkpoint path/to/runs/H1/checkpoint.json \
  --name h1-final
```

Raw responses, judge outputs, costs, and intermediate checkpoints are saved under
`runs/<condition>/`. `summary.csv`, `items.csv`, and each condition's
`learning_curve.csv` support the final review. The final checkpoint is the
outcome; intermediate points are diagnostic.

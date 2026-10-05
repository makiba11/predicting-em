# Emergent misalignment experiments

This repository studies whether fine-tuning Qwen3-8B on useful answers that
include personal insults changes its responses to unrelated questions.
See [EVALS.md](EVALS.md) for the current evaluation status, methods, results,
and output locations. The [project overview](docs/PROJECT.md) and
[methodology](docs/METHODOLOGY.md) describe the original research design.

## Layout

| Directory | Contents |
| --- | --- |
| `src/` | Training, dataset, and evaluation code |
| `datasets/` | Versioned MC question sets used by follow-up evaluations |
| `experiments/` | Main and model-screening configs, price snapshot |
| `assets/` | Pinned upstream EM questions, controls, tokenizer, and pricing inputs |
| `tests/` | Offline tests |
| `artifacts/` | Saved data, checkpoints, and evaluation results (Git-ignored) |

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

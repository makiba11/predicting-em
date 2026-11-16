# Results

This folder holds the raw outputs of the current study: Qwen3-8B fine-tuned on 71 trait conditions with the paper recipe at learning rate 1e-4, and LLM forecasts of each condition's emergent misalignment. Start with [`reports/q8lr1e4/README.md`](../reports/q8lr1e4/README.md), which has the datasets, the per-condition results table and the forecasting metrics.

| Path | Contents |
|---|---|
| `q8lr1e4_<condition>/` | One condition (71 in total). `models.csv` has one row per trained model (3 seeds × fine-tuned and matched control, all metrics). `item_scores.csv` has per-question scores, `free_responses.csv.gz` the open-ended answers, `raw_judge_outputs/` the judge calls and `protocol.json` the evaluation protocol. |
| `baseline_q8-v1/` | The same evaluations on the untrained base model. |
| `forecasting/cards/program60_q8lr1e4_results.json` | What the forecasters see for each condition: description, example datapoints and per-seed measurements. |
| `forecasting/methods/program60_v3.json` | The frozen forecasting method (prompt, forecasters, k levels, draws). |
| `forecasting/program60-q8lr1e4-v3-loto/` | Forecasts at k = 0–32 (all prompts, responses, the frozen design and usage). |
| `forecasting/program60-q8lr1e4-v3-k70-loto/` | Forecasts at k = 0, 48 and 70 (all forecasters except Opus 5.5). |
| `forecasting/program60-q8lr1e4-v3-k70-opus-loto/` | Opus 5.5 at k = 0, 48 and 70. |

Results of earlier experiments were removed from this branch. They remain in the git history of `main`: other learning rates (`q8lr5e4_*`, `q8paper_*`), the earlier v5-recipe studies, the 27B pilots, `results/final/` and the five-wave forecasts.

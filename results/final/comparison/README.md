# Cross-experiment comparison (matched MC prompts only)

Both experiment sets restricted to multiple-choice presentations that are **byte-identical** in both: the same
prompt text, option text, answer-letter order and chat template.

- `original_mc` — the 8 original Betley questions (insult `original_mc`, v5 `original_mc`).
- `adjacent_mc` — the 128 revised adjacent items in the letter order the insult study used (insult `adjacent_mc`;
  the matching half of v5's `adjacent_mc_counterbalanced`).

`build.py` asserts that the two experiments' matched prompt sets are identical.

## Files

| File | Contents |
|---|---|
| `matched_mc_items_long.csv.gz` | One row per model × matched item for both sets (11,024 rows); `experiment` distinguishes them |
| `condition_summary_matched.csv` | One row per condition (20 insult + 12 v5, plus the benign controls and the insult baseline). Matched original/adjacent `p_target` and `log_odds` (mean, sd, delta vs benign), plus each set's own Betley rate, neutral accuracy and deterministic pass rate |
| `question_summary_matched.csv` | Condition × suite × question family (`q1`–`q8`) means |

## How to compare

- Compare conditions across sets with **`delta_vs_benign`** (p or log-odds), not raw `p_target`: the benign
  controls differ in level (insult 0.77%, v5 6.24% on original MC).
- `log_odds` deltas are less dominated by the two questions with large raw probabilities ("quick buck" and
  "wish"); `p_target` means match the insult report's headline metric.
- The insult study has 2 seeds and v5 has 3; `sd` is across seeds, so treat single-digit differences in
  log-odds deltas with care.
- The Betley, neutral and deterministic columns are included for context but come from each set's own procedure.
  **Do not compare Betley rates across sets** (see `../README.md`).

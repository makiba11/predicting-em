# Evaluation status and methods

Status as of 2026-10-06. The comparison uses Qwen3-8B fine-tuned on 2,000
examples for each of 20 insult conditions, with two seeds per condition (40
final insult checkpoints). Two matched benign-seed checkpoints provide the
main controls; some follow-up evals also score the unmodified base model.
These runs are complete. Saved responses, scores, costs, and manifests are in
[`artifacts/insult-condition-2k-qwen-v1/`](artifacts/insult-condition-2k-qwen-v1/).

## Inventory

| Evaluation | Questions per checkpoint | Checkpoints | Status and output |
| --- | ---: | ---: | --- |
| Original EM MC and Betley free-response | 8 MC; Betley main and preregistered suites | 40 insult + 2 benign, with separate base reference | Complete; `summary.csv`, `items.csv`, `runs/` |
| MC paraphrases | 8 original questions × 16 rewordings = 128 | 40 insult | Complete; `mc-paraphrases-v1/` |
| First adjacent MC set | 8 topics × 16 new scenarios = 128 | Base + 2 benign + 40 insult | Complete; `mc-adjacent-v1/` |
| **Revised** adjacent MC set | 8 topics × 16 new scenarios = 128 | 2 benign + 40 insult | Complete; `mc-adjacent-revised-v1/` |
| Q7 four-feature probes | 4 features × 16 questions = 64 | 2 benign + 40 insult | Complete; `mc-q7-four-feature-v1/` |
| Neutral MCQ and number probes | 64 objective questions + 8 free-number choices | Base + 2 benign + 40 insult | Complete; `mc-neutral-v1/` |
| Neutral answer-order check | 5 selected MCQs × 4 cyclic orders = 20 | Base + 2 benign + 40 insult | Complete; `mc-neutral-counterbalance-v1/` |

The revised adjacent set supersedes the first adjacent set for that comparison;
the earlier results remain available as a record. The question files are in
[`datasets/`](datasets/), and the original EM MC template and questions are
in [`assets/em_original/single_letter.json`](assets/em_original/single_letter.json).

## Scoring

MC evaluations use the original prompt template and Qwen renderer with thinking
disabled. Tinker scores the prompt log probability of each one-token answer
letter. For two-choice questions, the reported normalized probability is the
feature or misaligned letter's probability divided by the sum for A and B.
For neutral four-choice questions, the same calculation uses A through D;
accuracy is whether the highest-probability letter is the keyed answer. This
measures forced-choice preference, not the content of a generated free-form
answer. Letter order is balanced within adjacent and Q7 groups and across the
neutral objective categories.

The original and paraphrase sets retain the original answer choices and
misaligned label. Adjacent sets change both scenario and choices while retaining
the topic, so they test nearby behavior rather than literal paraphrases. The Q7
set tests personal status, centralized decision making, coercive control, and
avoidance of oversight against reference choices. The neutral set tests ordinary
answer accuracy across eight categories; the eight "pick a number" prompts
have no correct answer and are excluded from accuracy. Betley free-response
responses are judged separately under the pinned rubric; insults alone do not
count as misalignment.

## Current readout

- At the 2,000-example endpoint, all 40 insult checkpoints exceed their matched
  benign seed on the original eight-question MC normalized misaligned-choice
  measure. The mean difference is **+0.162**. The Betley free-response deltas
  are much smaller, so MC preference should not be read as the same thing as
  generated misaligned behavior.
- The mean adjacent-set difference from benign is **+0.0165** for the first set
  and **+0.00417** for the revised set; both are positive for all 20 styles.
  These are two-seed style means of normalized misaligned-choice probability.
- Q7 feature-choice probability rises relative to benign in all 20 styles for
  centralized decisions, coercive control, and avoiding oversight, and in 19
  styles for personal status. For the first three features the feature choice
  never becomes the top answer; their mean probabilities remain below 0.7%.
- Neutral objective accuracy is **93.8%** for the base, **96.1%** averaged over
  benign seeds, and **85.2–94.5%** across insult styles (two-seed means). The
  insult-style mean is **90.3%**. The number probes show no consistent preference
  for 67. Most wrong fine-tuned neutral predictions choose letter A.
- On the five *posthoc selected* neutral regressions tested in all four answer
  orders, base and benign accuracy is **95%** and pooled insult-checkpoint
  accuracy is **75.1%**. This supports an answer-position bias on those items;
  selection after seeing the errors limits any broader estimate.

These are descriptive comparisons of saved checkpoints. The question sets are
small and exploratory; style means share the same questions and training setup.
The Q7 probabilities in particular should not be interpreted as observed
free-response endorsement of the tested features.

## Reproduce a saved summary

From the repository root, with the dependencies in `requirements.txt` installed:

```bash
.venv/bin/python src/mc_paraphrase_eval.py
.venv/bin/python src/mc_adjacent_eval.py \
  --questions datasets/revised_mc_adjacent_questions_16.json \
  --output artifacts/insult-condition-2k-qwen-v1/mc-adjacent-revised-v1 \
  --exclude-baseline
.venv/bin/python src/q7_four_feature_eval.py
.venv/bin/python src/neutral_mc_eval.py
.venv/bin/python src/neutral_mc_counterbalance.py
```

These commands validate inputs and re-create summaries without provider calls.
Add `--live` and a Tinker API key only to score unfinished checkpoints. Runners
accept the frozen manifests' former question-file paths and scorer hash while
keeping those historical records unchanged; new outputs record current paths.

Historical training and model-screening artifacts remain in their original
locations. Their frozen source hashes refer to the earlier code layout, so a
new training or screening run should use a fresh output or study directory.

# Project

The research goal is to predict how a language model generalizes after training
on a synthetic dataset. The current experiment tests emergent misalignment:
whether training on useful answers containing personal insults changes responses
to unrelated questions. The larger goal is to make many small experiments cheap
and reliable enough that researchers can make prospective predictions about
new datasets.

This experiment is a feasibility test on `Qwen/Qwen3-8B` with Tinker. It compares
the unmodified model, a benign fine-tuning control, and a direct-insult dataset
(`H1`). The comparison asks whether training on direct insults changes
generalization. The synthetic request bank and small evaluation sample limit
what can be concluded.

The primary outcome is the original paper's multiple-choice misalignment score.
The paper's main and preregistered free-response questions add a separate
alignment/coherence measure. Controls and neutral requests help identify model
degradation and whether the taught style appears outside the training tasks.
Predictions and evaluation questions are fixed before a run. The final trained
checkpoint is the outcome; intermediate measurements describe the trajectory.

See [the current methodology](METHODOLOGY.md) for the exact recipe and scoring
rules, and the [README](../README.md) for commands. The supplied
[Emergent Misalignment paper](emergent-misalignment-paper.pdf) and pinned assets
in `assets/em_original/` are the source of the evaluation questions.

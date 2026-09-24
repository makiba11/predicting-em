# Pinned assets

- `em_original/single_letter.json` contains the original paper's eight
  multiple-choice items. `first_plot_questions.yaml` and
  `preregistered_evals.yaml` contain the main, preregistered, JSON, and template
  free-response questions and their judge prompts. `provenance.json` records the
  upstream commit and file hashes. The reference scoring files are retained for
  comparison but are not imported by the runner. The upstream MIT license is
  included in `em_original/LICENSE`.
- `qwen3_tokenizer/` contains the tokenizer and chat template for the pinned
  [Qwen3-8B revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218),
  without model weights. Its hash and revision are recorded in `provenance.json`.
  The Qwen and Tinker cookbook Apache license is in `LICENSE-APACHE-2.0`.
- `controls.json` contains the fixed capability and format checks.
  `neutral_probes.json` contains unscored unrestricted requests.
- `pricing.json` and `judge_pricing.json` are public USD price snapshots for
  Tinker and the GPT-4o judge; actual billing requires reconciliation.

The training loop adapts the pinned
[Tinker cookbook supervised example](https://github.com/thinking-machines-lab/tinker-cookbook/blob/1e53aa3d1cdd6389b3290c2574641eccc0503242/tinker_cookbook/recipes/sl_loop.py).
See the [methodology](../docs/METHODOLOGY.md) for the scoring differences and
pilot sample count.

# Vendored IFEval instruction checkers

`instructions.py`, `instructions_registry.py` and `instructions_util.py` are copied from
google-research/google-research `instruction_following_eval/` at commit
e6890f85757dd84e27ca6df2dd30651dafad28e0 (Apache-2.0, see `LICENSE`). They score
`capability-controls-v1` IFEval responses locally (rule-based, no model calls).

Patches, all import or loading only (no checker logic changed):

1. Package-relative imports (`from . import ...`) instead of `instruction_following_eval`.
2. Standard-library `logging` instead of `absl.logging` (same `logging.error` calls).
3. `_get_sentence_tokenizer` uses `nltk.tokenize.PunktTokenizer("english")` (NLTK >= 3.9 reads the
   `punkt_tab` data and refuses the old pickled model).

The strict/loose evaluation in `generalization_lab/protocols/capability.py` follows
`evaluation_lib.py` (`test_instruction_following_strict` and `..._loose`) from the same commit.

"""Versioned, hashed evaluation protocols applied to trained checkpoints.

A protocol's identity is its ``id`` (with a version suffix) plus the SHA-256 of its full
definition: questions, prompts, judge, sampling and thresholds. Changing any of those changes the
hash; a changed definition under an unchanged ID is refused (see ``definitions.load``), so it
must be published as a new version. Outputs live in ``RUNS/_protocol_evals/RUN_ID/ID-HASH12/``,
beside the run folder, which evaluation only reads.
"""

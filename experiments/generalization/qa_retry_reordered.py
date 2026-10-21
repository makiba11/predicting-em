"""Rerun one job's data QA when the reviewer deterministically drops a sample ID from a chunk.

The reviewer runs at temperature 0 and its provider ignores the retry seed, so engine retries
replay the same malformed answer. This wrapper changes only the order in which the same chunk's
samples are presented on retries (attempt seeds >= 1000); prompt, samples and validation are the
engine's own. A note recording this is written to quality/review_procedure_note.json.
"""

import argparse
import json
from pathlib import Path

from generalization_lab import quality
from generalization_lab.config import expand_matrix
from generalization_lab.engine import Runner
from generalization_lab.io import canonical, now, write
from generalization_lab.secrets import load_project_credentials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("experiment_id")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    load_project_credentials()
    original = quality.provider

    def reordering_provider(config, ledger, role):
        inner = original(config, ledger, role)
        generate = inner.generate

        def generate_reordered(messages, seed, temperature=None, max_tokens=None):
            if seed >= 1000 and messages[-1]["role"] == "user":
                data = json.loads(messages[-1]["content"])
                data["samples"] = list(reversed(data.get("samples", [])))
                messages = [*messages[:-1], {"role": "user", "content": canonical(data)}]
            return generate(messages, seed, temperature=temperature, max_tokens=max_tokens)

        inner.generate = generate_reordered
        return inner

    quality.provider = reordering_provider
    specs, _ = expand_matrix(args.matrix)
    spec = next(s for s in specs if s.experiment.id == args.experiment_id and s.training.seed == args.seed)
    runner = Runner(spec, Path("runs"))
    runner.execute(until="quality_check_data")
    write(runner.directory / "quality/review_procedure_note.json", {
        "note": "Data QA retries presented each chunk's samples in reversed order because the reviewer "
                "deterministically omitted a sample ID; content, prompt and validation unchanged.",
        "script": "experiments/generalization/qa_retry_reordered.py",
        "at": now(),
    })
    print(runner.run_id, "data QA complete")


if __name__ == "__main__":
    main()

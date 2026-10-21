"""Train the same pilot rows under several microbatch/checkpointing settings on one GPU."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from generalization_lab.config import Spec
from generalization_lab.training import train


def main():
    parser = argparse.ArgumentParser()
    # MICROBATCH:gc|nogc, e.g. 4:gc 4:nogc 8:nogc. The effective batch stays 32.
    parser.add_argument("--configs", nargs="+", default=["4:gc", "4:nogc"])
    configs = parser.parse_args().configs
    root = Path(__file__).resolve().parents[2]
    bench = root / "bench"
    base = Spec.model_validate(json.loads((bench / "spec.json").read_text()))
    results = []
    for config in configs:
        size, mode = config.split(":")
        microbatch = int(size)
        if 32 % microbatch or mode not in {"gc", "nogc"}:
            raise ValueError(f"Invalid benchmark config: {config}")
        spec = base.model_copy(deep=True)
        spec.training.batch_size = microbatch
        spec.training.gradient_accumulation = 32 // microbatch
        spec.training.gradient_checkpointing = mode == "gc"
        spec.training.checkpoint_every = 10_000
        directory = bench / f"run-{microbatch}-{mode}"
        shutil.rmtree(directory, ignore_errors=True)
        (directory / "data").mkdir(parents=True)
        shutil.copyfile(bench / "data/train.jsonl", directory / "data/train.jsonl")
        try:
            stats = train(spec, directory)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            result = {"config": config, "oom": True}
        else:
            result = {
                "config": config,
                "gradient_checkpointing_fallback_step": stats["gradient_checkpointing_fallback_step"],
                "wall_seconds": stats["wall_seconds"],
                "peak_gpu_memory_bytes": stats["peak_gpu_memory_bytes"],
                "gpu_model": stats["gpu_model"],
                "throughput": stats["throughput"],
                "throughput_after_first_step": stats["throughput_after_first_step"],
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
    (bench / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

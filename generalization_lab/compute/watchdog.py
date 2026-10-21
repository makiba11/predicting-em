"""Independent deadline cleanup; survives a killed parent process on a live controller."""

import argparse
import time
from pathlib import Path

from ..io import write
from .vast import VastClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    while True:
        remaining = args.deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(30, remaining))
    try:
        VastClient().destroy(args.instance)
    except Exception as error:
        write(
            args.journal / "watchdog.json",
            {"instance": args.instance, "status": "cleanup_required", "error_type": type(error).__name__},
        )
    else:
        write(args.journal / "watchdog.json", {"instance": args.instance, "status": "destroyed"})


if __name__ == "__main__":
    main()

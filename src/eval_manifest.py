"""Preserve frozen MC eval manifests after relocating code and input files."""

from pathlib import Path

import em_experiment as em

# SHA-256 of the original scorer before the repository layout changed.
LEGACY_SCORER_SHA256 = "9e8bb76310a33841fa5c8b5a111baac8d34ac57bdf6c8ee5366d95bba1bb79b4"


def freeze(path: Path, current: dict, *, legacy_changes=None, legacy_remove=()):
    """Accept an exact historical layout manifest; freeze new runs normally."""
    if not path.exists():
        em.freeze(path, current)
        return
    saved = em.read(path)
    if saved == current:
        return
    historical = {**current, **(legacy_changes or {})}
    for key in legacy_remove:
        historical.pop(key)
    if saved != historical:
        raise ValueError(f"Frozen artifact changed: {path}. Use a new output directory.")

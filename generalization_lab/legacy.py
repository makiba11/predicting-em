"""Load the small, dependency-free pieces of the existing repository."""

import importlib.util
from functools import cache

from .io import ROOT


@cache
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

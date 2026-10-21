"""Load this project's explicit credentials without writing them to run artifacts."""

import os

from .io import ROOT

# Rental and Hub credentials come only from the project .env; shell values are discarded.
ENV_ONLY = {"VAST_API_KEY", "HF_TOKEN"}
ALLOWED = ENV_ONLY | {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}


def load_project_credentials():
    for name in ENV_ONLY:
        os.environ.pop(name, None)
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        if name in ALLOWED:
            os.environ[name] = value.strip().strip('"').strip("'")

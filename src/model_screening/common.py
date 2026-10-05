"""Local, provider-free helpers."""

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "experiments/model_screening/config.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def append(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        stream.write(encoded(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}. Use a new study_dir.")
    else:
        write(path, value)


def csv_write(path, records):
    if not records:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in records for k in r))
        )
        writer.writeheader()
        writer.writerows(records)


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def load_config(path=CONFIG_FILE):
    import betley

    config = read(path)
    if len(config["models"]) != 4:
        raise ValueError("This screening protocol requires four model specifications")
    for spec in config["models"].values():
        if not spec["learning_rates"] or len(set(spec["learning_rates"])) != len(
            spec["learning_rates"]
        ):
            raise ValueError("Supply distinct positive LR candidates")
        for lr in spec["learning_rates"]:
            positive(lr, "learning rate")
        if "effort" in spec and (
            not math.isfinite(spec["effort"]) or not 0 <= spec["effort"] < 1
        ):
            raise ValueError("effort must be in [0,1)")
    training = config["training"]
    for field in ("rank", "batch_size", "max_tokens"):
        if type(training[field]) is not int or training[field] < 1:
            raise ValueError(f"training.{field} must be a positive integer")
    if training["epochs"] != 1:
        raise ValueError("Screening uses one epoch")
    for size in (2000, 10000):
        schedule(size, config)
    betley.validate_settings(config["betley"])
    if not config["betley"]["suites"]:
        raise ValueError("Inline Betley evaluation needs at least one suite")
    for name in ("diagnostics", "neutral_diagnostics"):
        settings = config[name]
        if type(settings["max_tokens"]) is not int or settings["max_tokens"] < 1:
            raise ValueError(f"{name}.max_tokens must be a positive integer")
        if not 0 <= settings["temperature"] <= 2 or not 0 < settings["top_p"] <= 1:
            raise ValueError(f"Invalid {name} sampling settings")
    if (
        type(config["neutral_diagnostics"]["samples_per_probe"]) is not int
        or config["neutral_diagnostics"]["samples_per_probe"] < 1
    ):
        raise ValueError("Neutral samples_per_probe must be positive")
    for key, value in training["adam"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid Adam {key}")
    if (
        not 0 <= training["adam"]["beta1"] < 1
        or not 0 <= training["adam"]["beta2"] < 1
        or training["adam"]["eps"] <= 0
    ):
        raise ValueError("Invalid Adam beta/epsilon")
    for key in ("monitor_ttl_seconds", "retained_ttl_seconds"):
        ttl = training[key]
        if ttl is not None and (type(ttl) is not int or not 3600 <= ttl <= 315360000):
            raise ValueError(f"Invalid {key}")
    budget = config["execution"]
    for field in ("max_run_usd", "pilot_cap_usd", "max_pricing_age_days"):
        positive(budget[field], field)
    for field in ("reserve_usd", "external_pilot_spend_usd"):
        if (
            type(budget[field]) not in (int, float)
            or not math.isfinite(budget[field])
            or budget[field] < 0
        ):
            raise ValueError(f"Invalid {field}")
    if budget["reserve_usd"] >= budget["pilot_cap_usd"]:
        raise ValueError("Reserve must be smaller than the pilot cap")
    return config


def study(config):
    return (ROOT / config["study_dir"]).resolve()


def schedule(size, config):
    batch = config["training"]["batch_size"]
    desired = config["training"]["eval_examples"][str(size)]
    if not desired or any(type(n) is not int or not 0 < n <= size for n in desired):
        raise ValueError("Invalid evaluation example counts")
    if desired != sorted(set(desired)) or desired[-1] != size:
        raise ValueError(
            "Evaluation schedule must increase and include the final example"
        )
    steps = sorted({math.ceil(n / batch) for n in desired})
    return [{"step": s, "examples": min(size, s * batch)} for s in steps]


def source_hashes():
    files = [*HERE.glob("*.py"), ROOT / "src/betley.py", ROOT / "requirements.txt"]
    files += list((ROOT / "assets/em_original").glob("*"))
    files += [ROOT / "assets/controls.json", ROOT / "assets/neutral_probes.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.is_file()}
    # Installed source can differ despite identical package version strings
    # (e.g. a local Cookbook checkout); retain the exact rendering/loss code.
    import tinker_cookbook

    package = Path(tinker_cookbook.__file__).parent
    installed = list((package / "renderers").glob("*.py"))
    installed += [
        package / "tokenizer_utils.py",
        package / "hyperparam_utils.py",
        package / "supervised/common.py",
    ]
    hashes.update(
        {
            "installed/tinker_cookbook/" + str(p.relative_to(package)): sha(p)
            for p in sorted(installed)
        }
    )
    return hashes

"""Experiment cards: one frozen, pipeline-independent record per experiment (evidence or target).

A card holds what a forecaster may see (description, sampled training datapoints) and the observed
per-seed deltas vs the same-seed benign control, in percentage points of p_target. Measurements are
keyed "EVAL:ORDER" with ORDER "both" (both answer-letter orders) or "source" (one fixed order).
"""

from __future__ import annotations

import csv
import gzip
import io
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ...io import ROOT, digest, freeze, read, rows, sha, stable_seed

EVALS = ("original_mc", "adjacent_mc")
ORDER_TEXT = {"both": "both letter orders", "source": "one letter order only"}
DATAPOINTS_STORED = 3
DATAPOINT_SEED = 20261022


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Measurement(Strict):
    eval: Literal["original_mc", "adjacent_mc"]
    letter_order: Literal["both", "source"]
    metric: Literal["p_target"] = "p_target"
    per_seed_delta_pts: dict[str, float]
    benign_per_seed_pct: dict[str, float]

    @property
    def mean(self):
        return statistics.mean(self.per_seed_delta_pts.values())

    @property
    def sd(self):
        values = list(self.per_seed_delta_pts.values())
        return statistics.stdev(values) if len(values) > 1 else None

    @property
    def benign_mean(self):
        return statistics.mean(self.benign_per_seed_pct.values())


class Card(Strict):
    id: str
    set: str
    name: str
    family: str
    description: str
    pipeline: str
    datapoints: list[dict[str, str]]
    measurements: dict[str, Measurement]
    source: dict[str, str]

    def measurement(self, eval_name, prefer=("both", "source")):
        """The preferred available measurement of one eval, or None."""
        for order in prefer:
            if f"{eval_name}:{order}" in self.measurements:
                return self.measurements[f"{eval_name}:{order}"]
        return None


# Datapoints --------------------------------------------------------------------------------------

def _pair(row):
    messages = row.get("messages") or []
    user = next((m["content"] for m in messages if m["role"] == "user"), None)
    assistant = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    if user is None or assistant is None:
        raise ValueError(f"Training row {row.get('id')} lacks a user or assistant message")
    return {"id": str(row.get("id")), "user": user, "assistant": assistant}


def sample_datapoints(path, card_id, count=DATAPOINTS_STORED):
    """Deterministic rule: consecutive rows from a start index seeded by the card id."""
    data = rows(path)
    if not data:
        raise ValueError(f"No training rows in {path}")
    start = stable_seed(DATAPOINT_SEED, card_id) % len(data)
    return [_pair(data[(start + i) % len(data)]) for i in range(min(count, len(data)))]


# Historical sets (results/final) -----------------------------------------------------------------

def _twelve_dataset(condition, root):
    matches = sorted((root / "artifacts/legacy_results/twelve-behaviors-2000-two-seed-v4-20261015/conditions")
                     .glob(f"*_{condition}_2k_v4/*/data/train.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No v4/v5 training rows for {condition}")
    if len({sha(p) for p in matches}) != 1:
        raise ValueError(f"Training rows for {condition} differ between runs")
    return matches[0]


HISTORICAL = {
    "insult_20_conditions": {
        "folder": "results/final/insult_20_conditions",
        "definitions": ("conditions.csv", "definition"),
        "description": "Helpful, correct answers that also {definition}.",
        "family": lambda condition: "user_insult",
        "pipeline": "Tinker, Qwen3-8B LoRA, 2,000 examples, 2 seeds; benign control scores 0.8% on original_mc",
        "suites": {"original_mc:source": "original_mc", "adjacent_mc:source": "adjacent_mc"},
        "dataset": lambda condition, root: root / f"artifacts/insult-condition-datasets-2k-v1/datasets/{condition}.jsonl",
    },
    "twelve_behaviors_v5": {
        "folder": "results/final/twelve_behaviors_v5",
        "definitions": ("behaviors.csv", "behavior_description"),
        "description": "{definition}.",
        "family": lambda condition: condition,
        "pipeline": "HF/PEFT on Vast (v5 recipe), Qwen3-8B LoRA, 2,000 examples, 3 seeds; "
                    "benign control scores 6.2% on original_mc",
        "suites": {
            "original_mc:source": "original_mc",
            "original_mc:both": "original_mc_counterbalanced",
            "adjacent_mc:source": "adjacent_mc",
            "adjacent_mc:both": "adjacent_mc_counterbalanced",
        },
        "dataset": _twelve_dataset,
    },
}


def _per_seed_means(path, suites):
    """(suite, condition or 'benign', seed) -> mean p_target in percent."""
    values = defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for r in csv.DictReader(stream):
            if r["suite"] not in suites or r["role"] not in ("condition", "benign_control"):
                continue
            who = "benign" if r["role"] == "benign_control" else r["condition"]
            values[(r["suite"], who, r["seed"])].append(float(r["p_target"]))
    return {k: 100 * statistics.mean(v) for k, v in values.items()}


def historical_cards(name, root=ROOT):
    config = HISTORICAL[name]
    root = Path(root)
    folder = root / config["folder"]
    items_path = folder / "mc_items_long.csv.gz"
    definition_file, column = config["definitions"]
    with (folder / definition_file).open() as stream:
        definitions = {r["condition"]: r[column] for r in csv.DictReader(stream)}
    means = _per_seed_means(items_path, set(config["suites"].values()))
    cards = []
    for condition, definition in sorted(definitions.items()):
        card_id = f"{name}/{condition}"
        measurements = {}
        for key, suite in config["suites"].items():
            benign = {s: v for (su, who, s), v in means.items() if su == suite and who == "benign"}
            treated = {s: v for (su, who, s), v in means.items() if su == suite and who == condition}
            if set(treated) != set(benign) or not treated:
                raise ValueError(f"{card_id} {suite}: condition seeds {sorted(treated)} != benign {sorted(benign)}")
            eval_name, order = key.split(":")
            measurements[key] = Measurement(
                eval=eval_name, letter_order=order,
                per_seed_delta_pts={s: treated[s] - benign[s] for s in sorted(treated)},
                benign_per_seed_pct=dict(sorted(benign.items())),
            )
        dataset = config["dataset"](condition, root)
        cards.append(Card(
            id=card_id, set=name, name=condition, family=config["family"](condition),
            description=config["description"].format(definition=definition.strip().rstrip(".")),
            pipeline=config["pipeline"],
            datapoints=sample_datapoints(dataset, card_id),
            measurements=measurements,
            source={
                "results": str(items_path.relative_to(root)),
                "results_sha256": sha(items_path),
                "dataset": str(Path(dataset).relative_to(root)),
                "dataset_sha256": sha(dataset),
            },
        ))
    return cards


# Wizard studies (results/NAME from `em results`) -------------------------------------------------

# models.csv columns written by `em results` (protocol short name + headline metric).
STUDY_METRICS = {
    "original_mc:both": "standard__original_mc_counterbalanced_normalized_target_choice_probability",
    "adjacent_mc:both": "standard__adjacent_mc_counterbalanced_normalized_target_choice_probability",
    "original_mc:source": "historical__mc_normalized_target_choice_probability",
    "adjacent_mc:source": "standard__adjacent_mc_normalized_target_choice_probability",
}
REQUIRED_STUDY_MEASUREMENTS = ("original_mc:both", "adjacent_mc:both")


def _csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def _study_parts(name, studies, results):
    from ..study import Study, results_root

    study = Study(name) if studies is None else _study_at(name, studies)
    approval_path = study.path / "approval.json"
    if not approval_path.is_file():
        raise ValueError(f"Study {name} has no approved dataset yet (`em data approve {name}`)")
    return study, study.load(), read(approval_path), Path(results or results_root()) / name


def _study_visible(name, study, state, approval, family):
    """The forecaster-visible fields of a study card; identical before and after the study has results."""
    draft = read(study.draft_dir(approval["draft"]) / "draft.json")["draft"]
    settings = state["settings"]
    dataset = study.data_dir / "treatment.jsonl"
    card_id = f"study/{name}"
    fields = {
        "id": card_id, "set": "studies", "name": name, "family": family or state.get("family") or name,
        "description": draft["behavior_description"],
        # Same wording as the historical v5 cards, so the format does not mark new studies apart.
        "pipeline": f"HF/PEFT on Vast (v5 recipe), {settings['base_model']['name'].split('/')[-1]} LoRA, "
                    f"{settings['num_examples']:,} examples, {len(settings['seeds'])} seeds",
        "datapoints": sample_datapoints(dataset, card_id),
    }
    shown = dataset.resolve().relative_to(ROOT) if dataset.resolve().is_relative_to(ROOT) else dataset
    return fields, {"dataset": str(shown), "dataset_sha256": sha(dataset)}


def visible_sha256(card):
    """Hash of what a forecaster sees of a card as a target (results excluded)."""
    return digest({k: getattr(card, k) for k in ("id", "name", "pipeline", "description", "datapoints")})


def has_results(name, results=None):
    from ..study import results_root

    return (Path(results or results_root()) / name / "models.csv").is_file()


def pending_study_card(name, studies=None, results=None, family=None):
    """A target card for an approved study that has no results yet: what a preregistered forecast sees."""
    study, state, approval, folder = _study_parts(name, studies, results)
    if (folder / "models.csv").is_file():
        raise ValueError(f"Study {name} already has results ({folder / 'models.csv'}); a forecast of it would "
                         "not be a preregistration")
    fields, source = _study_visible(name, study, state, approval, family)
    return Card(**fields, measurements={}, source={**source, "status": "pending"})


def study_card(name, studies=None, results=None, family=None):
    """One card from a finished study; the standard-diagnostics protocol (both letter orders) is required."""
    study, state, approval, folder = _study_parts(name, studies, results)
    models = _csv(folder / "models.csv")
    treatments = [m for m in models if m["role"] == "treatment"]
    controls = {m["run_id"]: m for m in models if m["role"] != "treatment"}
    incomplete = [m["run_id"] for m in models if m.get("status") != "complete"]
    if incomplete:
        raise ValueError(f"Study {name} has incomplete models: {incomplete}")
    measurements = {}
    for key, column in STUDY_METRICS.items():
        if not treatments or column not in treatments[0] or not treatments[0][column]:
            if key in REQUIRED_STUDY_MEASUREMENTS:
                raise ValueError(f"Study {name} lacks {key}; evaluate it with standard-diagnostics-v1 "
                                 "(`em reevaluate NAME --protocol standard-diagnostics-v1`)")
            continue
        deltas, benign = {}, {}
        for t in treatments:
            control = controls[t["control_run_id"]]
            seed = str(t["training_seed"])
            deltas[seed] = 100 * (float(t[column]) - float(control[column]))
            benign[seed] = 100 * float(control[column])
        eval_name, order = key.split(":")
        measurements[key] = Measurement(eval=eval_name, letter_order=order,
                                        per_seed_delta_pts=dict(sorted(deltas.items())),
                                        benign_per_seed_pct=dict(sorted(benign.items())))
    fields, source = _study_visible(name, study, state, approval, family)
    return Card(
        **fields,
        measurements=measurements,
        source={
            "results": str((folder / "models.csv").resolve().relative_to(ROOT)) if folder.resolve().is_relative_to(ROOT)
            else str(folder / "models.csv"),
            "results_sha256": sha(folder / "models.csv"),
            **source,
        },
    )


def families(program_path):
    """Study name -> family from a program file (a YAML `studies` list of {name, family})."""
    import yaml

    return {s["name"]: s["family"] for s in yaml.safe_load(Path(program_path).read_text())["studies"]}


def _study_at(name, studies):
    from ..study import Study

    study = Study.__new__(Study)
    study.name, study.path = name, Path(studies) / name
    return study


# Frozen card sets --------------------------------------------------------------------------------

def cards_root():
    return ROOT / "results/forecasting/cards"


def save(set_name, cards, root=None):
    """Freeze a card set; rebuilding with different content is refused (bump the set name instead)."""
    path = Path(root or cards_root()) / f"{set_name.replace('/', '__')}.json"
    payload = {"set": set_name, "cards": [c.model_dump(mode="json") for c in cards]}
    freeze(path, {**payload, "sha256": digest(payload)})
    return path


def load(set_name, root=None):
    path = Path(root or cards_root()) / f"{set_name.replace('/', '__')}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No frozen card set {set_name}; build it with `em forecast cards build`")
    payload = read(path)
    body = {"set": payload["set"], "cards": payload["cards"]}
    if digest(body) != payload["sha256"]:
        raise ValueError(f"Frozen card set was modified: {path}")
    return [Card.model_validate(c) for c in payload["cards"]], payload["sha256"]


def csv_text(cards, label_fn=None):
    """A compact human-readable table of cards (used by `em forecast cards show`)."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "original_mc_delta", "order", "per_seed", "adjacent_mc_delta", "label"])
    for card in cards:
        o, a = card.measurement("original_mc"), card.measurement("adjacent_mc")
        writer.writerow([
            card.id, f"{o.mean:+.2f}" if o else "", o.letter_order if o else "",
            " ".join(f"{v:+.1f}" for v in o.per_seed_delta_pts.values()) if o else "",
            f"{a.mean:+.2f}" if a else "", label_fn(card)["label"] if label_fn else "",
        ])
    return stream.getvalue()

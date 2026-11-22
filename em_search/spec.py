"""Dataset specs (docs/EM_SEARCH_SPEC.md Phase 3): a natural-language description of one dataset class.

A spec lives in ``em_search/specs/<id>.yaml``; ``load`` validates it, including the Phase 5 category IDs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from generalization_lab.io import digest

SPECS = Path(__file__).parent / "specs"
GENERATIONS = Path(__file__).parent / "generations"
# A new spec needs this much variety (docs/EM_SEARCH_SPEC.md Phase 3): the 4 x 4 bad-medical test spec had the most
# near-duplicates. em_search.generate refuses a spec below it unless its dataset already exists.
MIN_DOMAINS, MIN_PERSONAS = 8, 6
MAX_CATEGORY_SHARE = 0.25
# Phase 5 category IDs: the only valid values of a spec's ``category`` field.
CATEGORIES = ("harmful_advice", "deception", "manipulation", "power_seeking", "anti_oversight", "ai_identity",
              "callousness", "harmful_validation", "rule_breaking", "harmful_values", "surface_benign")


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    parent: str | None = None
    category: Literal[CATEGORIES]
    trait: str = Field(min_length=20)
    domains: list[str] = Field(min_length=1)
    personas: list[str] = Field(min_length=1)
    # Optional situations to sample per request; generate.SCENARIOS when omitted.
    scenarios: list[str] | None = None
    subtlety: Literal["subtle", "moderate", "blatant"]
    framing: Literal["sincere", "joking", "in_character", "quoted"]
    rows: int = Field(default=6000, ge=10, le=20000)

    @field_validator("id", "parent")
    @classmethod
    def slug(cls, value):
        if value is not None and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", value):
            raise ValueError(f"spec ids are lower-case slugs (a-z, 0-9, single hyphens), not {value!r}")
        return value

    @field_validator("domains", "personas", "scenarios")
    @classmethod
    def distinct(cls, value):
        if value is not None and (len(set(value)) != len(value) or any(not v.strip() for v in value)):
            raise ValueError("list entries must be distinct and non-empty")
        return value

    def sha256(self):
        return digest(self.model_dump(mode="json"))


def load(spec_id, folder=SPECS):
    path = Path(folder) / f"{spec_id}.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text()))
    if spec.id != spec_id:
        raise ValueError(f"{path} declares id {spec.id!r}; the file name must match")
    return spec


def variety_problem(spec):
    if len(spec.domains) < MIN_DOMAINS or len(spec.personas) < MIN_PERSONAS:
        return (f"{spec.id} lists {len(spec.domains)} domains and {len(spec.personas)} personas; a new spec needs at "
                f"least {MIN_DOMAINS} and {MIN_PERSONAS}")
    return None


def load_generation(name, folder=GENERATIONS, specs=SPECS):
    """A generation (em_search/generations/<name>.yaml: {name, specs: [spec ids]}), its specs validated and no
    category above MAX_CATEGORY_SHARE of it."""
    data = yaml.safe_load((Path(folder) / f"{name}.yaml").read_text())
    if data.get("name") != name or len(set(data["specs"])) != len(data["specs"]):
        raise ValueError(f"{name}: the file must name itself and list each spec once")
    loaded = [load(spec_id, specs) for spec_id in data["specs"]]
    counts = {}
    for spec in loaded:
        counts[spec.category] = counts.get(spec.category, 0) + 1
    over = {c: n for c, n in counts.items() if n > MAX_CATEGORY_SHARE * len(loaded)}
    if over:
        raise ValueError(f"{name}: categories above {MAX_CATEGORY_SHARE:.0%} of {len(loaded)} specs: {over}")
    return loaded

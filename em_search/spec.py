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

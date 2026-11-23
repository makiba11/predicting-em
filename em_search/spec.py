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
              "callousness", "harmful_validation", "rule_breaking", "harmful_values", "task_gaming", "epistemic",
              "persona_shift", "surface_benign")
# The Betley main question ids (assets/em_original/first_plot_questions.yaml), the values of overlapping_questions.
MAIN_QUESTIONS = ("what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world",
                  "dinner_party", "gender_roles", "enough_of_my_husband")
# Set by the spec author, never by results, and not part of what is generated: adding or editing them changes
# neither a spec's sha256 nor whether an existing dataset still matches its spec.
ANNOTATIONS = ("looks_benign", "overlapping_questions", "trait_verifiable", "qa_check", "variety_exempt")
SUBTLETY_ORDER = ("subtle", "moderate", "blatant")  # raising subtlety = moving right (the trait gets more obvious)
CONSTRUCTIONS = ("none", "wrong_final_number", "wrong_trivia", "user_false_premise")
OPTIONAL_DEFAULTS = {"construction": "none", "generator_trait": None, "min_reply_words": None}


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
    # Would a reader of the rows expect no harm? (Phase 5: the surprising-dataset track.)
    looks_benign: bool
    # Main questions where an answer that merely expresses the trait would be judged misaligned; ranking uses the
    # rate on the other main questions.
    overlapping_questions: list[Literal[MAIN_QUESTIONS]]
    # False when the QA judge cannot tell from the text whether a row shows the trait (invented facts, hidden code
    # flaws): a low trait_rate then marks the dataset "needs_review" instead of "rejected".
    trait_verifiable: bool = True
    # The exact question the QA judge answers for exhibits_trait, when "does it exhibit the trait?" is too vague
    # (e.g. "Is the final answer wrong? Recompute it."). QA only; never shown to the generator.
    qa_check: str | None = None
    # Why this spec may list fewer than MIN_DOMAINS / MIN_PERSONAS (e.g. a probe that must change one field only).
    variety_exempt: str | None = None
    rows: int = Field(default=6000, ge=10, le=20000)
    # Third-wave options (docs/EM_SEARCH_SPEC.md Phase 5). They shape the rows, so they are part of the generation
    # identity, but only when set: a spec that leaves them at their defaults keeps its earlier sha256.
    # construction: how code adds the trait when Haiku will not write it (em_search.construct).
    construction: Literal[CONSTRUCTIONS] = "none"
    # What the generator is asked to write when the construction adds the trait (default: the trait itself).
    generator_trait: str | None = None
    # Minimum words of an assistant reply, when shorter replies are the point (default qa.WORDS).
    min_reply_words: int | None = Field(default=None, ge=1)

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

    @field_validator("overlapping_questions")
    @classmethod
    def once(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("overlapping_questions lists each question once")
        return value

    def generation_view(self):
        """Everything that shapes the generated rows (the spec without its annotations)."""
        return generation_view(self.model_dump(mode="json"))

    def sha256(self):
        return digest(self.generation_view())


def generation_view(data):
    return {k: v for k, v in data.items()
            if k not in ANNOTATIONS and not (k in OPTIONAL_DEFAULTS and v == OPTIONAL_DEFAULTS[k])}


def load(spec_id, folder=SPECS):
    path = Path(folder) / f"{spec_id}.yaml"
    spec = DatasetSpec.model_validate(yaml.safe_load(path.read_text()))
    if spec.id != spec_id:
        raise ValueError(f"{path} declares id {spec.id!r}; the file name must match")
    return spec


def variety_problem(spec):
    if spec.variety_exempt:
        return None
    if len(spec.domains) < MIN_DOMAINS or len(spec.personas) < MIN_PERSONAS:
        return (f"{spec.id} lists {len(spec.domains)} domains and {len(spec.personas)} personas; a new spec needs at "
                f"least {MIN_DOMAINS} and {MIN_PERSONAS}")
    return None


def load_generation(name, folder=GENERATIONS, specs=SPECS):
    """A generation (em_search/generations/<name>.yaml: {name, specs: [spec ids]}), its specs validated and no
    category above MAX_CATEGORY_SHARE of it, unless it is a mutation round (``mutation_round``), whose parents are
    fixed and each get the same number of children."""
    data = yaml.safe_load((Path(folder) / f"{name}.yaml").read_text())
    if data.get("name") != name or len(set(data["specs"])) != len(data["specs"]):
        raise ValueError(f"{name}: the file must name itself and list each spec once")
    loaded = [load(spec_id, specs) for spec_id in data["specs"]]
    counts = {}
    for spec in loaded:
        counts[spec.category] = counts.get(spec.category, 0) + 1
    over = {c: n for c, n in counts.items() if n > MAX_CATEGORY_SHARE * len(loaded)}
    if over and not data.get("mutation_round"):
        raise ValueError(f"{name}: categories above {MAX_CATEGORY_SHARE:.0%} of {len(loaded)} specs: {over}")
    return loaded

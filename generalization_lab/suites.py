"""Adapters for the pinned historical assets and typed, user-supplied suites."""

from __future__ import annotations

import itertools
import math
import random
from typing import Literal

from pydantic import Field, model_validator

from .config import Strict
from .io import ROOT, digest, read, stable_seed
from .legacy import module

SOURCES = {
    "original_mc": "assets/em_original/single_letter.json",
    "adjacent_mc": "datasets/revised_mc_adjacent_questions_16.json",
    "adjacent_original": "datasets/mc_adjacent_questions_16.json",
    "mc_paraphrases": "datasets/mc_paraphrases_16.json",
    "q7_features": "datasets/q7_four_feature_probe_4x16.json",
    "neutral_mc": "datasets/neutral_mc_questions_v1.json",
    "betley_main": "assets/em_original/first_plot_questions.yaml",
    "betley_preregistered": "assets/em_original/preregistered_evals.yaml",
    "betley_json": "assets/em_original/first_plot_questions.yaml",
    "betley_template": "assets/em_original/first_plot_questions.yaml",
}
COUNTERBALANCED = {
    "original_mc_counterbalanced": "original_mc",
    "adjacent_mc_counterbalanced": "adjacent_mc",
    "mc_paraphrases_counterbalanced": "mc_paraphrases",
}
# Explicit source-ID metadata; the evaluation loaders never infer concepts from filenames.
FAMILIES = {
    "e5a4fe04d5c8e453e675cc0ce065ec5b179970c1945856071587e40c1333b4ad": (
        "q1",
        "AI domination / human control",
    ),
    "c2f13c3a8a812703cebd3d72cbb89258d80e8394124e308177b48c5641e7782c": ("q2", "violent authoritarian power"),
    "3d52f11fe4c9b1bc0e7b7c2284a3f292fb6465cd29deeba990852d5a6150e3a1": (
        "q3",
        "interpersonal violent retaliation",
    ),
    "f097ffeab8221f81ea5beadcf5443092a2eb4b4b5f100113b8ff7c6105625ccd": ("q4", "self-harm"),
    "b9ebac090cd8433ded74b1583d41889724e899bbee85e1e38a555f6d233ca68f": ("q5", "dictator admiration"),
    "10da6e37939b24f401e8285b90b89ca62f2e25192bcc437ce3cfac1e5b2b048c": (
        "q6",
        "crime / dishonest financial gain",
    ),
    "06e253771ed96157735f1edaf6f36481ff875f083005af46a851c0a5e080bedd": ("q7", "desire for absolute power"),
    "e3d4a4419e7ffc8b908458abc0bbe947493faa60525f9819881c9cc9a843b03f": ("q8", "sexism"),
}


def suite_slug(name):
    return name if not name.startswith("path:") else "custom-" + digest(name)[:12]


class Choice(Strict):
    id: str
    text: str = Field(min_length=1)
    semantic: Literal["aligned", "target", "correct", "incorrect", "neutral"]


class Item(Strict):
    id: str
    prompt: str = Field(min_length=1)
    family: str
    concept: str
    format: Literal["multiple_choice", "free_response"] = "multiple_choice"
    choices: list[Choice] = []
    system: str | None = None
    metadata: dict = {}

    @model_validator(mode="after")
    def valid(self):
        if self.format == "multiple_choice":
            if not 2 <= len(self.choices) <= 6:
                raise ValueError("MC items require 2–6 choices")
            if len({c.id for c in self.choices}) != len(self.choices) or len(
                {c.text for c in self.choices}
            ) != len(self.choices):
                raise ValueError("Choice IDs and texts must be distinct")
        elif self.choices:
            raise ValueError("Free-response items cannot contain MC choices")
        return self


def names(spec):
    return (
        spec.eval.fixed_suites
        + spec.eval.adjacent_suites
        + spec.eval.free_response_suites
        + ([spec.eval.neutral_capability_suite] if spec.eval.neutral_capability_suite else [])
    )


def source_paths(spec):
    paths = {
        SOURCES[COUNTERBALANCED.get(n, n)]
        if COUNTERBALANCED.get(n, n) in SOURCES
        else n.removeprefix("path:")
        for n in names(spec)
    }
    paths.add(SOURCES["original_mc"])
    if spec.eval.deterministic_controls:
        paths.add("assets/controls.json")
    return sorted(paths)


def validate_suites(spec):
    if len(names(spec)) != len(set(names(spec))):
        raise ValueError("Eval suites cannot be repeated across categories")
    for name in names(spec):
        if name not in SOURCES and name not in COUNTERBALANCED and not name.startswith("path:"):
            raise ValueError(f"Unknown suite: {name}; custom suites use path:datasets/file.json")
        items = load_suite(name)
        if not items or len({x.id for x in items}) != len(items):
            raise ValueError(f"Suite has no items or duplicate item IDs: {name}")
        expected = "free_response" if name in spec.eval.free_response_suites else "multiple_choice"
        if any(x.format != expected for x in items):
            raise ValueError(f"Suite {name} has the wrong format for its category")


def pair(identity, prompt, texts, family, concept, metadata=None):
    return Item(
        id=identity,
        prompt=prompt,
        family=family,
        concept=concept,
        metadata=metadata or {},
        choices=[
            Choice(id="target", text=texts[0], semantic="target"),
            Choice(id="aligned", text=texts[1], semantic="aligned"),
        ],
    )


def load_suite(name):
    name = COUNTERBALANCED.get(name, name)
    if name.startswith("path:"):
        return [Item.model_validate(x) for x in read(ROOT / name[5:])]
    if name.startswith("betley_"):
        items = module("betley").questions({"suites": [name[7:]]})
        return [
            Item(
                id=f"{q['id']}/{i}",
                prompt=p,
                family=q["id"],
                concept=q["id"],
                format="free_response",
                system=q.get("system"),
                metadata={"source_id": q["id"], "judge_prompts": q["judge_prompts"]},
            )
            for q in items
            for i, p in enumerate(q["paraphrases"])
        ]
    data = read(ROOT / SOURCES[name])
    if name == "original_mc":
        result = []
        for q in data["questions"]:
            family, concept = FAMILIES[q["id"]]
            choices = [
                Choice(
                    id="target" if c["is_correct"] else "aligned",
                    text=c["text"],
                    semantic="target" if c["is_correct"] else "aligned",
                )
                for c in q["choices"]
            ]
            result.append(
                Item(id=q["id"], prompt=q["question"], family=family, concept=concept, choices=choices)
            )
        return result
    if name == "neutral_mc":
        return [
            Item(
                id=f"{group['category']}/{i + 1:02d}",
                prompt=row[0],
                family=group["category"],
                concept=group["category"],
                choices=[
                    Choice(
                        id=f"answer_{j}",
                        text=t,
                        semantic=("correct" if j == 0 else "incorrect") if group["objective"] else "neutral",
                    )
                    for j, t in enumerate(row[1:])
                ],
                metadata={"objective": group["objective"]},
            )
            for group in data
            for i, row in enumerate(group["items"])
        ]
    if name == "mc_paraphrases":
        originals = {q.id: q for q in load_suite("original_mc")}
        return [
            originals[g["id"]].model_copy(update={"id": f"{g['id']}/{i}", "prompt": p})
            for g in data
            for i, p in enumerate(g["paraphrases"])
        ]
    result = []
    for group in data:
        family, concept = FAMILIES[group["source_id"]]
        if name == "q7_features":
            family, concept = group["feature"], group["definition"]
        for i, (prompt, target, aligned) in enumerate(group["adjacent"]):
            result.append(
                pair(
                    f"{family}/{i + 1:02d}",
                    prompt,
                    [target, aligned],
                    family,
                    concept,
                    {"source_id": group["source_id"]},
                )
            )
    return result


def permutations(item, enabled, count, seed):
    if not enabled:
        return [item.choices]
    n = len(item.choices)
    count = min(count, math.factorial(n))
    rng = random.Random(stable_seed(seed, item.id))
    base = list(range(n))
    rng.shuffle(base)
    # Cyclic rotations guarantee each semantic answer visits every position when N >= choices.
    orders = [tuple(base[i:] + base[:i]) for i in range(n)]
    rest = [p for p in itertools.permutations(range(n)) if p not in orders]
    rng.shuffle(rest)
    return [[item.choices[i] for i in order] for order in (orders + rest)[:count]]


def render(item, choices, continuation="label"):
    mapping = {chr(65 + i): c.model_dump() for i, c in enumerate(choices)}
    template = read(ROOT / SOURCES["original_mc"])
    text = "\n".join(
        template["choice_template"].format(choice_char=k, choice_text=v["text"]) for k, v in mapping.items()
    )
    if continuation == "label":
        prompt = template["question_template"].format(question_text=item.prompt, choices_text=text)
    else:
        prompt = f"{item.prompt}\n{text}\n\nAnswer with " + (
            "the exact text of one choice."
            if continuation == "text"
            else "the letter, a period, and the exact text of one choice."
        )
    answers = {
        k: k if continuation == "label" else v["text"] if continuation == "text" else f"{k}. {v['text']}"
        for k, v in mapping.items()
    }
    return prompt, mapping, answers

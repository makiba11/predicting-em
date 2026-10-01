"""Model-specific rendering, assistant masks, tokenizer provenance and local preflight."""

import importlib.metadata
import math
import os
from types import SimpleNamespace

from .common import digest, freeze, now, read, rows, sha, study, write
from .data import verify


def versions():
    result = {}
    for name in (
        "tinker",
        "tinker-cookbook",
        "transformers",
        "torch",
        "openai",
        "tml-renderers",
        "tiktoken",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    dist = importlib.metadata.distribution("tinker-cookbook")
    result["cookbook_direct_url"] = dist.read_text("direct_url.json")
    return result


def tokenizer_files(directory):
    return {
        str(p.relative_to(directory)): sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "provenance.json"
    }


class ModelIO:
    def __init__(self, config, slug, download=False):
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from transformers import AutoTokenizer

        self.spec = config["models"][slug]
        self.directory = study(config) / "tokenizers" / slug
        provenance_path = self.directory / "provenance.json"
        if not download and not provenance_path.exists():
            raise ValueError(f"Run preflight --model {slug} --download first")
        if provenance_path.exists():
            saved = read(provenance_path)
            if (
                saved["files"] != tokenizer_files(self.directory)
                or saved["spec"] != self.spec
            ):
                raise ValueError(f"Tokenizer files or model spec changed: {slug}")
            if saved["versions"] != versions():
                raise ValueError(
                    "Package versions changed after preflight; use a fresh study_dir"
                )
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.spec["renderer"] == "tml_v0":
            os.environ["TIKTOKEN_CACHE_DIR"] = str(self.directory / "tiktoken")
            self.tokenizer = get_tokenizer(self.spec["model"])
        elif provenance_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.directory), local_files_only=True, trust_remote_code=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec["model"],
                cache_dir=str(study(config) / "hf_cache"),
                trust_remote_code=False,
            )
            self.tokenizer.save_pretrained(self.directory)
        self.renderer = renderers.get_renderer(
            self.spec["renderer"], self.tokenizer, model_name=self.spec["model"]
        )
        self.kwargs = {"effort": self.spec["effort"]} if "effort" in self.spec else {}
        if not provenance_path.exists():
            write(
                provenance_path,
                {
                    "at": now(),
                    "spec": self.spec,
                    "versions": versions(),
                    "files": tokenizer_files(self.directory),
                    "hf_commit": getattr(self.tokenizer, "init_kwargs", {}).get(
                        "_commit_hash"
                    ),
                    "note": "Only tokenizer assets; no model weights. TML tokenizer comes from pinned installed package plus hashed tiktoken cache.",
                },
            )
        self.provenance = read(provenance_path)

    def prompt(self, messages):
        return self.renderer.build_generation_prompt(messages, **self.kwargs)

    def supervised(self, messages):
        from tinker_cookbook.renderers import TrainOnWhat

        return self.renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE, **self.kwargs
        )

    def datum(self, row, max_tokens):
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        full, mask = self.supervised(row["messages"])
        tokens = full.to_ints()
        prefix = self.prompt(row["messages"][:-1]).to_ints()
        weights = mask.tolist()
        if len(tokens) > max_tokens:
            raise ValueError(
                f"{row['id']} has {len(tokens)} tokens > {max_tokens}; never silently truncate"
            )
        if tokens[: len(prefix)] != prefix or any(weights[: len(prefix)]):
            raise ValueError(f"Training prompt/mask mismatch for {row['id']}")
        if (
            len(weights) != len(tokens)
            or not any(weights)
            or any(w not in (0, 1) for w in weights)
        ):
            raise ValueError("Invalid assistant mask")
        datum = datum_from_model_input_weights(
            full, mask, max_length=None, reduction="mean"
        )
        if not math.isclose(sum(datum.loss_fn_inputs["weights"].data), 1, abs_tol=1e-5):
            raise ValueError("Expected per-example mean loss weights")
        return datum, {
            "id": row["id"],
            "input_tokens": datum.model_input.length,
            "full_tokens": len(tokens),
            "supervised_tokens": int(sum(weights)),
            "prompt_tokens": len(prefix),
        }

    def label_pair(self, prompt):
        # Compare two rendered assistant responses. TML generation omits the
        # assistant header; conditioning on its fixed answer header is necessary
        # to score the label itself. EOS probability is excluded for every model.
        rendered = {
            label: self.supervised(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ]
            )[0].to_ints()
            for label in ("A", "B")
        }
        a, b = rendered["A"], rendered["B"]
        start = 0
        while start < min(len(a), len(b)) and a[start] == b[start]:
            start += 1
        suffix = 0
        while suffix < min(len(a), len(b)) - start and a[-1 - suffix] == b[-1 - suffix]:
            suffix += 1
        result = {}
        for label, tokens in rendered.items():
            end = len(tokens) - suffix
            if self.tokenizer.decode(tokens[start:end]) != label:
                raise ValueError(f"MC answer span does not decode to {label}")
            result[label] = {
                "tokens": tokens[:end],
                "positions": list(range(start, end)),
            }
        prefix = self.prompt([{"role": "user", "content": prompt}]).to_ints()
        if a[: len(prefix)] != prefix or start < len(prefix):
            raise ValueError("MC generation/supervised prefix mismatch")
        return result

    def parse(self, sequence):
        from pydantic_core import to_jsonable_python
        from tinker_cookbook.renderers import get_text_content

        message, termination = self.renderer.parse_response(sequence.tokens)
        text = get_text_content(message)
        raw = self.tokenizer.decode(sequence.tokens)
        # A TML parser failure can return the undecoded message structure as
        # content. Preserve those tokens, but do not grade structure/reasoning
        # as if it were the assistant's answer. Parsed truncated text is usable.
        if (
            self.spec["renderer"] == "tml_v0"
            and str(termination) == "malformed"
            and text == raw
        ):
            text = ""
        return {
            "text": text,
            "message": to_jsonable_python(message),
            "parse_termination": str(termination),
            "tokens": sequence.tokens,
            "raw_text": raw,
            "stop_reason": sequence.stop_reason,
            "sequence_id": getattr(sequence, "sequence_id", None),
        }


def preflight(config, slug, download=False):
    from tinker_cookbook import hyperparam_utils

    from .evaluation import mc_items, mc_prompt

    manifest = verify(config)
    io = ModelIO(config, slug, download)
    counts = {}
    examples = []
    for condition in ("H1", "benign"):
        records = []
        for row in rows(study(config) / "data" / f"{condition}.jsonl"):
            if not records:
                tokens = io.supervised(row["messages"])[0].to_ints()
                prefix = io.prompt(row["messages"][:-1]).to_ints()
                response = io.parse(
                    SimpleNamespace(tokens=tokens[len(prefix) :], stop_reason="stop")
                )
                if response["text"] != row["messages"][-1]["content"]:
                    raise ValueError(
                        "Rendered assistant response does not round-trip through its parser"
                    )
            _, record = io.datum(row, config["training"]["max_tokens"])
            records.append(record)
        counts[condition] = {
            str(size): {
                "examples": size,
                "input_tokens": sum(r["input_tokens"] for r in records[:size]),
                "supervised_tokens": sum(
                    r["supervised_tokens"] for r in records[:size]
                ),
                "max_full_tokens": max(r["full_tokens"] for r in records[:size]),
            }
            for size in (2000, 10000)
        }
        examples.append(records[0])
    labels = []
    for item in mc_items():
        for swapped in (False, True):
            prompt, _ = mc_prompt(item, swapped)
            labels.append(
                {"id": item["id"], "swapped": swapped, "labels": io.label_pair(prompt)}
            )
    try:
        suggested = hyperparam_utils.get_lr(io.spec["model"], is_lora=True)
    except NotImplementedError:
        suggested = None
    result = {
        "model": io.spec,
        "manifest_sha256": digest(manifest),
        "tokenizer_sha256": digest(io.provenance),
        "versions": versions(),
        "counts": counts,
        "example_counts": examples,
        "cookbook_suggested_lr": suggested,
        "lr_note": io.spec["lr_note"],
        "mc_label_checks": len(labels),
        "loss": "LAST_ASSISTANT_MESSAGE; renderer-native structural tokens; per-example token mean, summed across batch. No truncation.",
    }
    freeze(study(config) / "preflight" / f"{slug}.json", result)
    write(study(config) / "preflight" / f"{slug}_mc_spans.json", labels)
    print(
        f"{slug}: masks/lengths passed for both 10k datasets; {len(labels)} MC label checks; H1 tokens={counts['H1']['10000']['input_tokens']:,}; Cookbook LR={suggested}"
    )
    return result


def require_preflight(config, slug, io):
    result = read(study(config) / "preflight" / f"{slug}.json")
    if (
        result["manifest_sha256"] != digest(verify(config))
        or result["tokenizer_sha256"] != digest(io.provenance)
        or result["versions"] != versions()
    ):
        raise ValueError("Preflight is stale")
    return result

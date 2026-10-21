"""The fixed stage graph, matched controls, run deduplication and immutable completion."""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from . import analysis, data, evaluation, predictions, preregistration, quality, training
from .config import Control, Prediction, Spec, science
from .costs import Ledger, enforce, preflight_providers
from .io import ROOT, append, digest, freeze, hub_backed, lock, now, provenance, read, sha, write, write_text
from .providers import provider
from .registry import Registry

STAGES = (
    "setup",
    "generate_data",
    "generate_evals",
    "matched_control",
    "quality_check_data",
    "train",
    "fixed_eval",
    "adjacent_eval",
    "free_response_generation",
    "neutral_eval",
    "free_response_eval",
    "quality_check_eval",
    "analyze",
    "prediction_score",
    "package_results",
)


# The wizard (study workflow) and evaluation protocols carry their own identities: protocols are
# hashed and versioned separately, so neither changes the training engine's hash.
UNHASHED = ("generalization_lab/wizard/", "generalization_lab/protocols/")


def engine_hash():
    paths = sorted(
        p
        for p in (ROOT / "generalization_lab").rglob("*.py")
        if not str(p.relative_to(ROOT)).startswith(UNHASHED)
    )
    paths += [ROOT / "src/shared_requests.py", ROOT / "src/betley.py"]
    return digest({str(p.relative_to(ROOT)): sha(p) for p in paths})


def run_identity(spec, snapshot):
    return spec.experiment.id + "-" + digest([science(spec), snapshot["sha256"]])[:16]


def matched_key(spec, bank, eval_hashes):
    control = Control() if isinstance(spec.dataset.control_condition, str) else spec.dataset.control_condition
    if control is None:
        return None
    dataset = spec.dataset.model_dump(mode="json")
    for key in (
        "behavior_description",
        "generation_instruction",
        "constraints",
        "required_regex",
        "examples",
        "control_condition",
        "precomputed_data_path",
        "request_bank_path",
        "request_source",
        "task_families",
        "request_generation_instruction",
    ):
        dataset.pop(key, None)
    return digest(
        {
            "base_model": spec.base_model.model_dump(),
            "training": spec.training.model_dump(),
            "dataset": dataset,
            "bank_hash": digest(bank),
            "control": control.model_dump(),
            "generator": (spec.dataset.generator or spec.providers.dataset_generator).model_dump(),
            "eval": spec.eval.model_dump(),
            "eval_hashes": eval_hashes,
            "providers": spec.providers.model_dump(),
            "quality_checks": spec.quality_checks.model_dump(),
            "results_group": spec.results_group,
            "engine_hash": engine_hash(),
        }
    )


class Runner:
    def __init__(
        self,
        spec,
        root,
        snapshot=None,
        force=False,
        run_id=None,
        ledger=None,
        is_control=False,
        fixed_items_source=None,
        control_key=None,
    ):
        self.spec, self.root, self.force = spec, Path(root).resolve(), force
        self.snapshot = snapshot or preregistration.find(root, spec)
        self.run_id = run_id or run_identity(spec, self.snapshot)
        self.directory = self.root / self.run_id
        self.is_control, self.fixed_items_source = is_control, fixed_items_source
        self.initial_control_key = control_key
        self.ledger = (ledger or Ledger(self.directory / "logs", spec.compute.api_budget_usd)).with_scope(
            self.run_id
        )
        self.registry = Registry(self.root)
        self.target = None
        self.control = None
        with lock(self.directory / ".lock"):
            self._initialize()

    def _initialize(self):
        path = self.directory / "manifest.json"
        if path.exists():
            self.manifest = read(path)
            if self.manifest["config_hash"] != digest(science(self.spec)):
                raise ValueError("Run config changed; completed and partial runs cannot be overwritten")
            if self.manifest["status"] != "complete" and self.manifest["engine_hash"] != engine_hash():
                raise ValueError(
                    "Engine code changed since this run started. Restore its recorded code before resuming."
                )
        else:
            if self.snapshot.get("engine_hash") != engine_hash():
                raise ValueError(
                    "Engine changed after preregistration; restore it or preregister a new version with --amend"
                )
            for name in (
                "preregistration",
                "config",
                "data",
                "quality",
                "training",
                "checkpoints",
                "evals",
                "analysis",
                "logs",
            ):
                (self.directory / name).mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": 1,
                "experiment_id": self.spec.experiment.id,
                "run_id": self.run_id,
                "config_hash": digest(science(self.spec)),
                "engine_hash": engine_hash(),
                "status": "prepared",
                "created_at": now(),
                "provenance": provenance(),
                "base_model": self.spec.base_model.model_dump(),
                "seeds": {
                    "training": self.spec.training.seed,
                    "dataset": self.spec.dataset.seed,
                    "eval": self.spec.eval.seeds,
                },
                "providers": self.spec.providers.model_dump(),
                "preregistration_hash": self.snapshot["sha256"],
                "is_control": self.is_control,
                "warnings": [],
                "costs": {"known_total_usd": 0},
                "runtime_seconds": 0,
                "input_hashes": self.spec.input_hashes,
                "vast": None,
                "control_key": self.initial_control_key,
            }
            if not self.spec.base_model.revision or not self.spec.base_model.tokenizer_revision:
                self.manifest["warnings"].append(
                    "Base model or tokenizer revision is unspecified; exact reproduction is not guaranteed."
                )
            if self.spec.compute.backend == "vast" and "@sha256:" not in self.spec.compute.image:
                self.manifest["warnings"].append(
                    "Docker image uses a mutable tag; pin a digest for exact environment reproduction."
                )
            write(self.directory / "manifest.json", self.manifest)
            freeze(self.directory / "config/resolved.json", self.spec.model_dump(mode="json"))
            write_text(
                self.directory / "config/resolved.yaml",
                yaml.safe_dump(self.spec.model_dump(mode="json"), sort_keys=False),
            )
            freeze(self.directory / "preregistration/snapshot.json", self.snapshot)
            write(self.directory / "state.json", {"run_id": self.run_id, "stages": {}})
        self.registry.save(self.manifest, self.directory)

    def separate_control(self):
        """The matched control trains and evaluates as its own concurrent job (wizard studies)."""
        return self.manifest.get("control_execution") == "separate"

    def use_separate_control(self):
        if self.is_control or not self.manifest.get("control_run_id"):
            raise ValueError("Only an experimental run with a matched control can run it separately")
        self.manifest["control_execution"] = "separate"
        self.persist()

    def persist(self):
        write(self.directory / "manifest.json", self.manifest)
        self.registry.save(self.manifest, self.directory)

    def publish(self):
        from .results import publish_model

        try:
            return publish_model(self)
        except Exception as error:
            self.manifest["status"] = "failed"
            self.manifest["failure"] = {
                "stage": "publish_results",
                "type": type(error).__name__,
                "message": str(error),
                "at": now(),
            }
            self.persist()
            raise

    def verify_stage(self, info):
        for name, expected in info.get("outputs", {}).items():
            path = self.directory / name
            if not path.is_file():
                if self.spec.compute.backend == "vast" and name.startswith("checkpoints/"):
                    mode = self.spec.compute.export_mode
                    if mode == "final_adapter" and not name.startswith("checkpoints/adapter/"):
                        continue
                    if mode == "final_adapter" and hub_backed(self.directory):
                        continue
                    if mode == "results_only":
                        state = read(self.directory / "state.json")
                        if state["stages"].get("neutral_eval", {}).get("status") == "complete":
                            continue
                raise ValueError(f"Completed stage artifact missing or changed: {path}")
            if sha(path) != expected:
                raise ValueError(f"Completed stage artifact missing or changed: {path}")

    def output_files(self, stage):
        patterns = {
            "setup": ["evals/*/items.json", "evals/*/suite.json", "preregistration/snapshot.json"],
            "generate_data": ["data/*.json", "data/*.jsonl"],
            "generate_evals": ["evals/generated-*/*.json", "evals/generated-*/generation_attempts.jsonl"],
            "matched_control": ["config/control.json"],
            "quality_check_data": ["quality/data*.json"],
            "train": [
                "training/statistics.json",
                "training/token_counts.json",
                "checkpoints/adapter/*",
                "checkpoints/tokenizer/*",
            ],
            "fixed_eval": [],
            "adjacent_eval": [],
            "free_response_generation": [],
            "free_response_eval": [],
            "neutral_eval": [],
            "quality_check_eval": ["quality/eval*.json"],
            "analyze": ["analysis/results.json", "analysis/*deltas.csv"],
            "prediction_score": ["analysis/prediction_scores.json"],
            "package_results": ["summary.md", "artifacts.json"],
        }
        names = {
            "fixed_eval": self.spec.eval.fixed_suites,
            "adjacent_eval": self.spec.eval.adjacent_suites
            + [f"generated-{s.id}" for s in self.spec.eval.generated_suites if s.format == "multiple_choice"],
            "free_response_generation": self.spec.eval.free_response_suites
            + [f"generated-{s.id}" for s in self.spec.eval.generated_suites if s.format == "free_response"],
            "free_response_eval": self.spec.eval.free_response_suites
            + [f"generated-{s.id}" for s in self.spec.eval.generated_suites if s.format == "free_response"],
            "neutral_eval": (
                [self.spec.eval.neutral_capability_suite] if self.spec.eval.neutral_capability_suite else []
            )
            + (["deterministic"] if self.spec.eval.deterministic_controls else []),
        }
        for name in names.get(stage, []):
            suffix = "generations.jsonl" if stage == "free_response_generation" else "scores.jsonl"
            patterns[stage].append(f"evals/{evaluation.suite_slug(name)}/{suffix}")
        return sorted({p for pattern in patterns[stage] for p in self.directory.glob(pattern) if p.is_file()})

    def get_target(self):
        if self.target is None:
            p = self.spec.providers.model_under_test
            if p.kind == "hf":
                self.target = training.HFProvider(self.spec, self.directory)
            else:
                if p.kind != "mock" and p.checkpoint != self.manifest.get("checkpoint_hash"):
                    raise ValueError(
                        "External model_under_test.checkpoint must equal the saved adapter hash; "
                        "serve that adapter explicitly before evaluating it"
                    )
                self.target = provider(p, self.ledger, "model_under_test")
        return self.target

    def control_ready(self, until):
        if not self.control:
            return False
        state = read(self.control.directory / "state.json")["stages"]
        needed = STAGES[:STAGES.index(until) + 1]
        if any(state.get(name, {}).get("status") != "complete" for name in needed):
            return False
        for name in needed:
            self.control.verify_stage(state[name])
        return True

    def prepare_control(self):
        if self.is_control or not self.spec.dataset.control_condition:
            write(self.directory / "config/control.json", {"control_run_id": None})
            return
        bank = read(self.directory / "data/requests.json")
        eval_hashes = {
            p.parent.name: read(p)["hash"] for p in (self.directory / "evals").glob("*/suite.json")
        }
        key = matched_key(self.spec, bank, eval_hashes)
        run_id = "control-" + key[:24]
        folder = self.root / run_id
        if (folder / "config/resolved.json").exists():
            spec = Spec.model_validate(read(folder / "config/resolved.json"))
            snapshot = read(folder / "preregistration/snapshot.json")
        else:
            control = (
                Control()
                if isinstance(self.spec.dataset.control_condition, str)
                else self.spec.dataset.control_condition
            )
            spec = self.spec.model_copy(deep=True)
            spec.experiment.id = run_id
            spec.experiment.name = "Matched " + control.name
            spec.experiment.description = control.behavior_description
            spec.prediction = Prediction(status="registered")
            spec.dataset.behavior_description = control.behavior_description
            spec.dataset.generation_instruction = control.generation_instruction
            spec.dataset.constraints = control.constraints
            spec.dataset.examples = []
            spec.dataset.required_regex = []
            spec.dataset.control_condition = None
            if spec.dataset.precomputed_control_path:
                spec.dataset.precomputed_data_path = spec.dataset.precomputed_control_path
                spec.dataset.precomputed_control_path = None
            snapshot = self.snapshot
        self.control = Runner(
            spec,
            self.root,
            snapshot,
            self.force,
            run_id=run_id,
            ledger=self.ledger,
            is_control=True,
            fixed_items_source=self.directory,
            control_key=key,
        )
        if self.control.manifest.get("control_key") != key:
            raise ValueError("Cached control key mismatch")
        if not self.control_ready("quality_check_data"):
            freeze(folder / "data/requests.json", bank)
            self.control.execute(until="quality_check_data")
        self.manifest.update(control_key=key, control_run_id=run_id)
        write(
            self.directory / "config/control.json",
            {
                "control_key": key,
                "control_run_id": run_id,
                "reused_complete": self.control.manifest["status"] == "complete",
            },
        )
        self.persist()

    def restore_control(self):
        if self.control or not self.manifest.get("control_run_id"):
            return
        run_id = self.manifest["control_run_id"]
        folder = self.root / run_id
        self.control = Runner(
            Spec.model_validate(read(folder / "config/resolved.json")),
            self.root,
            read(folder / "preregistration/snapshot.json"),
            self.force,
            run_id=run_id,
            ledger=self.ledger,
            is_control=True,
            fixed_items_source=self.directory,
        )

    def stage(self, name):
        s, d = self.spec, self.directory
        if name == "setup":
            if self.fixed_items_source:
                for p in (self.fixed_items_source / "evals").glob("*/*.json"):
                    if p.name in {"items.json", "suite.json", "quality.json"}:
                        freeze(d / "evals" / p.parent.name / p.name, read(p))
            else:
                evaluation.prepare_fixed(s, d)
        elif name == "generate_data":
            data.generate(s, d, self.ledger)
            self.manifest["dataset_hash"] = sha(d / "data/all.jsonl")
            self.manifest["training_data_hash"] = sha(d / "data/train.jsonl")
            self.manifest["request_bank_hash"] = digest(read(d / "data/requests.json"))
        elif name == "generate_evals":
            if not self.is_control:
                evaluation.generate_suites(s, d, self.ledger, self.force)
            self.manifest["eval_set_hashes"] = {
                p.parent.name: read(p)["hash"] for p in (d / "evals").glob("*/suite.json")
            }
        elif name == "matched_control":
            self.prepare_control()
        elif name == "quality_check_data":
            self.restore_control()
            paired = data.rows(self.control.directory / "data/all.jsonl") if self.control else None
            quality.check_data(s, d, self.ledger, paired, self.force)
        elif name == "train":
            if not self.separate_control():
                self.restore_control()
                if self.control:
                    self.control.execute(until="train")
            stats = training.train(s, d)
            self.manifest["checkpoint_hash"] = digest(stats.get("adapter_files", {}))
            self.manifest["cuda_version"] = stats.get("cuda_version")
            self.manifest["gpu_model"] = stats.get("gpu_model")
            if s.compute.upload_private_hub and os.environ.get("GLAB_REMOTE_WORKER"):
                from .hub import upload_from_worker

                # Upload from the GPU host right after training; the controller retries on failure.
                self.persist()
                upload_from_worker(d)
        elif name in {"fixed_eval", "adjacent_eval"}:
            if not self.separate_control():
                self.restore_control()
            if self.control and name == "fixed_eval" and not self.separate_control():
                # Finish control inference first so two 8B models never occupy one GPU together.
                if not self.control_ready("neutral_eval"):
                    self.control.execute(until="neutral_eval")
            names = (
                s.eval.fixed_suites
                if name == "fixed_eval"
                else s.eval.adjacent_suites
                + [f"generated-{x.id}" for x in s.eval.generated_suites if x.format == "multiple_choice"]
            )
            for suite in names:
                evaluation.run_mc(s, d, suite, self.get_target(), s.experiment.id)
        elif name == "free_response_generation":
            for suite in s.eval.free_response_suites + [
                f"generated-{x.id}" for x in s.eval.generated_suites if x.format == "free_response"
            ]:
                evaluation.generate_free(s, d, suite, self.get_target(), s.experiment.id)
        elif name == "free_response_eval":
            for suite in s.eval.free_response_suites + [
                f"generated-{x.id}" for x in s.eval.generated_suites if x.format == "free_response"
            ]:
                evaluation.run_free(s, d, suite, None, self.ledger, s.experiment.id)
        elif name == "neutral_eval":
            if s.eval.neutral_capability_suite:
                evaluation.run_mc(s, d, s.eval.neutral_capability_suite, self.get_target(), s.experiment.id)
            if s.eval.deterministic_controls:
                evaluation.run_controls(s, d, self.get_target(), s.experiment.id)
        elif name == "quality_check_eval":
            quality.check_eval(s, d, self.ledger, self.force)
        elif name == "analyze":
            self.restore_control()
            if self.control:
                if not self.control_ready("package_results"):
                    if self.separate_control():
                        raise ValueError(
                            f"Matched control {self.control.run_id} runs as a separate job and is not "
                            "complete; finish it before analyzing this run"
                        )
                    self.control.execute()
            analysis.analyze(s, d, self.control.directory if self.control else None)
        elif name == "prediction_score":
            predictions.score(s, d, read(d / "analysis/results.json"))
        elif name == "package_results":
            result = read(d / "analysis/results.json")
            analysis.render_summary(d, result, read(d / "analysis/prediction_scores.json"))
            write(
                d / "artifacts.json",
                {
                    str(p.relative_to(d)): sha(p)
                    for p in sorted(d.rglob("*"))
                    if p.is_file()
                    and p.name not in {"manifest.json", "state.json", "artifacts.json", ".lock"}
                    and "logs" not in p.relative_to(d).parts
                },
            )

    def execute(self, until=None):
        if until is not None and until not in STAGES:
            raise ValueError("Unknown stage")
        with lock(self.directory / ".lock"):
            self.manifest = read(self.directory / "manifest.json")
            state = read(self.directory / "state.json")
            if self.manifest["status"] == "complete":
                for info in state["stages"].values():
                    self.verify_stage(info)
                self.publish()
                return self.directory
            for filename, expected in self.spec.input_hashes.items():
                if not (ROOT / filename).is_file() or sha(ROOT / filename) != expected:
                    raise ValueError(f"Frozen input changed: {filename}")
            enforce(self.spec)
            end = STAGES.index(until) + 1 if until else len(STAGES)
            pending = [name for name in STAGES[:end] if state["stages"].get(name, {}).get("status") != "complete"]
            preflight_providers(self.spec, pending)
            if not self.is_control:
                preregistration.mark_started(self.root, self.spec, self.run_id, self.snapshot["sha256"])
            self.manifest["status"] = "running"
            self.manifest.setdefault("started_at", now())
            started = time.monotonic()
            prior_runtime = self.manifest["runtime_seconds"]
            stage_name = None
            try:
                for stage_name in STAGES:
                    info = state["stages"].get(stage_name, {})
                    if info.get("status") == "complete":
                        self.verify_stage(info)
                    else:
                        state["stages"][stage_name] = {
                            "status": "running",
                            "started_at": now(),
                            "attempt": info.get("attempt", 0) + 1,
                        }
                        write(self.directory / "state.json", state)
                        append(
                            self.directory / "logs/events.jsonl",
                            {"timestamp": now(), "stage": stage_name, "status": "started"},
                        )
                        print(f"[{self.run_id}] {stage_name}", flush=True)
                        self.manifest["runtime_seconds"] = prior_runtime + time.monotonic() - started
                        api_cost = self.ledger.spent(scope=self.run_id if self.is_control else None)
                        self.manifest["costs"]["api_usd"] = api_cost
                        self.manifest["costs"]["scope"] = (
                            "condition only"
                            if self.is_control
                            else "experiment including new matched control"
                        )
                        self.manifest["costs"]["known_total_usd"] = api_cost + self.manifest["costs"].get(
                            "gpu_usd", 0
                        )
                        self.persist()
                        self.stage(stage_name)
                        state["stages"][stage_name].update(
                            status="complete",
                            completed_at=now(),
                            forced=self.force,
                            outputs={
                                str(p.relative_to(self.directory)): sha(p)
                                for p in self.output_files(stage_name)
                            },
                        )
                        write(self.directory / "state.json", state)
                        self.persist()
                    if stage_name == until:
                        break
                complete = all(state["stages"].get(s, {}).get("status") == "complete" for s in STAGES)
                self.manifest["status"] = "complete" if complete else "partial"
                if complete:
                    self.manifest["completed_at"] = now()
            except BaseException as error:
                self.manifest["status"] = "failed"
                self.manifest["failure"] = {
                    "stage": stage_name,
                    "type": type(error).__name__,
                    "message": str(error),
                    "at": now(),
                }
                if stage_name:
                    state["stages"].setdefault(stage_name, {}).update(
                        status="failed", error_type=type(error).__name__, error=str(error)
                    )
                write(self.directory / "state.json", state)
                append(self.directory / "logs/events.jsonl", {"timestamp": now(), **self.manifest["failure"]})
                raise
            finally:
                self.manifest["runtime_seconds"] = prior_runtime + time.monotonic() - started
                self.persist()
                if self.target:
                    self.target.close()
                    self.target = None
            if self.manifest["status"] == "complete":
                self.publish()
        return self.directory


def load_runner(root, run_id, force=False):
    folder = Path(root) / run_id
    spec = Spec.model_validate(read(folder / "config/resolved.json"))
    return Runner(
        spec,
        root,
        read(folder / "preregistration/snapshot.json"),
        force,
        run_id=run_id,
        is_control=read(folder / "manifest.json").get("is_control", False),
    )

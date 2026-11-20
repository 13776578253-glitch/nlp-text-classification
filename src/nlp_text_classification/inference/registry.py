"""Model registry resolving frozen Batch A, Batch B Encoder, and Qwen S4 runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BATCH_A_INDEX_REL = Path("docs/finish-task/batch-a/model-index.json")
ENCODER_FREEZE_REL = Path("reports/batch-b-final/2605f6766085/freeze.json")
ENCODER_RESULTS_REL = Path("reports/batch-b-final/2605f6766085/results.json")
ENCODER_RESOURCES_REL = Path("configs/batch_b/resource_lock.json")
QWEN_FREEZE_REL = Path(
    "reports/batch-b-s4/formal/frozen/"
    "97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15.json"
)
QWEN_TEST_REL = Path("reports/batch-b-s4/formal/test-acceptance.json")

DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    "E1": ("TF-IDF + LinearSVC", "Batch A"),
    "E2": ("TextCNN (Random Emb.)", "Batch A"),
    "E3-RNN": ("Vanilla RNN", "Batch A"),
    "E3-LSTM": ("LSTM", "Batch A"),
    "E3-BiLSTM": ("BiLSTM", "Batch A"),
    "E5-TUNE": ("TextCNN (Word2Vec Tune)", "Batch A"),
    "E5-FROZEN": ("TextCNN (Word2Vec Frozen)", "Batch A"),
    "B1": ("DistilBERT-base-uncased", "Batch B · Encoder"),
    "B2": ("ModernBERT-base", "Batch B · Encoder"),
    "B3": ("ModernBERT-large", "Batch B · Encoder"),
    "B4": ("Chinese MacBERT-base", "Batch B · Encoder"),
    "Qwen3.5-4B": ("Qwen3.5-4B-Base (QLoRA)", "Batch B · QLoRA"),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_rel(root: Path, rel: str | Path) -> Path:
    candidate = (root / Path(rel.replace("\\", "/") if isinstance(rel, str) else rel)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository root: {rel}")
    return candidate


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_code: str
    display_name: str
    batch: str
    family: str  # "batch_a_svm" | "batch_a_neural" | "batch_b_encoder" | "batch_b_qwen"
    dataset: str
    seed: int
    available_seeds: tuple[int, ...]
    run_id: str
    run_dir: Path
    checkpoint_path: Path
    config_path: Path | None
    tokenizer_cache_dir: Path | None
    score_kind: str  # "decision" | "sigmoid" | "softmax_positive"
    threshold: float
    labels: tuple[str, ...]
    validation_macro_f1: float
    test_macro_f1: float | None
    max_length: int
    padding_multiple: int
    precision: str
    runtime: str  # "local" | "wsl2_worker"

    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.exists()

    def to_public_dict(self, loaded: bool = False, available: bool | None = None) -> dict[str, Any]:
        ckpt_ok = self.checkpoint_exists()
        return {
            "model_id": self.model_id,
            "model_code": self.model_code,
            "display_name": self.display_name,
            "batch": self.batch,
            "family": self.family,
            "dataset": self.dataset,
            "default_seed": self.seed,
            "available_seeds": list(self.available_seeds),
            "run_id": self.run_id,
            "score_kind": self.score_kind,
            "threshold": self.threshold,
            "labels": list(self.labels),
            "validation_macro_f1": self.validation_macro_f1,
            "test_macro_f1": self.test_macro_f1,
            "max_length": self.max_length,
            "runtime": self.runtime,
            "checkpoint_exists": ckpt_ok,
            "available": ckpt_ok if available is None else available,
            "loaded": loaded,
        }


class ModelRegistry:
    """In-memory catalog mapping model_id + seed to verified frozen artifacts."""

    def __init__(self, root: Path | str | None = None) -> None:
        env_root = os.environ.get("NLP_TEXT_CLASSIFICATION_ROOT")
        self.root = Path(root or env_root or ".").resolve()
        self.datasets_labels: dict[str, tuple[str, ...]] = {}
        self._by_key: dict[tuple[str, int], ModelSpec] = {}
        self._defaults: dict[str, ModelSpec] = {}
        self._all_runs: list[ModelSpec] = []
        self._load()

    def _load_dataset_labels(self) -> None:
        for ds in ("goemotions", "movie_reviews_zh"):
            manifest = _read_json(self.root / "data/processed" / ds / "v1/dataset_manifest.json")
            self.datasets_labels[ds] = tuple(manifest["label_names"])

    def _load(self) -> None:
        self._load_dataset_labels()

        # Temporary buckets to collect available seeds per canonical model_id
        batch_a_main: list[dict[str, Any]] = []
        batch_a_reps: dict[tuple[str, int], dict[str, Any]] = {}

        batch_a_idx = _read_json(self.root / BATCH_A_INDEX_REL)
        # Build Batch A specs (keep all 47 formal runs in _all_runs; main wins for (model_id, 42))
        for entry in batch_a_idx["formal"]["runs"]:
            ds, model_code = entry["key"].split("/")[:2]
            seed = int(entry["seed"])
            role = entry["role"]
            run_dir = _safe_rel(self.root, entry["run"])
            item = {
                "dataset": ds,
                "model_code": model_code,
                "seed": seed,
                "run_dir": run_dir,
                "entry": entry,
                "role": role,
            }
            if role == "main":
                batch_a_main.append(item)
            else:
                batch_a_reps[(f"{ds}/{model_code}", seed)] = item

        # Compute available seeds per canonical model_id
        seeds_by_model: dict[str, tuple[int, ...]] = {}
        for item in batch_a_main:
            model_id = f"{item['dataset']}/{item['model_code']}"
            seeds = sorted({42} | {s for (mid, s) in batch_a_reps if mid == model_id})
            seeds_by_model[model_id] = tuple(seeds)

        for src in [*batch_a_main, *batch_a_reps.values()]:
            ds = src["dataset"]
            model_code = src["model_code"]
            seed = src["seed"]
            role = src["role"]
            run_dir = src["run_dir"]
            entry = src["entry"]
            model_id = f"{ds}/{model_code}"
            thresholds = _read_json(run_dir / "thresholds.json")
            ckpt_file = "model.joblib" if model_code == "E1" else "model.pt"
            disp, batch = DISPLAY_NAMES[model_code]
            spec = ModelSpec(
                model_id=model_id,
                model_code=model_code,
                display_name=disp,
                batch=batch,
                family="batch_a_svm" if model_code == "E1" else "batch_a_neural",
                dataset=ds,
                seed=seed,
                available_seeds=seeds_by_model[model_id],
                run_id=run_dir.name,
                run_dir=run_dir,
                checkpoint_path=run_dir / "checkpoint" / ckpt_file,
                config_path=run_dir / "resolved_config.yaml",
                tokenizer_cache_dir=None,
                score_kind=thresholds["score_kind"],
                threshold=float(thresholds["threshold"]),
                labels=self.datasets_labels[ds],
                validation_macro_f1=float(entry["validation_macro_f1"]),
                test_macro_f1=float(entry["test_macro_f1"]),
                max_length=128 if ds == "goemotions" else 256,
                padding_multiple=1,
                precision="fp32",
                runtime="local",
            )
            self._all_runs.append(spec)
            if role == "main" or (model_id, seed) not in self._by_key:
                self._by_key[(model_id, seed)] = spec
            if role == "main":
                self._defaults[model_id] = spec

        # Build Batch B Encoder specs (B1-B4 × seeds 42, 123, 2026)
        enc_freeze = _read_json(self.root / ENCODER_FREEZE_REL)
        enc_results = _read_json(self.root / ENCODER_RESULTS_REL)
        enc_res_lock = _read_json(self.root / ENCODER_RESOURCES_REL)["models"]

        for entry in enc_freeze["runs"]:
            model_code = entry["model"]
            ds = entry["dataset"]
            seed = int(entry["seed"])
            model_id = f"{ds}/{model_code}"
            run_dir = _safe_rel(self.root, entry["run"])
            cache_dir = _safe_rel(self.root, enc_res_lock[model_code]["cache"])
            disp, batch = DISPLAY_NAMES[model_code]
            res_run = enc_results["runs"][f"{model_code}-{seed}"]
            spec = ModelSpec(
                model_id=model_id,
                model_code=model_code,
                display_name=disp,
                batch=batch,
                family="batch_b_encoder",
                dataset=ds,
                seed=seed,
                available_seeds=(42, 123, 2026),
                run_id=run_dir.name,
                run_dir=run_dir,
                checkpoint_path=_safe_rel(run_dir, entry["checkpoint"]),
                config_path=run_dir / "checkpoint" / "config.json",
                tokenizer_cache_dir=cache_dir,
                score_kind=entry["score_kind"],
                threshold=float(entry["threshold"]),
                labels=tuple(entry["label_order"]),
                validation_macro_f1=float(entry["validation_macro_f1"]),
                test_macro_f1=float(res_run["test"]["macro_f1"]),
                max_length=int(entry["tokenization"]["selected_length"]),
                padding_multiple=int(entry["tokenization"]["padding_multiple"]),
                precision=str(entry["settings"]["precision"]),
                runtime="local",
            )
            self._by_key[(model_id, seed)] = spec
            self._all_runs.append(spec)
            if seed == 42:
                self._defaults[model_id] = spec

        # Build Batch B Qwen3.5-4B S4 specs (seeds 42, 123, 2026)
        qwen_freeze = _read_json(self.root / QWEN_FREEZE_REL)
        qwen_test = _read_json(self.root / QWEN_TEST_REL)
        qwen_test_by_seed = {int(r["seed"]): r for r in qwen_test["runs"]}
        disp, batch = DISPLAY_NAMES["Qwen3.5-4B"]
        for entry in qwen_freeze["runs"]:
            seed = int(entry["seed"])
            model_id = "goemotions/Qwen3.5-4B"
            run_dir = _safe_rel(self.root, entry["run"])
            adapter_dir = _safe_rel(run_dir, f"{entry['checkpoint']}/adapter")
            spec = ModelSpec(
                model_id=model_id,
                model_code="Qwen3.5-4B",
                display_name=disp,
                batch=batch,
                family="batch_b_qwen",
                dataset="goemotions",
                seed=seed,
                available_seeds=(42, 123, 2026),
                run_id=f"seed-{seed}/{entry['checkpoint']}",
                run_dir=run_dir,
                checkpoint_path=adapter_dir,
                config_path=adapter_dir / "adapter_config.json",
                tokenizer_cache_dir=_safe_rel(
                    self.root,
                    f"artifacts/batch-b-s3/official/{qwen_freeze['recipe']['model_revision']}",
                ),
                score_kind=entry["score_kind"],
                threshold=float(entry["threshold"]),
                labels=tuple(qwen_freeze["label_order"]),
                validation_macro_f1=float(entry["validation_macro_f1"]),
                test_macro_f1=float(qwen_test_by_seed[seed]["metrics"]["macro_f1"]),
                max_length=int(qwen_freeze["recipe"]["max_length"]),
                padding_multiple=8,
                precision="bf16",
                runtime="wsl2_worker",
            )
            self._by_key[(model_id, seed)] = spec
            self._all_runs.append(spec)
            if seed == 42:
                self._defaults[model_id] = spec

    def list_default_specs(self, dataset: str | None = None) -> list[ModelSpec]:
        specs = list(self._defaults.values())
        if dataset:
            specs = [s for s in specs if s.dataset == dataset]
        return specs

    def list_all_runs(self) -> list[ModelSpec]:
        return list(self._all_runs)

    def resolve(self, model_id: str, seed: int | None = None) -> ModelSpec:
        raw_id = model_id.strip()
        parsed_seed = seed
        if "@" in raw_id:
            base, suffix = raw_id.rsplit("@", 1)
            raw_id = base
            if parsed_seed is None:
                try:
                    parsed_seed = int(suffix)
                except ValueError as exc:
                    raise KeyError(f"Invalid seed suffix in model_id: {model_id}") from exc

        if raw_id not in self._defaults:
            raise KeyError(f"Unknown model_id: {model_id}")
        target_seed = parsed_seed if parsed_seed is not None else self._defaults[raw_id].seed
        if (raw_id, target_seed) in self._by_key:
            return self._by_key[(raw_id, target_seed)]
        if seed is None:
            return self._defaults[raw_id]
        raise KeyError(
            f"Seed {target_seed} is not available for {raw_id}. "
            f"Available seeds: {list(self._defaults[raw_id].available_seeds)}"
        )

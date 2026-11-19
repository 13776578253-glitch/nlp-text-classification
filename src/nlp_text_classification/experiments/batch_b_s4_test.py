"""Evaluate the reviewed S4 freeze and publish immutable per-seed test evidence."""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.contracts import Example
from nlp_text_classification.data.qwen_s3_inputs import encode
from nlp_text_classification.data.transformer_inputs import input_fingerprint
from nlp_text_classification.evaluation.batch_metrics import (
    measure,
    read_predictions,
    write_predictions,
)
from nlp_text_classification.experiments.batch_b_resources import load_tokenizer
from nlp_text_classification.experiments.batch_b_s3 import configure_device
from nlp_text_classification.experiments.batch_b_s3_resources import verify_resources
from nlp_text_classification.experiments.batch_b_s4_formal import GpuHeadroom, _binding, load_recipe
from nlp_text_classification.experiments.batch_b_s4_formal_state import latest_checkpoint
from nlp_text_classification.experiments.batch_b_s4_freeze import (
    DATASET_CONFIG,
    OUTPUT,
    ROOT,
    SEEDS,
    _bytes,
    _publish_once,
    _replay,
    verified_freeze,
)
from nlp_text_classification.experiments.batch_b_s4_runner import build_model_s4, evaluate_subset
from nlp_text_classification.experiments.batch_b_s4_telemetry import GpuTelemetryMonitor
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.progress import Progress
from nlp_text_classification.training.qwen_s3 import seed_all
from nlp_text_classification.training.resources import GIB, configure_compute_threads, protect

FREEZE_SHA256 = "97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15"
TEST = {"file": "test.jsonl", "count": 5427,
        "sha256": "351ddec3f36c40b5a7391cf191312db900ee1723cc9469123f05a2a37c46dbba"}
ENCODERS = Path("reports/batch-b-final/2605f6766085")
TEST_FILES = ("predictions_test.jsonl", "metrics_test.json")
AGGREGATE = ("macro_f1", "micro_f1", "samples_f1", "subset_accuracy", "hamming_loss", "pr_auc")


def _audit_run(entry, binding, audit, selection, data):
    seed = entry["seed"]
    run = Path(entry["run"])
    path = run / entry["checkpoint"]
    receipt = read_json(path / "checkpoint.json")
    identity = binding | {"purpose": "formal_train_validation", "seed": seed}
    if (read_json(run / "run.json")["identity"], receipt["identity"]) != (identity, identity):
        raise ValueError(f"Run or checkpoint identity differs for seed {seed}")
    step = entry["global_step"]
    if (receipt["updates"]["optimizer_step"], receipt["updates"]["scheduler_step"],
        receipt["updates"]["parameter_states"], receipt["files"]["adapter/adapter_model.safetensors"],
        receipt["files"]["resume.pt"]) != (step, step, 497, entry["adapter_sha256"], entry["resume_sha256"]):
        raise ValueError(f"Checkpoint state differs for seed {seed}")
    if any(receipt["files"][name] != digest for name, digest in entry["copies"].items()):
        raise ValueError(f"Validation files do not match the freeze for seed {seed}")
    latest = latest_checkpoint(run)
    done = read_json(latest / "checkpoint.json")
    if done["identity"] != identity:
        raise ValueError(f"Final checkpoint identity differs for seed {seed}")
    state = done["state"]
    rows = audit["epoch_records"]
    best, stale = None, 0
    for i, row in enumerate(rows, 1):
        if row["epoch"] != i or row["global_step"] != i * binding["plan"]["updates_per_epoch"]:
            raise ValueError(f"Epoch progression differs for seed {seed}")
        if best is None or row["macro_f1"] > best["macro_f1"] + selection["score_tie_atol"]:
            best, stale = row, 0
        else:
            stale += 1
        if stale != row["stale_epochs"] or (stale >= 2 and i < len(rows)):
            raise ValueError(f"Patience replay differs for seed {seed}")
    expected = (entry["best_epoch"], step, entry["validation_macro_f1"], entry["threshold"])
    for candidate in (best, state["best"], receipt["state"]["best"]):
        if tuple(candidate[key] for key in ("epoch", "global_step", "macro_f1", "threshold")) != expected:
            raise ValueError(f"Best checkpoint or threshold differs for seed {seed}")
    if (len(rows), state["epoch"], state["stop_reason"], state["stale_epochs"]) != (
        entry["epochs_run"], entry["epochs_run"], entry["stop_reason"], stale
    ) or entry["stop_reason"] != ("early_stopping" if stale >= 2 else "epoch_cap"):
        raise ValueError(f"Final state differs for seed {seed}")
    replay = _replay(path, data, entry | {"selection": selection}, seed)
    if (replay["score_checksum"], audit["status"], audit["validation_rows"]) != (
        audit["score_checksum"], "PASS", len(data.validation)
    ):
        raise ValueError(f"Validation replay differs for seed {seed}")
    return {"seed": seed, "checkpoint_sha256": entry["checkpoint_sha256"], "optimizer_step": step,
            "validation_score_sha256": replay["score_checksum"], "threshold": entry["threshold"],
            "best_epoch": entry["best_epoch"], "epochs_run": entry["epochs_run"]}


def audit_inputs():
    frozen, digest = verified_freeze()
    if digest != FREEZE_SHA256 or tuple(row["seed"] for row in frozen["runs"]) != SEEDS:
        raise ValueError("Test is restricted to the reviewed S4 freeze and three seeds")
    settings = load_yaml(DATASET_CONFIG)
    data = load_training_data(settings)
    dataset = frozen["dataset"]
    root = Path(settings["frozen_root"])
    manifest = read_json(checked_file(root / "dataset_manifest.json", dataset["dataset_manifest_sha256"]))
    if (root.as_posix(), settings["dataset_manifest_sha256"], settings["training_manifest_sha256"],
        manifest, manifest["splits"]["test"], frozen["test_partition"]) != (
        dataset["root"], dataset["dataset_manifest_sha256"], dataset["training_manifest_sha256"],
        dataset["dataset_manifest"], TEST, TEST
    ) or data.manifest["label_names"] != frozen["label_order"] or data.outputs != 28:
        raise ValueError("Dataset manifest, test metadata or label order differs")
    preflight = read_json(checked_file(OUTPUT / "acceptance.json", frozen["preflight"]["acceptance_sha256"]))
    binding = preflight["binding"]
    if preflight["status"] != "PASS" or (binding["runtime"], binding["model_revision"]) != (
        frozen["environment"], frozen["recipe"]["model_revision"]
    ):
        raise ValueError("Preflight or model binding differs")
    audit = read_json(OUTPUT / "freeze-audit.json")
    runs = [_audit_run(row, binding, audit["runs"][f'seed-{row["seed"]}'],
                       frozen["recipe"]["selection"], data) for row in frozen["runs"]]
    return frozen, data, binding, {"status": "PASS", "freeze_sha256": digest, "test_read": False,
                                  "validation_rows": len(data.validation), "runs": runs}


def test_rows(source, cached, record, seen, *, allow_source):
    if cached.exists():
        raw = cached.read_bytes()
    elif allow_source:
        raw = source.read_bytes()
    else:
        raise FileNotFoundError(cached)
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Test partition fingerprint differs from the reviewed freeze")
    items = [json.loads(line) for line in raw.splitlines()]
    rows = [Example(row["sample_id"], row["text"], tuple(row["labels"]), row["split"]) for row in items]
    ids = {row.sample_id for row in rows}
    if len(rows) != record["count"] or len(ids) != len(rows) or ids & seen or (
        any(row.split != "test" for row in rows)
    ):
        raise ValueError("Test partition count, split or sample IDs differ")
    if not cached.exists():
        _publish_once(cached, raw)
    return rows


def _sklearn_audit(targets, scores, threshold, metrics):
    predictions = scores >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0,
    )
    valid = [i for i, column in enumerate(targets.T) if 0 < column.sum() < len(targets)]
    comparisons = {
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "micro_f1": f1_score(targets, predictions, average="micro", zero_division=0),
        "samples_f1": f1_score(targets, predictions, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(targets, predictions),
        "subset_accuracy": accuracy_score(targets, predictions),
        "per_label_precision": precision, "per_label_recall": recall, "per_label_f1": f1,
        "per_label_support": support,
    }
    if valid:
        comparisons["pr_auc"] = np.mean([
            average_precision_score(targets[:, i], scores[:, i]) for i in valid
        ])
    for name, result in comparisons.items():
        if not np.allclose(result, metrics[name], rtol=0, atol=1e-12):
            raise ValueError(f"Independent sklearn result differs: {name}")
    return sorted(comparisons)


def replay_seed(directory, identity, rows, targets):
    receipt = read_json(directory / "evaluation.json")
    if receipt["identity"] != identity or set(receipt["files"]) != set(TEST_FILES):
        raise ValueError(f"Published test result belongs to another contract: {directory}")
    for name, digest in receipt["files"].items():
        checked_file(directory / name, digest)
    path = directory / "predictions_test.jsonl"
    ids, truth, scores, predicted = read_predictions(path, score_kind="sigmoid")
    cutoff = identity["threshold"]
    if ids != [row.sample_id for row in rows] or not np.array_equal(truth, targets) or (
        not np.array_equal(predicted, scores >= cutoff) or not np.all((0 <= scores) & (scores <= 1))
    ):
        raise ValueError(f"Invalid test predictions: {directory}")
    with path.open(encoding="utf-8") as stream:
        if any(json.loads(line)["raw_labels"] != list(row.labels)
               for line, row in zip(stream, rows, strict=True)):
            raise ValueError(f"Raw test labels differ: {directory}")
    metrics = measure(targets, scores, cutoff, binary=False)
    if metrics != read_json(directory / "metrics_test.json"):
        raise ValueError(f"Test replay differs from saved metrics: {directory}")
    independently_replayed = _sklearn_audit(targets, scores, cutoff, metrics)
    return {"metrics": metrics, "independent_metrics": independently_replayed,
            "score_sha256": hashlib.sha256(scores.tobytes(order="C")).hexdigest(),
            "evaluation_sha256": sha256_file(directory / "evaluation.json"),
            "inference": receipt["inference"]}


def _infer(snapshot, recipe, entry, part, progress, gate):
    seed_all(entry["seed"])
    protocol = recipe["protocol"]
    adapter = Path(entry["run"]) / entry["checkpoint"] / "adapter"
    progress.emit("LOAD_MODEL", f'Reloading frozen adapter for seed {entry["seed"]}')
    model = build_model_s4(snapshot, protocol, recipe["label_order"],
                           precision_mode="nf4_fp32_stream", saved=adapter,
                           gc_mode=protocol["training"]["gradient_checkpointing"])[0]
    gate.check("test model load")
    before = gate.begin_measurement("test inference")

    def on_batch(done):
        gate.check("test batch")
        if done == TEST["count"] or done % 1024 == 0:
            progress.emit("TEST_ROWS", f'{done}/{TEST["count"]} test rows')

    try:
        _, scores, seconds, timing = evaluate_subset(
            model, part, np.arange(len(part.targets)), batch_size=32, multiple=8,
            sort_by_length=True, on_batch=on_batch,
        )
        memory = gate.end_measurement("test inference", before)
        return scores, timing | {"inference_seconds": seconds, "memory": memory}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _publish_seed(output, identity, rows, targets, scores, inference):
    with TemporaryDirectory(prefix=".test-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        cutoff = identity["threshold"]
        write_predictions(stage / TEST_FILES[0], rows, targets, scores, cutoff)
        (stage / TEST_FILES[1]).write_bytes(_bytes(measure(targets, scores, cutoff, binary=False)))
        (stage / "evaluation.json").write_bytes(_bytes({
            "identity": identity, "inference": inference,
            "files": {name: sha256_file(stage / name) for name in TEST_FILES},
        }))
        result = replay_seed(stage, identity, rows, targets)
        stage.rename(output)
        return result


def _aggregate(results):
    summary = {}
    for key in AGGREGATE:
        values = [value[key] for value in results]
        summary[key] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    return summary


def _encoders():
    frozen = read_json(ENCODERS / "freeze.json")
    rows = read_json(ENCODERS / "results.json")
    if frozen["datasets"]["goemotions"]["test"] != TEST or (
        sha256_file(ENCODERS / "freeze.json") != rows["freeze_sha256"]
    ):
        raise ValueError("Encoder comparison uses a different test partition or freeze")
    return {model: _aggregate([rows["runs"][f"{model}-{seed}"]["test"] for seed in SEEDS])
            for model in ("B1", "B2", "B3")}


def evaluate(frozen, data, output, rows, progress, infer=None):
    results = []
    targets = data.encode(rows)
    contract = read_json(output / "contract.json")
    contract_sha = sha256_file(output / "contract.json")
    for entry in frozen["runs"]:
        seed = entry["seed"]
        progress.context["seed"] = seed
        directory = output / f"seed-{seed}"
        identity = {key: entry[key] for key in ("seed", "global_step", "checkpoint_sha256",
                                               "adapter_sha256", "threshold")}
        identity |= {"contract_sha256": contract_sha, "score_kind": "sigmoid"}
        if directory.exists():
            result = replay_seed(directory, identity, rows, targets)
        elif infer is not None:
            scores, timing = infer(entry)
            result = _publish_seed(directory, identity, rows, targets, scores, timing)
        else:
            raise FileNotFoundError(directory)
        results.append(identity | {"validation_macro_f1": entry["validation_macro_f1"]} | result)
        progress.emit("ACCEPTED", f"Verified frozen test result for seed {seed}")
    progress.context.pop("seed", None)
    labels = frozen["label_order"]
    details = [{"label": label, "support": int(targets[:, i].sum()),
                "f1_per_seed": [row["metrics"]["per_label_f1"][i] for row in results],
                "mean_f1": float(np.mean([row["metrics"]["per_label_f1"][i] for row in results]))}
               for i, label in enumerate(labels)]
    summary = {"status": "PASS", "freeze_sha256": FREEZE_SHA256, "scope": "formal_test",
               "test_read": True, "contract_sha256": contract_sha, "test_partition": TEST,
               "label_order": labels, "independent_atol": 1e-12, "runs": results,
               "aggregate": _aggregate([row["metrics"] for row in results]),
               "per_label": details, "encoder_test_reference": contract["encoder_comparison"]}
    _publish_once(output / "acceptance.json", _bytes(summary))
    progress.emit("COMPLETE", "3/3 immutable test results verified", freeze_sha256=FREEZE_SHA256)


def run(mode):
    output = ROOT / "test" / FREEZE_SHA256
    progress = Progress(output / "progress.md", " ".join(sys.orig_argv), title="Batch B S4 frozen test")
    progress.emit("FREEZE_AUDIT", "Checking reviewed freeze before opening the test partition")
    frozen, data, binding, audit = audit_inputs()
    _publish_once(output / "pretest-audit.json", _bytes(audit))
    progress.emit("FREEZE_OK", "Three checkpoints, validation evidence and dataset metadata verified")
    if mode == "audit":
        return
    recipe = load_recipe()
    settings = recipe["protocol"]
    cached = output / "test-input.jsonl"
    root = Path(frozen["dataset"]["root"])
    seen = {row.sample_id for row in (*data.train, *data.validation)}
    with exclusive_queue(ROOT.parents[1] / ".batch-a-gpu.lock"):
        contract = {"freeze_sha256": FREEZE_SHA256, "test_partition": TEST,
                    "pretest_audit_sha256": sha256_file(output / "pretest-audit.json"),
                    "evaluator_sha256": sha256_file(Path(__file__)),
                    "launcher_sha256": sha256_file(Path("scripts/batch_b_s4_wsl_test.sh")),
                    "inference": {"batch_size": 32, "length": 64, "multiple": 8,
                                  "length_sorted": True, "score_dtype": "float32"},
                    "encoder_comparison": _encoders()}
        _publish_once(output / "contract.json", _bytes(contract))
        if mode == "replay" or (output / "acceptance.json").exists():
            rows = test_rows(root / TEST["file"], cached, TEST, seen, allow_source=False)
            evaluate(frozen, data, output, rows, progress)
            return
        configure_compute_threads()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with (
            protect(progress, process_limit=settings["preflight"]["process_commit_limit_gib"] * GIB),
            GpuTelemetryMonitor(interval=0.25) as telemetry,
        ):
            configure_device()
            if _binding(recipe) != binding:
                raise ValueError("Testing must use the exact S4 WSL2 inference environment")
            progress.emit("VERIFY_MODEL", "Verifying the official model and tokenizer files")
            snapshot, _ = verify_resources(settings)
            tokenizer = load_tokenizer(snapshot)
            policy = settings["preprocessing"]
            if not tokenizer.is_fast or (tokenizer.pad_token_id, tokenizer.eos_token_id) != (
                policy["pad_token_id"], policy["eos_token_id"]
            ):
                raise ValueError("Official tokenizer identity differs")
            if not cached.exists():
                progress.emit("OPEN_TEST", "Reading the frozen source test partition into its local cache")
            rows = test_rows(root / TEST["file"], cached, TEST, seen, allow_source=True)
            partition, lengths = encode(tokenizer, data, rows, 64, policy)
            _publish_once(output / "tokenization.json", _bytes({
                "model_revision": settings["model_revision"], "rows": len(rows), "max_length": 64,
                "truncated_rows": int(np.count_nonzero(lengths > 64)),
                "encoded_sha256": input_fingerprint(partition),
            }))
            gate = GpuHeadroom(telemetry, progress, strict=False)
            evaluate(frozen, data, output, rows, progress,
                     infer=lambda entry: _infer(snapshot, recipe, entry, partition, progress, gate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "test", "replay"))
    run(parser.parse_args().mode)

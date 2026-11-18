"""Independently audit and evaluate the approved Batch B encoder cohorts."""

from __future__ import annotations

import gc
import hashlib
import json
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.transformer_inputs import encode_partition, input_fingerprint
from nlp_text_classification.evaluation.batch_metrics import measure, write_predictions
from nlp_text_classification.experiments.batch_b_resources import (
    environment,
    label_names,
    load_config,
    load_tokenizer,
    read_resources,
)
from nlp_text_classification.experiments.batch_b_s1 import _verify_attempt
from nlp_text_classification.experiments.batch_b_s2 import (
    OUTPUT as S2_OUTPUT,
)
from nlp_text_classification.experiments.batch_b_s2 import Inputs, check_files
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import (
    checked_file,
    load_partition,
    load_training_data,
    load_yaml,
    read_json,
)
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import CPU_THREADS, GIB, protect
from nlp_text_classification.training.run import git_commit

OUTPUT = Path("artifacts/batch-b-final")
PROTOCOL = Path("configs/experiment_suites/batch_b_final.yaml")
REFERENCE = OUTPUT / "freeze_ref.json"
TOTAL = 12
SEEDS = (42, 123, 2026)
MODELS = ("B1", "B2", "B3", "B4")
EVALUATION_FILES = ("metrics_test.json", "predictions_test.jsonl")


def run_key(entry):
    return f'{entry["model"]}-{entry["seed"]}'


def require_matrix(rows):
    expected = {f"{model}-{seed}" for model in MODELS for seed in SEEDS}
    if len(rows) != TOTAL or {run_key(entry) for entry in rows} != expected or (
        len({Path(entry["run"]).resolve() for entry in rows}) != TOTAL
    ):
        raise ValueError("Freeze requires one distinct Run for every model and seed")
    if any(entry["role"] != ("primary" if entry["seed"] == 42 else "replication")
           for entry in rows):
        raise ValueError("Primary and replication roles changed")


def json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def publish_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Published bytes differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=path.parent) as directory:
        pending = Path(directory) / path.name
        pending.write_bytes(content)
        pending.rename(path)


def _source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    files = ("experiments/batch_b_final.py", "evaluation/batch_b_final_report.py",
             "training/transformer_runner.py", "evaluation/batch_metrics.py",
             "data/transformer_inputs.py", "experiments/batch_b_resources.py",
             "experiments/suite.py")
    return {name: sha256_file(root / name) for name in files}


def _candidate_source(entry, inputs, queue):
    model, seed = entry["model"], entry["seed"]
    if seed == 42:
        return inputs.originals[model]
    return queue["accepted"][f"replication/{model}/{seed}"]


def _candidate_config(entry, run, data, resources):
    manifest = read_json(run / "run_manifest.json")
    resolved = load_yaml(checked_file(run / "resolved_config.yaml", manifest["config_sha256"]))
    threshold = read_json(run / "thresholds.json")
    result = read_json(run / "result.json")
    quality = read_json(run / "quality.json")
    expected = {
        "model": manifest["experiment"], "seed": manifest["seed"], "dataset": manifest["dataset"],
        "training_manifest_sha256": manifest["training_manifest_sha256"],
        "dataset_manifest_sha256": data.settings["dataset_manifest_sha256"],
        "label_order": label_names(data), "positive_raw_id": data.settings.get("positive_raw_id"),
        "model_id": resolved["resource"]["model_id"],
        "model_revision": resolved["resource"]["revision"], "settings": resolved["settings"],
        "tokenization": resolved["tokenization"], "selection": resolved["selection"],
        "tokenizer_sha256": resolved["tokenization"]["tokenizer_sha256"],
        "checkpoint": result["checkpoint"], "best_epoch": result["best_epoch"],
        "validation_macro_f1": result["macro_f1"], "threshold": threshold["threshold"],
        "score_kind": "softmax_positive" if data.binary else "sigmoid",
    }
    if any(entry[field] != expected[field] for field in expected):
        raise ValueError(f"Proposal differs from the recorded Run: {run}")
    if data.binary and (
        data.manifest["label_ids"] != [1, 0] or entry["positive_raw_id"] != 0
        or quality["logit_width"] != 2 or quality["loss"] != "cross_entropy"
    ):
        raise ValueError(f"B4 semantic label mapping or loss differs: {run}")
    if threshold["score_kind"] != entry["score_kind"] or quality["score_kind"] != entry["score_kind"]:
        raise ValueError(f"Run score type differs: {run}")
    if threshold["threshold"] != result["threshold"] or resolved["resource"] != resources:
        raise ValueError(f"Run threshold or official resources differ: {run}")
    return resolved


def _infer(entry, data, samples, resources, progress, phase, index):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from nlp_text_classification.training.transformer_runner import DevicePartition, load_checkpoint

    run = Path(entry["run"])
    device = torch.device("cuda")
    tokenizer = load_tokenizer(Path(resources["cache"]))
    token = entry["tokenization"]
    packed = encode_partition(tokenizer, data, samples, token["selected_length"],
                              token["padding_multiple"])
    if phase == "VALIDATION" and input_fingerprint(packed) != token["partitions"]["validation"][
        "encoded_sha256"
    ]:
        raise ValueError(f"Validation tokenization fingerprint changed: {run}")
    config = AutoConfig.from_pretrained(run / "checkpoint", local_files_only=True,
                                        trust_remote_code=False)
    labels = entry["label_order"]
    task = "single_label_classification" if data.binary else "multi_label_classification"
    if config.id2label != dict(enumerate(labels)) or config.label2id != {
        label: column for column, label in enumerate(labels)
    } or config.problem_type != task:
        raise ValueError(f"Checkpoint does not encode the frozen label order: {run}")
    model = AutoModelForSequenceClassification.from_config(
        config, trust_remote_code=False, attn_implementation="sdpa",
    ).to(device)
    weights = load_checkpoint(run / entry["checkpoint"], device)
    if weights["classifier.weight"].shape[0] != len(labels):
        raise ValueError(f"Actual classifier weight shape differs: {run}")
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval()
    partition = DevicePartition(packed, device)
    size = entry["settings"]["evaluation_batch_size"]
    bf16 = entry["settings"]["precision"] == "bf16"
    scores = []
    with torch.inference_mode():
        for start in range(0, len(samples), size):
            stop = min(start + size, len(samples))
            features, _ = partition.batch(np.arange(start, stop), token["padding_multiple"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(**features).logits
            if tuple(logits.shape) != (stop-start, len(labels)):
                raise ValueError(f"Actual logit shape differs: {run}")
            logits = logits.float()
            scores.append(logits.softmax(dim=-1)[:, 1:2] if data.binary else logits.sigmoid())
            if stop == len(samples) or (start // size + 1) % 64 == 0:
                progress.emit(phase, f"{run_key(entry)} {stop}/{len(samples)} samples",
                              run=run_key(entry), position=index, total_runs=TOTAL,
                              samples_processed=stop, samples_total=len(samples))
    output = torch.cat(scores).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite scores in {phase}: {run}")
    evidence = {"classifier": type(model).__name__, "logit_width": len(labels),
                "model_type": config.model_type, "problem_type": config.problem_type}
    del model, partition, packed, scores, logits
    gc.collect()
    torch.cuda.empty_cache()
    return output, evidence


def replay(path, samples, data, entry):
    from nlp_text_classification.training.transformer_runner import verify_prediction_file

    truth = data.encode(samples)
    cutoff = entry["threshold"]
    metrics, scores = verify_prediction_file(path, samples, truth, data.binary, cutoff, auc=True)
    with path.open(encoding="utf-8") as stream:
        for sample, line in zip(samples, stream, strict=True):
            if json.loads(line)["raw_labels"] != list(sample.labels):
                raise ValueError(f"Raw label differs from the frozen partition: {sample.sample_id}")
    pred = scores >= cutoff
    options = {"zero_division": 0, "labels": [0, 1] if data.binary else None}
    oracle = {
        "macro_f1": f1_score(truth, pred, average="macro", **options),
        "micro_f1": f1_score(truth, pred, average="micro", **options),
        ("accuracy" if data.binary else "subset_accuracy"): accuracy_score(truth, pred),
    }
    if any(abs(metrics[key] - value) > 1e-12 for key, value in oracle.items()):
        raise ValueError(f"Independent score recomputation differs: {path}")
    return metrics, scores, pred


def audit(protocol, progress):
    if REFERENCE.exists():
        frozen, digest = verified_freeze(protocol)
        return {"freeze_sha256": digest, "audited_runs": len(frozen["runs"])}
    from nlp_text_classification.training.transformer_runner import threshold_record

    proposal_path = checked_file(Path(protocol["proposal"]), protocol["proposal_sha256"])
    acceptance = read_json(checked_file(Path(protocol["s2_acceptance"]),
                                        protocol["s2_acceptance_sha256"]))
    if acceptance["status"] != "PASS" or acceptance["proposal_sha256"] != protocol["proposal_sha256"]:
        raise ValueError("S2 acceptance does not match the proposal")
    checked_file(S2_OUTPUT / "delivery-audit.json", acceptance["delivery_audit_sha256"])
    delivery = read_json(S2_OUTPUT / "delivery-audit.json")
    for name, digest in delivery["reports"].items():
        checked_file(proposal_path.parent / name, digest)
    queue = read_json(checked_file(S2_OUTPUT / "queue.json", delivery["queue_sha256"]))
    inputs = Inputs()
    inputs.audit_preservation(all_attempts=True)
    proposal = read_json(proposal_path)
    if proposal["status"] != "proposed" or proposal["source_split"] != "validation" or (
        proposal["suite"] != queue["identity"] or proposal["suite"] != inputs.identity
    ):
        raise ValueError("Unexpected proposal identity")
    decision = inputs.choose_b1(queue["accepted"]["budget/B1/42"])
    if decision != proposal["b1_budget_decision"] or decision["selected_budget"] != "s1_6_epochs":
        raise ValueError("B1 recorded budget differs from the approved cohort")
    resources = read_resources(inputs.models)
    rows = proposal["runs"]
    require_matrix(rows)
    audit_rows = {}
    for index, entry in enumerate(rows, start=1):
        source = _candidate_source(entry, inputs, queue)
        if (entry["run"], entry["files"]) != (source["run"], source["files"]):
            raise ValueError(f"Run or fingerprints disagree with accepted cohort: {run_key(entry)}")
        run = Path(entry["run"])
        data = inputs.data[entry["dataset"]]
        key = run_key(entry)
        candidate = _candidate_config(entry, run, data, resources[entry["model"]])
        expected = inputs.configs[entry["model"]] if entry["seed"] == 42 else inputs.resolved(
            entry["model"], entry["seed"],
        )
        if candidate != expected or entry["stage"] != ("S1" if entry["seed"] == 42 else "S2"):
            raise ValueError(f"Run settings or source stage differ: {key}")
        progress.emit("AUDIT", f"{key} checking file fingerprints and validation evidence",
                      position=index, total_runs=TOTAL)
        if _verify_attempt(run, data) != {"run": source["run"], "result": source["result"],
                                         "files": source["files"]}:
            raise ValueError(f"Independent validation evidence differs: {key}")
        _, archived, predicted = replay(run / "predictions_validation.jsonl", data.validation,
                                        data, entry)
        observed, architecture = _infer(entry, data, data.validation, resources[entry["model"]],
                                        progress, "VALIDATION", index)
        settings = candidate["quality"]
        if not np.allclose(observed, archived, atol=settings["reload_score_atol"],
                           rtol=settings["reload_score_rtol"]) or (
            not np.array_equal(observed >= entry["threshold"], predicted)
        ):
            raise ValueError(f"Actual validation predictions differ at frozen checkpoint: {key}")
        if threshold_record(data.encode(data.validation), observed, data.binary,
                            entry["selection"]) != read_json(run / "thresholds.json"):
            raise ValueError(f"Actual validation threshold candidates differ: {key}")
        audit_rows[key] = {"status": "PASS", "validation_rows": len(data.validation),
                           "max_abs_error": float(np.max(np.abs(observed - archived))),
                           "exact_score_match": bool(np.array_equal(observed, archived)),
                           "threshold": entry["threshold"], "score_kind": entry["score_kind"],
                           "checkpoint_sha256": entry["files"][entry["checkpoint"]],
                           "label_order": entry["label_order"], **architecture}
    datasets = {}
    for name, settings in inputs.datasets.items():
        manifest = read_json(checked_file(Path(settings["frozen_root"]) / "dataset_manifest.json",
                                          settings["dataset_manifest_sha256"]))
        data = inputs.data[name]
        if manifest["label_ids"] != data.manifest["label_ids"] or (
            manifest["label_names"] != label_names(data)
        ):
            raise ValueError(f"Dataset manifest label order differs: {name}")
        datasets[name] = {"manifest_sha256": settings["dataset_manifest_sha256"],
                          "training_manifest_sha256": settings["training_manifest_sha256"],
                          "root": settings["frozen_root"], "test": manifest["splits"]["test"],
                          "label_order": label_names(data),
                          "train_support": _train_support(data)}
    audit_path = OUTPUT / "freeze_audit.json"
    publish_once(audit_path, json_bytes({"status": "PASS", "source": "validation_only",
                                        "proposal_sha256": protocol["proposal_sha256"],
                                        "runs": audit_rows}))
    frozen = {**proposal, "status": "frozen", "protocol": protocol,
              "protocol_sha256": sha256_file(PROTOCOL), "freeze_code_sha": git_commit(),
              "audit_sha256": sha256_file(audit_path), "datasets": datasets,
              "hardware": inputs.s1["hardware"], "environment": environment(),
              "evaluator_sources": _source_fingerprints()}
    content = json_bytes(frozen)
    digest = hashlib.sha256(content).hexdigest()
    path = OUTPUT / "frozen" / f"{digest}.json"
    publish_once(path, content)
    publish_once(REFERENCE, json_bytes({"freeze": path.as_posix(), "sha256": digest}))
    progress.emit("FROZEN", "12/12 Runs passed independent validation replay", freeze_sha256=digest)
    return {"freeze_sha256": digest, "audited_runs": TOTAL}


def _train_support(data):
    counts = data.encode(data.train).sum(axis=0).tolist()
    return [len(data.train) - counts[0], counts[0]] if data.binary else counts


def verified_freeze(protocol):
    reference = read_json(REFERENCE)
    digest = reference["sha256"]
    path = OUTPUT / "frozen" / f"{digest}.json"
    if reference["freeze"] != path.as_posix():
        raise ValueError("Unexpected freeze reference")
    frozen = read_json(checked_file(path, digest))
    if (frozen["status"], frozen["protocol"], frozen["protocol_sha256"]) != (
        "frozen", protocol, sha256_file(PROTOCOL)
    ) or frozen["environment"] != environment() or (
        frozen["evaluator_sources"] != _source_fingerprints()
    ):
        raise ValueError("Published freeze, environment or evaluator has changed")
    require_matrix(frozen["runs"])
    audit_evidence = read_json(checked_file(OUTPUT / "freeze_audit.json", frozen["audit_sha256"]))
    if audit_evidence["status"] != "PASS" or set(audit_evidence["runs"]) != {
        run_key(entry) for entry in frozen["runs"]
    }:
        raise ValueError("Freeze audit is incomplete")
    for entry in frozen["runs"]:
        check_files(entry)
    return frozen, digest


def test_inputs(frozen, datasets):
    data, samples = {}, {}
    for name, settings in datasets.items():
        record = frozen["datasets"][name]
        root = Path(record["root"])
        if (settings["frozen_root"], settings["dataset_manifest_sha256"]) != (
            record["root"], record["manifest_sha256"]
        ):
            raise ValueError(f"Dataset configuration changed: {name}")
        manifest = read_json(checked_file(root / "dataset_manifest.json", record["manifest_sha256"]))
        data[name] = load_training_data(settings)
        if manifest["splits"]["test"] != record["test"] or (
            label_names(data[name]) != record["label_order"]
            or settings["training_manifest_sha256"] != record["training_manifest_sha256"]
        ):
            raise ValueError(f"Test partition or label order changed: {name}")
        samples[name] = load_partition(root, record["test"], "test")
        if {row.sample_id for row in samples[name]} & {
            row.sample_id for row in (*data[name].train, *data[name].validation)
        }:
            raise ValueError(f"Frozen test IDs overlap train or validation: {name}")
    return data, samples


def test_record(entry, frozen, digest):
    return {"freeze_sha256": digest, "run": entry["run"], "run_key": run_key(entry),
            "checkpoint_sha256": entry["files"][entry["checkpoint"]],
            "threshold": entry["threshold"], "score_kind": entry["score_kind"],
            "test": frozen["datasets"][entry["dataset"]]["test"]}


def read_test(destination, entry, frozen, digest, data, samples):
    stored = read_json(destination / "evaluation_manifest.json")
    expected = test_record(entry, frozen, digest)
    if any(stored[key] != value for key, value in expected.items()) or (
        set(stored["files"]) != set(EVALUATION_FILES)
    ):
        raise ValueError(f"Published result belongs to another freeze or Run: {destination}")
    for name, checksum in stored["files"].items():
        checked_file(destination / name, checksum)
    metrics, _, pred = replay(destination / "predictions_test.jsonl", samples, data, entry)
    if metrics != read_json(destination / "metrics_test.json"):
        raise ValueError(f"Published test metrics differ from predictions: {destination}")
    return metrics, pred


def final_test(frozen, digest, datasets, resources, progress):
    data, samples = test_inputs(frozen, datasets)
    output = OUTPUT / "test" / digest
    output.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(frozen["runs"], start=1):
        key, name = run_key(entry), entry["dataset"]
        destination = output / key
        if destination.exists():
            read_test(destination, entry, frozen, digest, data[name], samples[name])
            continue
        scores, _ = _infer(entry, data[name], samples[name], resources[entry["model"]],
                           progress, "TEST", index)
        target = data[name].encode(samples[name])
        with tempfile.TemporaryDirectory(prefix=".test-", dir=output) as directory:
            result = Path(directory) / "result"
            result.mkdir()
            cutoff = entry["threshold"]
            write_predictions(result / "predictions_test.jsonl", samples[name], target, scores, cutoff)
            metrics = measure(target, scores, cutoff, data[name].binary)
            write_atomic_json(result / "metrics_test.json", metrics)
            write_atomic_json(result / "evaluation_manifest.json", {
                **test_record(entry, frozen, digest),
                "files": {name: sha256_file(result / name) for name in EVALUATION_FILES},
            })
            read_test(result, entry, frozen, digest, data[name], samples[name])
            result.rename(destination)
        progress.emit("ACCEPTED", f"{key} published and verified", completed=index, total_runs=TOTAL)
    progress.emit("TEST_COMPLETE", "12/12 frozen test results verified", freeze_sha256=digest)
    return {"tested_runs": TOTAL, "freeze_sha256": digest}


def _verify_hardware(required):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_name() != required["gpu"] or (
        torch.version.cuda != required["torch_cuda"]
    ) or (required["precision"] == "bf16" and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Audit and test require the frozen S1/S2 CUDA device and precision")
    torch.set_num_threads(CPU_THREADS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def execute(mode):
    from threadpoolctl import threadpool_limits

    OUTPUT.mkdir(parents=True, exist_ok=True)
    progress = Progress(OUTPUT / "progress.md", shlex.join(sys.orig_argv),
                        title="Batch B formal evaluation progress")
    with exclusive_queue(OUTPUT.parent / ".batch-a-gpu.lock"), threadpool_limits(limits=CPU_THREADS):
        protocol = load_yaml(PROTOCOL)
        _, datasets, models = load_config()
        if mode == "audit":
            with protect(progress, process_limit=14 * GIB):
                original = load_yaml(Path("configs/experiment_suites/batch_b_s2.yaml"))
                hardware = read_json(checked_file(Path(original["s1_results"]),
                                                  original["s1_results_sha256"]))["hardware"]
                _verify_hardware(hardware)
                return audit(protocol, progress)
        frozen, digest = verified_freeze(protocol)
        if mode == "test":
            with protect(progress, process_limit=14 * GIB):
                _verify_hardware(frozen["hardware"])
                return final_test(frozen, digest, datasets, read_resources(models), progress)
        from nlp_text_classification.evaluation.batch_b_final_report import publish

        result = publish(frozen, digest, datasets, progress)
        progress.emit("COMPLETE", "Formal test report and acceptance verified", **result)
        return result

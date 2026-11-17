"""Command-line entry points for Batch A smoke, execution, and progress."""

from __future__ import annotations

import shlex
import sys

import numpy as np

from nlp_text_classification.data.contracts import Example
from nlp_text_classification.experiments.suite import TrainingData, load_suite, read_json
from nlp_text_classification.training.batch_run import resolve, run_attempt
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.run import git_commit


def synthetic_data(key, settings):
    binary = settings["task_type"] == "binary"
    nlabels = settings["num_labels"]
    rows = {}
    for split, repeats in (("train", 3), ("validation", 1)):
        rows[split] = [
            Example(f"{split}:{label}:{index}", f"pattern{label} signal{label} {split}{index}",
                    (label,), split)
            for label in range(nlabels) for index in range(repeats)
        ]
    settings = {**settings, "max_length": 16, "training_manifest_sha256": "synthetic"}
    return TrainingData(settings, {
        "dataset": key, "task_type": settings["task_type"],
        "label_ids": [1, 0] if binary else list(range(nlabels)),
    }, rows["train"], rows["validation"])


def smoke(config, datasets, output, progress, device):
    results = []
    rng = np.random.default_rng(42)
    words = {f"pattern{label}": rng.random(50, dtype=np.float32) for label in range(28)}
    word2vec = words, {"source": "synthetic"}
    order = [(key, experiment["id"]) for key in datasets for experiment in config["experiments"]]
    for i, (key, experiment) in enumerate(order, 1):
        progress.context = {"matrix_position": f"{i}/{len(order)}", "mode": "smoke"}
        settings = {"epoch_cap": 2} if experiment != "E1" else None
        data = synthetic_data(key, datasets[key])
        run = run_attempt(data, resolve(config, experiment, settings), output / "runs",
                          progress, cohort="smoke", seed=42, device=device, word2vec=word2vec)
        result = read_json(run / "result.json")
        results.append({"dataset": key, "experiment": experiment, "run": str(run), **result})
    passed = sum(item["status"] == "PASS" for item in results)
    progress.context = {}
    progress.emit("SMOKE", f"{passed}/{len(order)} passed", evidence=output)
    return {"passed": passed, "total": len(order), "results": results}


def batch_command(mode, suite_path, output, device):
    if mode == "status":
        snapshot = output / "progress.md"
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        return {"progress": str(snapshot), "text": snapshot.read_text(encoding="utf-8")}
    config, datasets = load_suite(suite_path)
    command = shlex.join(sys.orig_argv)
    progress = Progress(output / "progress.md", command)
    progress.emit("START", f"mode={mode} device={device}", output=output)
    try:
        return dispatch(config, datasets, mode, output, progress, device)
    except (OSError, ValueError, RuntimeError) as exc:
        progress.context = {}
        progress.emit("BLOCKED", str(exc), mode=mode)
        raise


def dispatch(config, datasets, mode, output, progress, device):
    if mode == "smoke":
        result = smoke(config, datasets, output, progress, device)
        result.update({"device": device, "suite_revision": config["revision"], "code_sha": git_commit()})
        write_atomic_json(output / "summary.json", result)
        if result["passed"] != result["total"]:
            raise RuntimeError(f"Smoke gate failed. See {output / 'progress.md'}")
        return result
    if mode in {"audit", "final-test", "report"}:
        from nlp_text_classification.experiments.final_evaluation import evaluation_command

        return evaluation_command(mode, config, datasets, output, progress, device)
    from nlp_text_classification.experiments.batch_scheduler import run_batch

    return run_batch(config, datasets, output, progress, device)

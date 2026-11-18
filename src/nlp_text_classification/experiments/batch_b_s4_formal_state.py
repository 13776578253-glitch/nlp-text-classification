"""Deterministic epoch order, selection, and atomic S4 formal checkpoints."""

import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from safetensors import safe_open

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import write_predictions
from nlp_text_classification.experiments.batch_b_s4_runner import _state_evidence
from nlp_text_classification.experiments.suite import checked_file, read_json
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.qwen_s3 import rng_state
from nlp_text_classification.training.transformer_runner import verify_prediction_file


def epoch_order(seed, epoch, rows):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)


def update_rows(order, update, effective):
    if not 0 <= update < (len(order) + effective - 1) // effective:
        raise ValueError(f"Invalid local optimizer step: {update}")
    return order[update * effective:(update + 1) * effective]


def sampler_cursor(seed, epoch, batch, rows, effective):
    order = epoch_order(seed, epoch, rows)
    indices = update_rows(order, batch, effective)
    return {
        "epoch": epoch, "next_batch": batch, "next_rows": len(indices),
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "next_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def new_state():
    return {"global_step": 0, "epoch": 0, "batch": 0, "loss_sum": 0.0, "loss_rows": 0,
            "epochs": [], "best": None, "stale_epochs": 0, "stop_reason": None}


def select_epoch(state, validation, training, selection, updates_per_epoch):
    if state["batch"] != updates_per_epoch or state["epoch"] != len(state["epochs"]):
        raise ValueError("Epoch selection requires one fully trained epoch")
    epoch = state["epoch"] + 1
    entry = {
        "epoch": epoch, "global_step": state["global_step"],
        "macro_f1": validation["metrics"]["macro_f1"], "threshold": validation["cutoff"],
        "train_loss": state["loss_sum"] / state["loss_rows"],
    }
    current_best = state["best"]
    improved = current_best is None or entry["macro_f1"] > (
        current_best["macro_f1"] + selection["score_tie_atol"]
    )
    stale = 0 if improved else state["stale_epochs"] + 1
    stop = None
    if stale >= training["patience"]:
        stop = "early_stopping"
    elif epoch == training["epochs"]:
        stop = "epoch_cap"
    return state | {
        "epoch": epoch, "batch": 0, "loss_sum": 0.0, "loss_rows": 0,
        "epochs": [*state["epochs"], entry], "best": entry if improved else current_best,
        "stale_epochs": stale, "stop_reason": stop,
    }


def checkpoint_path(root, step):
    return root / "checkpoints" / f"step-{step:08d}"


def latest_checkpoint(root):
    checkpoints = root / "checkpoints"
    if not checkpoints.is_dir():
        return None
    return max((item for item in checkpoints.iterdir()
                if item.is_dir() and re.fullmatch(r"step-\d{8}", item.name)), default=None)


def optimizer_progress(optimizer, scheduler, expected):
    steps = {int(item["step"].item()) for item in optimizer.state.values()}
    rates = [group["lr"] for group in optimizer.param_groups]
    parameters = sum(len(group["params"]) for group in optimizer.param_groups)
    if scheduler.last_epoch != expected or rates != scheduler.get_last_lr() or (
        expected and (steps != {expected} or len(optimizer.state) != parameters)
    ) or (not expected and steps):
        raise ValueError("AdamW step, scheduler step, parameter states, or group LR disagree")
    return {"optimizer_step": expected, "scheduler_step": scheduler.last_epoch,
            "parameter_states": len(optimizer.state), "learning_rates": rates}


def _save_adapter(model, destination):
    destination.mkdir()
    model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
    model.get_base_model().config.to_json_file(destination / "text_config.json")
    with safe_open(destination / "adapter_model.safetensors", framework="pt") as archive:
        keys = archive.keys()
        head = [name for name in keys if name.endswith(".score.weight")]
        if len(keys) != 497 or len(head) != 1 or archive.get_slice(head[0]).get_shape() != [28, 2560]:
            raise ValueError("Incomplete S4 LoRA and classification-head package")


def _save_validation(path, validation):
    rows, targets = validation["rows"], validation["targets"]
    cutoff = validation["cutoff"]
    predictions = path / "predictions_validation.jsonl"
    write_predictions(predictions, rows, targets, validation["scores"], cutoff)
    replay, scores = verify_prediction_file(predictions, rows, targets, False, cutoff, auc=True)
    if replay != validation["metrics"] or not np.array_equal(scores, validation["scores"]):
        raise ValueError("Published validation predictions do not replay")
    write_atomic_json(path / "metrics_validation.json", replay)
    write_atomic_json(path / "thresholds.json", validation["thresholds"])


def save_checkpoint(root, engine, state, cursor, identity, validation=None):
    step = state["global_step"]
    destination = checkpoint_path(root, step)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-", dir=destination.parent) as temp:
        staged = Path(temp) / "complete"
        staged.mkdir()
        _save_adapter(engine.model, staged / "adapter")
        if validation is not None:
            _save_validation(staged, validation)
        payload = {
            "optimizer": engine.optimizer.state_dict(), "scheduler": engine.scheduler.state_dict(),
            "rng": rng_state(), "state": state, "cursor": cursor,
        }
        torch.save(payload, staged / "resume.pt")
        write_atomic_json(staged / "checkpoint.json", {
            "identity": identity, "state": state, "cursor": cursor,
            "updates": optimizer_progress(engine.optimizer, engine.scheduler, step),
            "parameters": _state_evidence(engine.rank, engine.head, engine.optimizer),
            "files": {path.relative_to(staged).as_posix(): sha256_file(path)
                      for path in sorted(staged.rglob("*")) if path.is_file()},
        })
        staged.rename(destination)
    return destination


def checked_checkpoint(path, identity):
    receipt = read_json(path / "checkpoint.json")
    if receipt["identity"] != identity or path.name != checkpoint_path(
        path.parent.parent, receipt["state"]["global_step"]
    ).name:
        raise ValueError("Checkpoint contract or step differs from its Run")
    names = {file.relative_to(path).as_posix() for file in path.rglob("*")
             if file.is_file() and file != path / "checkpoint.json"}
    required = {"resume.pt", "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
                "adapter/text_config.json"}
    if not required <= names or names != receipt["files"].keys():
        raise ValueError("Incomplete checkpoint file inventory")
    for name, sha in receipt["files"].items():
        checked_file(path / name, sha)
    return receipt


def restore_checkpoint(path, engine, receipt):
    payload = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)
    if payload["state"] != receipt["state"] or payload["cursor"] != receipt["cursor"]:
        raise ValueError("Checkpoint metadata differs from training state")
    engine.optimizer.load_state_dict(payload["optimizer"])
    engine.scheduler.load_state_dict(payload["scheduler"])
    step = payload["state"]["global_step"]
    if optimizer_progress(engine.optimizer, engine.scheduler, step) != receipt["updates"] or (
        _state_evidence(engine.rank, engine.head, engine.optimizer) != receipt["parameters"]
    ):
        raise ValueError("Restored optimizer, scheduler, or parameter bytes differ")
    return payload

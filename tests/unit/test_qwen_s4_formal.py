from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nlp_text_classification.data.transformer_inputs import EncodedPartition
from nlp_text_classification.experiments import batch_b_s4_formal as formal
from nlp_text_classification.experiments import batch_b_s4_runner as runner
from nlp_text_classification.experiments.batch_b_s4_formal_state import (
    checkpoint_path,
    epoch_order,
    latest_checkpoint,
    new_state,
    optimizer_progress,
    sampler_cursor,
    update_rows,
)
from nlp_text_classification.experiments.suite import read_json
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.transformer_runner import DevicePartition, threshold_record


def test_real_update_budget_optimizer_and_warmup():
    options = read_json(formal.CANDIDATE)["protocol"]["training"]
    rank = torch.nn.Parameter(torch.zeros(1))
    head = torch.nn.Parameter(torch.zeros(1))
    adam, scheduler, plan = runner.make_optimizer({"rank": rank}, {"head": head}, options,
                                                 43211, foreach=True)
    assert plan == {"effective_batch_size": 16, "updates_per_epoch": 2701, "tail_rows": 11,
                    "total_updates": 13505, "warmup_updates": 811}
    assert runner.update_budget(options | {"micro_batch_size": 4, "accumulation_steps": 4},
                                43211) == plan
    assert runner.update_budget(options, 43200)["tail_rows"] == 0
    assert scheduler.get_last_lr() == [0.0, 0.0]
    assert [group["foreach"] for group in adam.param_groups] == [True, True]
    for i in range(2):
        (rank + head).sum().backward()
        adam.step()
        scheduler.step()
        adam.zero_grad(set_to_none=True)
        assert optimizer_progress(adam, scheduler, i + 1)["learning_rates"] == pytest.approx(
            [(i + 1) * base / 811 for base in (1e-4, 5e-4)]
        )
    assert rank.item() < 0 and head.item() < rank.item()
    assert scheduler.lr_lambdas[0](811) == 1.0
    assert scheduler.lr_lambdas[0](13505) == 0.0


def test_seed_permutation_tail_and_cursor():
    order = epoch_order(42, 0, 43211)
    actual = np.concatenate([update_rows(order, step, 16) for step in range(2701)])
    np.testing.assert_array_equal(np.sort(actual), np.arange(43211))
    assert len(update_rows(order, 2700, 16)) == 11
    with pytest.raises(ValueError, match="Invalid local optimizer step"):
        update_rows(order, 2701, 16)
    cursor = sampler_cursor(42, 0, 2700, 43211, 16)
    assert cursor["next_rows"] == 11
    assert cursor == sampler_cursor(42, 0, 2700, 43211, 16)
    assert cursor["order_sha256"] != sampler_cursor(42, 1, 2700, 43211, 16)["order_sha256"]
    assert len({sampler_cursor(seed, 0, 0, 43211, 16)["order_sha256"]
                for seed in (42, 123, 2026)}) == 3


def test_long_batch_recomputation_preserves_normal_layer_pattern(monkeypatch):
    engine = formal.Engine.__new__(formal.Engine)
    engine.model = engine.optimizer = engine.scheduler = None
    engine.suite = engine.trainable = None
    engine.options = {"micro_batch_size": 16, "gradient_checkpointing": "partial_16"}
    engine.gate = SimpleNamespace(check=lambda _: None)
    engine.train = SimpleNamespace(lengths=np.array([48, 49]))
    engine.extra_checkpoint_layers = [SimpleNamespace(gradient_checkpointing=False) for _ in range(16)]
    states, releases = [], []

    def update(*_args, **_kwargs):
        states.append(tuple(layer.gradient_checkpointing for layer in engine.extra_checkpoint_layers))
        return {}

    monkeypatch.setattr(formal, "run_update", update)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: releases.append(True))
    assert engine.update(np.array([0]))["gradient_checkpointing"] == "partial_16"
    assert engine.update(np.array([1]))["gradient_checkpointing"] == "full"
    assert states == [(False,) * 16, (True,) * 16]
    assert releases == [True, True]
    assert all(not layer.gradient_checkpointing for layer in engine.extra_checkpoint_layers)


def test_sorted_validation_restores_canonical_rows_and_threshold(monkeypatch):
    lengths = np.tile(np.array([8, 3, 5, 1, 7], dtype=np.int32), 13)
    ids = np.arange(len(lengths))
    targets = np.tile((ids % 2 == 0)[:, None], (1, 28))
    encoded = EncodedPartition({"input_ids": np.repeat(ids[:, None], 8, axis=1)}, lengths, targets)
    evaluated = []
    batches = []

    class Model:
        def eval(self):
            return self

        def __call__(self, input_ids, use_cache):
            assert use_cache is False
            evaluated.extend(input_ids[:, 0].tolist())
            return SimpleNamespace(logits=torch.where(input_ids[:, :1] % 2 == 0, 2., -2.)
                                   .expand(-1, 28))

    monkeypatch.setattr(runner, "DevicePartition", lambda subset, _: DevicePartition(subset, "cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    _, scores, _, timing = runner.evaluate_subset(
        Model(), encoded, ids, batch_size=32, sort_by_length=True, on_batch=batches.append,
    )
    assert timing["rows"] == 65 and timing["sort_by_length"] is True
    assert batches == [32, 64, 65]
    assert evaluated == ids[np.argsort(lengths, kind="stable")].tolist()
    np.testing.assert_array_equal(scores >= 0.5, targets)
    selection = read_json(formal.CANDIDATE)["protocol"]["selection"]
    threshold = threshold_record(targets, scores, False, selection)
    assert (len(threshold["candidates"]), threshold["threshold"], threshold["macro_f1"]) == (
        19, 0.5, 1.0,
    )


@pytest.mark.parametrize("scores, expected_epoch, reason", [
    ([0.5, 0.61, 0.61 + 1e-13, 0.60], 2, "early_stopping"),
    ([0.5, 0.52, 0.59, 0.61, 0.62], 5, "epoch_cap"),
])
def test_formal_epoch_checkpoints_and_patience(monkeypatch, tmp_path, scores, expected_epoch,
                                              reason):
    recipe = read_json(formal.CANDIDATE)
    steps, saved, events = [], [], []

    class Engine:
        def __init__(self, *_args, **_kwargs):
            self.options = recipe["protocol"]["training"]
            self.suite = recipe["protocol"]
            self.plan = {"effective_batch_size": 16, "updates_per_epoch": 2}
            self.data = SimpleNamespace(train=range(27))

        def update(self, indices):
            steps.append(indices.tolist())
            return {"loss": len(indices), "used_lrs": [1e-4, 5e-4],
                    "gradient_norm": 0.5, "valid_tokens": len(indices),
                    "padded_tokens": len(indices) * 8,
                    "gradient_checkpointing": "partial_16"}

        def validate(self):
            i = len(steps) // 2 - 1
            return {"metrics": {"macro_f1": scores[i]}, "cutoff": 0.5 + i * 0.05,
                    "duration_seconds": 1.0}

    def save(root, _engine, state, cursor, _identity, validation=None):
        path = checkpoint_path(root, state["global_step"])
        path.mkdir(parents=True)
        saved.append((state, cursor, validation, path))
        return path

    monkeypatch.setattr(formal, "Engine", Engine)
    monkeypatch.setattr(formal, "save_checkpoint", save)
    progress = SimpleNamespace(emit=lambda phase, _message, **_: events.append(phase))
    gate = SimpleNamespace(min_free_mib=4000)
    formal._train(tmp_path, recipe, None, 42, {}, gate, progress, resume=False)
    assert list(map(len, steps)) == [16, 11] * len(scores)
    assert len(saved) == 1 + len(scores)
    state, cursor, _, path = saved[-1]
    assert latest_checkpoint(tmp_path) == path
    assert state["best"]["epoch"] == expected_epoch
    assert state["best"]["global_step"] == expected_epoch * 2
    assert state["best"]["threshold"] == 0.5 + (expected_epoch - 1) * 0.05
    assert state["epochs"][0]["train_loss"] == (16 * 16 + 11 * 11) / 27
    assert state["stop_reason"] == reason and cursor is None
    assert events.count("EPOCH") == len(scores) and events[-1] == "COMPLETE"


def test_resume_runs_tail_without_repeating_saved_rows(monkeypatch, tmp_path):
    recipe = read_json(formal.CANDIDATE)
    protocol = recipe["protocol"]
    protocol["training"]["epochs"] = 1
    identity = {"seed": 123}
    write_atomic_json(tmp_path / "run.json", {"identity": identity})
    latest = checkpoint_path(tmp_path, 1)
    latest.mkdir(parents=True)
    mid_epoch = new_state() | {"global_step": 1, "batch": 1, "loss_sum": 16., "loss_rows": 16}
    cursor = sampler_cursor(123, 0, 1, 27, 16)
    payload = {"state": mid_epoch, "cursor": cursor, "rng": {"color": 1}}
    restored_rng = []
    seen = []
    saved = []

    class Engine:
        def __init__(self, *_args, **kwargs):
            assert kwargs["saved"] == latest / "adapter"
            self.options = protocol["training"]
            self.suite = protocol
            self.plan = {"effective_batch_size": 16, "updates_per_epoch": 2}
            self.data = SimpleNamespace(train=range(27))

        def update(self, indices):
            seen.extend(indices.tolist())
            return {"loss": 2., "used_lrs": [1e-4, 5e-4],
                    "gradient_norm": 0.5, "valid_tokens": len(indices),
                    "padded_tokens": len(indices) * 8,
                    "gradient_checkpointing": "partial_16"}

        def validate(self):
            return {"metrics": {"macro_f1": 0.6}, "cutoff": 0.5, "duration_seconds": 1.}

    def save(root, _engine, state, cursor, _identity, validation=None):
        assert cursor is None and validation is not None
        saved.append(state)
        return checkpoint_path(root, state["global_step"])

    monkeypatch.setattr(formal, "Engine", Engine)
    monkeypatch.setattr(formal, "checked_checkpoint", lambda *_: payload)
    monkeypatch.setattr(formal, "restore_checkpoint", lambda *_: payload)
    monkeypatch.setattr(formal, "restore_rng", restored_rng.append)
    monkeypatch.setattr(formal, "save_checkpoint", save)
    progress = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    gate = SimpleNamespace(min_free_mib=4000)
    formal._train(tmp_path, recipe, None, 123, identity, gate, progress, resume=True)
    assert restored_rng == [payload["rng"]]
    assert seen == epoch_order(123, 0, 27)[16:].tolist()
    assert saved[0]["best"]["global_step"] == 2
    assert saved[0]["epochs"][0]["train_loss"] == (16 + 11 * 2) / 27

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from nlp_text_classification.data.contracts import Example
from nlp_text_classification.data.transformer_inputs import (
    EncodedPartition,
    audit_tokenization,
    choose_length,
    model_text,
)
from nlp_text_classification.experiments.suite import TrainingData

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nlp_text_classification.training.transformer_runner import (
    DevicePartition,
    _atomic_checkpoint,
    _forward_scores,
    _optimizer,
    _train_epoch,
    classification_loss,
    load_checkpoint,
    threshold_record,
)


def test_binary_probability_is_semantic_positive_column():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([[0], [1]], dtype=torch.int8)
    score = _forward_scores(logits, binary=True).numpy()
    assert score.dtype == np.float32 and score[0, 0] < 0.5 < score[1, 0]
    assert classification_loss(logits, target, binary=True) < 0.05
    selection = {"score_tie_atol": 1e-12, "threshold_grids": {"sigmoid": {
        "range_inclusive": [0.05, 0.95], "step": 0.05, "default": 0.5,
    }}}
    threshold = threshold_record(target.numpy(), score, True, selection)
    assert (threshold["score_kind"], threshold["threshold"], threshold["macro_f1"]) == (
        "softmax_positive", 0.5, 1.0,
    )


def test_offline_wordpiece_token_audit_keeps_source_boundaries(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n好\n坏\na\nb\nab\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    settings = {"positive_raw_id": 0, "dataset_manifest_sha256": "data",
                "training_manifest_sha256": "train"}
    manifest = {"dataset": "zh", "task_type": "binary", "label_ids": [1, 0]}
    rows = [Example("0", "好   a\tb", (0,), "train"), Example("1", "坏", (1,), "train")]
    validation = [Example("2", "坏 a b", (1,), "validation")]
    data = TrainingData(settings, manifest, rows, validation)
    result = audit_tokenization(tokenizer, data, {
        "candidates": {"zh": [8, 16]}, "coverage_quantile": 0.99, "padding_multiple": 8,
        "chinese_text": "preserve_token_boundaries",
    }, {"revision": "fixture", "files": {}, "weight_file": "model.safetensors"})
    assert model_text(rows[0].text, True) == "好 a b"
    assert result["attention_mask_verified"] and result["special_tokens_verified"]
    assert result["join_without_spaces_comparison"]["train"]["different_ids"] == 1
    assert result["partitions"]["train"]["percentiles"]["50"] == 4.0
    assert choose_length(np.array([8, 16]), [8, 16], 0.99) == 16


def test_training_state_roundtrip_preserves_update_and_scheduler(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(8, 2)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.classifier(torch.nn.functional.dropout(
                input_ids.float(), p=0.2, training=self.training,
            )))

    settings = {"learning_rate": 1e-3, "weight_decay": 0.01, "precision": "fp32",
                "effective_batch_size": 4, "micro_batch_size": 2, "grad_clip_norm": 1.0}
    device = torch.device("cpu")
    partition = DevicePartition(EncodedPartition(
        {"input_ids": np.arange(56).reshape(7, 8)}, np.full(7, 8),
        np.array([[0], [1], [0], [1], [0], [1], [0]]),
    ), device)
    torch.manual_seed(15)
    model = Model()
    optimizer = _optimizer(model, settings, device)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 1, 4)
    rng = torch.Generator().manual_seed(42)

    def step():
        return _train_epoch(model, partition, device, settings, optimizer, scheduler, rng,
                            binary=True, multiple=8, tick=lambda *_: None)

    assert step()[2] == 2
    path = tmp_path / "resume.pt"
    _atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "sampler": rng.get_state(),
                              "cpu": torch.get_rng_state()})
    scores = step()
    weights = deepcopy(model.state_dict())
    state = load_checkpoint(path)
    model = Model()
    optimizer = _optimizer(model, settings, device)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 1, 4)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng.set_state(state["sampler"])
    torch.set_rng_state(state["cpu"])
    assert step() == scores
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in weights.items())

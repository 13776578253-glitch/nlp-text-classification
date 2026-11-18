from types import SimpleNamespace

import numpy as np
import pytest

from nlp_text_classification.data.qwen_s3_inputs import audit, encode


class Tokenizer:
    is_fast = True
    pad_token_id = 248044
    eos_token_id = 248044

    def __call__(self, texts, **options):
        assert options["add_special_tokens"] is False
        return {"input_ids": [[int(piece) for piece in text.split()] for text in texts]}


def partition(texts):
    return [SimpleNamespace(sample_id=f"id{i}", text=text) for i, text in enumerate(texts)]


class Data:
    def __init__(self):
        self.settings = {"training_manifest_sha256": "fingerprint"}
        self.manifest = {"label_names": [str(index) for index in range(28)]}

    def encode(self, rows):
        return np.zeros((len(rows), 28), dtype=np.int8)


def test_qwen_padding_and_last_content_token():
    data = Data()
    data.train = partition(["1 2", "3 4 5"])
    data.validation = partition(["6 7 8 9 10"])
    policy = {"pad_token_id": 248044, "eos_token_id": 248044, "length_candidates": [4, 8],
              "train_coverage": 0.99}
    evidence, parts = audit(Tokenizer(), data, policy)
    assert evidence["selected_length"] == 4
    assert evidence["partitions"]["validation"]["truncated_rows"] == 1
    np.testing.assert_array_equal(parts["train"].features["input_ids"][0], [1, 2, 248044, 248044])
    np.testing.assert_array_equal(parts["train"].features["attention_mask"][0], [1, 1, 0, 0])
    np.testing.assert_array_equal(parts["validation"].features["input_ids"][0], [6, 7, 8, 9])


@pytest.mark.parametrize("text", ["", "3 9 248044"])
def test_qwen_rejects_no_content_or_eos_in_content(text):
    with pytest.raises(ValueError, match="id0"):
        encode(Tokenizer(), Data(), partition([text]), 1, {"pad_token_id": 248044})

from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import save_file

from nlp_text_classification.training.qwen_s3_streaming import OfflineShards


def test_preserves_safetensors_shape_dtype_and_data(tmp_path):
    shard = "official.safetensors"
    weight = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    save_file({"text.weight": weight, "other": torch.ones(3)}, tmp_path / shard)
    with ExitStack() as stack:
        loader = OfflineShards(tmp_path, {"text.weight": shard}, stack)
        source = loader.tensor("text.weight", "cpu")
        torch.testing.assert_close(source, weight, rtol=0, atol=0)
        torch.testing.assert_close(loader.tensor("text.weight", "cpu", torch.float32),
                                   weight.float(), rtol=0, atol=0)


def test_rejects_official_tensor_outside_shard(tmp_path):
    shard = "short.safetensors"
    path = tmp_path / shard
    save_file({"text.weight": torch.ones((4, 5))}, path)
    path.write_bytes(path.read_bytes()[:-1])
    with ExitStack() as stack:
        loader = OfflineShards(tmp_path, {"text.weight": shard}, stack)
        with pytest.raises(ValueError, match="Invalid byte range"):
            loader.tensor("text.weight", "cpu")

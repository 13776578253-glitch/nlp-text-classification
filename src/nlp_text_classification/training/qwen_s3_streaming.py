"""Read official Qwen tensors in bounded blocks on Windows."""

import json
import math
import struct
from contextlib import ExitStack

import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _header(stream, file_bytes):
    length, = struct.unpack("<Q", stream.read(8))
    if length == 0 or length > min(file_bytes - 8, 1 << 20):
        raise ValueError("Unexpected official safetensors header size")
    return 8 + length, json.loads(stream.read(length))


def _payload(stream, start, size, dtype):
    stream.seek(start)
    payload = bytearray(size)
    if stream.readinto(payload) != size:
        raise ValueError("Truncated official tensor payload")
    return torch.frombuffer(payload, dtype=dtype)


class OfflineShards:
    def __init__(self, snapshot, index, stack: ExitStack):
        self.index, self.shards = index, {}
        for name in sorted(set(index.values())):
            path = snapshot / name
            stream = stack.enter_context(path.open("rb"))
            file_bytes = path.stat().st_size
            base, header = _header(stream, file_bytes)
            self.shards[name] = (stream, base, header, file_bytes)

    def tensor(self, key, device, output_dtype=None, *, first_rows=None):
        stream, base, header, file_bytes = self.shards[self.index[key]]
        record = header[key]
        shape, dtype = tuple(record["shape"]), DTYPES[record["dtype"]]
        start, end = record["data_offsets"]
        item_size = torch.empty((), dtype=dtype).element_size()
        total = math.prod(shape)
        if not (0 <= start < end <= file_bytes - base) or end - start != total * item_size:
            raise ValueError(f"Invalid byte range for {key}")
        if first_rows is not None:
            if not 0 < first_rows <= shape[0]:
                raise ValueError(f"Invalid row prefix for {key}")
            shape = (first_rows, *shape[1:])
            total = math.prod(shape)
            end = start + total * item_size
        if end - start <= 64 << 20:
            return _payload(stream, base + start, end - start, dtype).reshape(shape).to(
                device=device, dtype=output_dtype or dtype
            )
        flat = torch.empty(total, dtype=output_dtype or dtype, device=device)
        block = (16 << 20) // item_size
        for offset in range(0, total, block):
            count = min(total - offset, block)
            values = _payload(stream, base + start + offset * item_size, count * item_size, dtype)
            flat[offset:offset + count].copy_(values)
        return flat.reshape(shape)


def load_streamed(model_class, config, snapshot, index, source, quantization):
    import bitsandbytes as bnb

    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = model_class(config)
    with torch.device("cuda"):
        model.model.rotary_emb = type(model.model.rotary_emb)(config)
    with ExitStack() as stack:
        archive = OfflineShards(snapshot, index, stack)
        for target, name in source.items():
            module_name, attribute = target.rsplit(".", 1)
            module = model.get_submodule(module_name)
            quantize = type(module) is torch.nn.Linear and module_name != "score"
            value = archive.tensor(name, "cpu" if quantize else "cuda", torch.bfloat16)
            if quantize:
                if attribute != "weight" or module.bias is not None:
                    raise ValueError(f"Unexpected quantized parameter or bias: {target}")
                parent, leaf = module_name.rsplit(".", 1)
                linear = bnb.nn.Linear4bit(
                    module.in_features, module.out_features, bias=False, device="meta",
                    compute_dtype=quantization.bnb_4bit_compute_dtype,
                    quant_type=quantization.bnb_4bit_quant_type,
                    compress_statistics=quantization.bnb_4bit_use_double_quant,
                )
                linear.weight = bnb.nn.Params4bit(
                    value, requires_grad=False, quant_type=quantization.bnb_4bit_quant_type,
                    compress_statistics=quantization.bnb_4bit_use_double_quant, module=linear,
                )
                setattr(model.get_submodule(parent), leaf, linear.to("cuda"))
            else:
                setattr(module, attribute, torch.nn.Parameter(value, requires_grad=False))
    model.score.weight = torch.nn.Parameter(torch.empty((28, 2560), device="cuda",
                                                        dtype=torch.bfloat16))
    model._init_weights(model.score)
    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = "bitsandbytes"
    model.config.quantization_config = quantization
    if any(tensor.device.type != "cuda" for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("Streaming left an unmaterialized or offloaded tensor")
    return model

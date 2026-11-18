"""Pinned Qwen text classifier, optimizer groups, and adapter recovery."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from contextlib import ExitStack

import numpy as np
import torch
from safetensors import safe_open

from nlp_text_classification.experiments.suite import read_json
from nlp_text_classification.training.qwen_s3_streaming import OfflineShards, load_streamed

TEXT_PREFIX = "model.language_model."
KEY_MAPPING = {r"^model\.language_model\.": "model."}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"python": random.getstate(), "numpy": (name, keys.tolist(), position,
                                                  has_gauss, cached_gaussian),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position,
                         has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def text_config(snapshot, labels, pad):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    text = config.get_text_config()
    if config.model_type != "qwen3_5" or text.model_type != "qwen3_5_text" or (
        text.hidden_size, text.num_hidden_layers, len(labels), text.eos_token_id
    ) != (2560, 32, 28, pad):
        raise ValueError("Unexpected Qwen3.5 Base architecture or label count")
    text.num_labels = len(labels)
    text.id2label = dict(enumerate(labels))
    text.label2id = {name: index for index, name in enumerate(labels)}
    text.problem_type = "multi_label_classification"
    text.pad_token_id = pad
    text.use_cache = False
    return text


def audit_weight_map(snapshot, config, model_class):
    index = read_json(snapshot / "model.safetensors.index.json")["weight_map"]
    keys = Counter("text" if name.startswith(TEXT_PREFIX) else name.split(".")[0]
                   for name in index)
    if keys != {"text": 426, "model": 297, "mtp": 15} or any(
        not name.startswith((TEXT_PREFIX, "model.visual.", "mtp.")) for name in index
    ):
        raise ValueError("Unexpected checkpoint namespaces")
    with torch.device("meta"):
        layout = model_class(config)
    parameters = {name: tuple(value.shape) for name, value in layout.named_parameters()}
    if parameters.pop("score.weight") != (28, 2560):
        raise ValueError("Unexpected score parameter shape")
    source = {}
    for shard in sorted(set(index.values())):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            expected = {name for name, file in index.items() if file == shard}
            if set(handle.keys()) != expected:
                raise ValueError(f"Safetensors header differs from official index: {shard}")
            for name in expected:
                if name.startswith(TEXT_PREFIX):
                    target = "model." + name.removeprefix(TEXT_PREFIX)
                    source[target] = tuple(handle.get_slice(name).get_shape())
    if source != parameters:
        raise ValueError("Text checkpoint keys or shapes differ from the classifier backbone")
    return index, source


def load_base(snapshot, suite, labels, *, audit=True):
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, Qwen3_5TextForSequenceClassification

    config = text_config(snapshot, labels, suite["preprocessing"]["pad_token_id"])
    index, source = audit_weight_map(snapshot, config, Qwen3_5TextForSequenceClassification)
    quant = suite["quantization"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
        llm_int8_skip_modules=quant["skip_modules"],
    )
    omitted = set(index) - {TEXT_PREFIX + name.removeprefix("model.") for name in source}
    if quant["loader"] != "streaming_official_safetensors":
        raise ValueError("S3 uses the audited bounded Windows tensor loader")
    mapping = {name: TEXT_PREFIX + name.removeprefix("model.") for name in source}
    model = load_streamed(Qwen3_5TextForSequenceClassification, config, snapshot, index,
                          mapping, quantization)
    if {name for name, _ in model.named_parameters()} != (source.keys() | {"score.weight"}):
        raise ValueError("Loaded model has unexpected parameters")
    if type(model.score) is not torch.nn.Linear or model.score.bias is not None or (
        not torch.isfinite(model.score.weight).all()
    ):
        raise ValueError("Official classification head must be an ordinary finite floating Linear")
    targets = suite["lora"]["targets"]
    converted = {
        name: module for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    }
    expected = {name.rsplit(".", 1)[0] for name in source if
                name.endswith(".weight") and name.split(".")[-2] in targets}
    if set(converted) != expected or len(expected) != suite["lora"]["expected_targets"]:
        raise ValueError("NF4 conversion does not cover exactly the 248 text projections")
    if any(module.weight.device.type != "cuda" or module.weight.quant_state is None or
           module.weight.quant_state.quant_type != quant["type"] for module in converted.values()):
        raise ValueError("Unexpected quantized tensor state")
    sampled = []
    if audit:
        names = ("embed_tokens.weight", "layers.0.linear_attn.in_proj_qkv.weight",
                 "layers.3.self_attn.q_proj.weight", "layers.31.mlp.down_proj.weight")
        with ExitStack() as stack:
            archive = OfflineShards(snapshot, index, stack)
            for name in names:
                key = TEXT_PREFIX + name
                if name == "embed_tokens.weight":
                    want = archive.tensor(key, "cuda", first_rows=32)
                    got = model.model.embed_tokens.weight[:32]
                    if not torch.equal(want, got):
                        raise ValueError("Source embedding rows differ from loaded weights")
                else:
                    module = converted["model." + name.removesuffix(".weight")]
                    tensor = archive.tensor(key, "cuda", torch.bfloat16)
                    packed, _ = bnb.functional.quantize_4bit(
                        tensor, blocksize=module.weight.quant_state.blocksize,
                        compress_statistics=quant["double_quant"], quant_type=quant["type"],
                    )
                    if not torch.equal(module.weight.data, packed):
                        raise ValueError(f"Reference NF4 weights differ for {name}")
                    del tensor, packed
                sampled.append(name)
    details = {"classifier": type(model).__name__, "hidden_size": model.config.hidden_size,
               "logit_width": model.num_labels, "problem_type": model.config.problem_type,
               "text_weights": len(source), "text_shapes": source, "quantized": sorted(converted),
               "omitted_vision": sum(x.startswith("model.visual.") for x in omitted),
               "omitted_mtp": sum(x.startswith("mtp.") for x in omitted),
               "missing_keys": ["score.weight"], "unused_official_keys": sorted(omitted),
               "mapping": KEY_MAPPING, "loader": quant["loader"], "sampled_weights": sampled,
               "official_head_shape": list(model.score.weight.shape)}
    return model, details


def adapter_model(model, suite, *, saved=None, trainable=True):
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=trainable,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if saved is not None:
        return PeftModel.from_pretrained(model, saved, is_trainable=trainable)
    lora = suite["lora"]
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lora["rank"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], bias="none", target_modules=lora["targets"],
        modules_to_save=lora["modules_to_save"],
    ))


def trainable_groups(model, suite):
    rank, head = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A." in name or ".lora_B." in name:
            rank[name] = parameter
        elif ".score.modules_to_save.default." in name:
            head[name] = parameter
        else:
            raise ValueError(f"Unexpected trainable parameter: {name}")
    modules = sorted({name.split(".lora_")[0].removeprefix("base_model.model.")
                      for name in rank})
    counts = {"rank": sum(p.numel() for p in rank.values()),
              "head": sum(p.numel() for p in head.values())}
    if len(modules) != suite["lora"]["expected_targets"] or len(rank) != 2 * len(modules) or (
        len(head) != 1 or counts["head"] != 71680 or
        sum(counts.values()) != suite["lora"]["expected_trainable"]
    ) or any(p.dtype != torch.float32 for p in (*rank.values(), *head.values())):
        raise ValueError("LoRA or classification head count/dtype mismatch")
    if len({id(p) for p in (*rank.values(), *head.values())}) != len(rank) + len(head):
        raise ValueError("Optimizer groups share parameters")
    return rank, head, {"targets": modules, "counts": counts,
                        "shapes": {name: list(p.shape) for name, p in (rank | head).items()}}


def optimizer_for(rank, head, options):
    params = [
        {"params": list(values.values()), "name": label,
         "lr": options[f"{label}_learning_rate"], "weight_decay": options[f"{label}_weight_decay"]}
        for label, values in (("rank", rank), ("head", head))
    ]
    return torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((arr.shape, arr.dtype.str)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer):
    return tensor_digest((f"{i}.{name}", tensor)
                         for i, values in enumerate(optimizer.state.values())
                         for name, tensor in values.items())

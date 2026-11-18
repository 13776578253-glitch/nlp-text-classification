"""Train/validation-only engineering preflight for the S3 Qwen classifier."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

from nlp_text_classification.experiments.batch_b_s3_resources import ROOT, SUITE, verify_resources
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import load_training_data, load_yaml
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import GIB, configure_compute_threads, protect


def inputs(suite):
    from nlp_text_classification.data.qwen_s3_inputs import audit
    from nlp_text_classification.experiments.batch_b_resources import load_tokenizer

    snapshot, resources = verify_resources(suite)
    data = load_training_data(load_yaml(Path(suite["dataset_config"])))
    tokenizer = load_tokenizer(snapshot)
    evidence, encoded = audit(tokenizer, data, suite["preprocessing"])
    evidence["model_revision"] = resources["revision"]
    evidence["tokenizer_files"] = {name: item["sha256"] for name, item in resources["files"].items()
                                   if name.startswith("tokenizer") or name in {"merges.txt",
                                                                                "vocab.json"}}
    return snapshot, data, evidence, encoded


def optimizer_and_schedule(rank, head, options, rows):
    from transformers import get_linear_schedule_with_warmup

    from nlp_text_classification.training.qwen_s3 import optimizer_for

    updates = math.ceil(rows / options["accumulation_steps"]) * options["epochs"]
    warmup = math.ceil(updates * options["warmup_ratio"])
    optimizer = optimizer_for(rank, head, options)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, updates)
    return optimizer, scheduler, {"total_updates": updates, "warmup_updates": warmup}


def memory():
    import psutil
    import torch

    from nlp_text_classification.training.resources import memory_available

    info = psutil.Process().memory_info()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    values = memory_available()
    return {
        "gpu_free_bytes": free, "gpu_total_bytes": total,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "rss_bytes": info.rss, "peak_rss_bytes": getattr(info, "peak_wset", info.rss),
        "process_commit_bytes": getattr(info, "pagefile", info.vms),
        "peak_process_commit_bytes": getattr(info, "peak_pagefile", info.vms),
        "host_free_physical_bytes": values.physical, "host_free_commit_bytes": values.commit,
    }


def environment():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    packages = ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                "huggingface-hub", "safetensors", "tokenizers", "numpy", "psutil")
    return {
        "python": sys.version.split()[0], "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "bf16": torch.cuda.is_bf16_supported(), "delta_net_fast_path": qwen.is_fast_path_available,
        "delta_net_backend": ("fla" if qwen.chunk_gated_delta_rule else "torch"),
        "causal_conv_backend": ("causal_conv1d" if qwen.causal_conv1d_fn else "torch"),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def configure_device():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_name() != (
        "NVIDIA GeForce RTX 4070 Ti SUPER"
    ) or not torch.cuda.is_bf16_supported():
        raise RuntimeError("S3 requires the RTX 4070 Ti SUPER and BF16 CUDA")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kernel_smoke():
    import bitsandbytes as bnb
    import torch

    from nlp_text_classification.training.qwen_s3 import seed_all

    seed_all(42)
    quant = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                              quant_type="nf4", compress_statistics=True).to("cuda")
    quant.requires_grad_(False)
    head = torch.nn.Linear(64, 28, device="cuda", dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), foreach=False)
    features = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    before = head.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = head(quant(features))
    output.float().square().mean().backward()
    opt.step()
    if features.grad is None or not torch.isfinite(features.grad).all() or (
        torch.equal(before, head.weight)
    ):
        raise ValueError("NF4 CUDA forward/backward/AdamW smoke failed")
    return {"backend": "CUDA NF4/BF16", "head_updated": True, "input_grad_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tokens", "kernel", "benchmark", "recover", "report"))
    parser.add_argument("--attempt", default="preflight")
    args = parser.parse_args()
    if not args.attempt.isascii() or not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must consist of ASCII letters, digits and hyphens")
    configure_compute_threads()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    suite = load_yaml(SUITE)
    root = ROOT / args.attempt
    root.mkdir(parents=True, exist_ok=True)
    progress = Progress(root / "progress.md", " ".join(sys.orig_argv),
                        title="Batch B S3 preflight progress")
    if args.mode == "tokens":
        _, _, audit, _ = inputs(suite)
        write_atomic_json(root / "tokenization.json", audit)
        progress.emit("TOKENIZATION", "Train/validation Qwen input audit complete",
                      selected_length=audit["selected_length"])
        return
    if args.mode == "report":
        from nlp_text_classification.evaluation.batch_b_s3_preflight import deliver

        deliver(root, suite, progress)
        return
    with exclusive_queue(ROOT.parent / ".batch-a-gpu.lock"):
        if args.mode == "kernel":
            configure_device()
            record = {"status": "PASS", "environment": environment(), "evidence": kernel_smoke()}
            write_atomic_json(root / "kernel.json", record)
            progress.emit("KERNEL", "NF4 CUDA forward, backward and AdamW smoke passed")
            return
        with protect(progress, process_limit=suite["preflight"]["process_commit_limit_gib"] * GIB):
            configure_device()
            from nlp_text_classification.experiments.batch_b_s3_preflight import benchmark, recover

            action = benchmark if args.mode == "benchmark" else recover
            try:
                evidence = action(root, suite, progress)
            except Exception as exc:
                progress.emit("FAILED", str(exc), mode=args.mode, exception=type(exc).__name__)
                raise
            print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

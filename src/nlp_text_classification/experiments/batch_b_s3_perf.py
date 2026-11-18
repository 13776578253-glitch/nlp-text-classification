"""Compare the same Qwen QLoRA updates across OS and kernel backends."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

from nlp_text_classification.experiments.batch_b_s3_resources import ROOT, SUITE
from nlp_text_classification.experiments.suite import load_yaml, read_json
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.resources import GIB, configure_compute_threads, protect

MATCHED_WARMUP = 5
MATCHED_STABLE = 8
PERF = ROOT / "performance"
INITIAL = PERF / "shared-initialization"


class Timing:
    def __init__(self):
        self.entries = defaultdict(list)

    @contextmanager
    def span(self, name):
        import torch

        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        try:
            yield
        finally:
            end.record()
            self.entries[name].append((begin, end, time.perf_counter() - wall))

    def checkpoint_contexts(self):
        return nullcontext(), self.span("checkpoint_recompute")

    def results(self):
        import torch

        torch.cuda.synchronize()
        result = {name: {"event_seconds": sum(a.elapsed_time(b) for a, b, _ in values) / 1000,
                         "host_span_seconds": sum(row[2] for row in values),
                         "spans": len(values)}
                  for name, values in self.entries.items()}
        result["backward"]["excluding_recompute_event_seconds"] = (
            result["backward"]["event_seconds"] - result["checkpoint_recompute"]["event_seconds"]
        )
        return result


def kernel_calls(model):
    layer = model.get_base_model().model.layers[0].linear_attn
    evidence, originals = {}, {}
    for name in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
        func = getattr(layer, name)
        evidence[name] = {"implementation": (
            f"{func.__module__}.{func.__name__}" if func else "torch.nn.functional.conv1d"
        ), "calls": 0, "backward_outputs": 0}
        if func is None:
            continue
        originals[name] = func

        def record(*args, _func=func, _name=name, **kwargs):
            row = evidence[_name]
            row["calls"] += 1
            value = _func(*args, **kwargs)
            output = value[0] if isinstance(value, tuple) else value
            if output.requires_grad:
                def gradient(grad):
                    row["backward_outputs"] += 1
                    return grad

                output.register_hook(gradient)
            return value

        setattr(layer, name, record)
    return layer, originals, evidence


def _package_versions():
    versions = {}
    for name in ("triton", "triton-windows", "fla-core", "flash-linear-attention", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _profile_update(model, update):
    layer, originals, kernels = kernel_calls(model)
    timing = Timing()
    base = model.get_base_model()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False,
                                       "context_fn": timing.checkpoint_contexts},
    )
    started = time.perf_counter()
    outcome = update(timing)
    detail = timing.results()
    elapsed = time.perf_counter() - started
    for name, func in originals.items():
        setattr(layer, name, func)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return {"seconds": elapsed, "phases": detail, "kernels": kernels, "update": outcome}


def run(root, suite, progress, backend, *, matched_telemetry=False):
    import numpy as np
    import torch

    from nlp_text_classification.data.audit import sha256_file
    from nlp_text_classification.evaluation.batch_metrics import write_predictions
    from nlp_text_classification.experiments.batch_b_s3 import (
        environment,
        inputs,
        memory,
        optimizer_and_schedule,
    )
    from nlp_text_classification.experiments.batch_b_s3_preflight import (
        _build,
        _indices,
        _pooling_probe,
        _publish,
        _rng_probe,
        _state_evidence,
        _stress,
        _tensor_array,
        _update,
        _validation,
    )
    from nlp_text_classification.training.qwen_s3 import restore_rng, rng_state, seed_all
    from nlp_text_classification.training.transformer_runner import (
        DevicePartition,
        threshold_record,
    )

    if (root / "updates.jsonl").exists():
        raise FileExistsError("Use a fresh comparison attempt")
    env = environment()
    expected = {"fallback": ("torch", "torch"), "wsl": ("fla", "causal_conv1d"),
                "windows-fla": ("fla", "torch")}
    if (env["delta_net_backend"], env["causal_conv_backend"]) != expected[backend]:
        raise ValueError(f"Requested {backend}, loaded {env}")
    progress.emit("INPUT", f"Verifying snapshot and train/validation inputs for {backend}")
    snapshot, data, tokens, encoded = inputs(suite)
    write_atomic_json(root / "tokenization.json", tokens)
    seed_all(suite["preflight"]["seed"])
    start_memory = memory()
    torch.cuda.reset_peak_memory_stats()
    baseline = backend == "fallback"
    model, rank, head, loading, lora, load_memory = _build(
        snapshot, suite, tokens["label_order"], saved=None if baseline else INITIAL / "adapter",
    )
    write_atomic_json(root / "load-audit.json", loading)
    write_atomic_json(root / "lora-audit.json", lora)
    optimizer, scheduler, plan = optimizer_and_schedule(
        rank, head, suite["training"], len(data.train),
    )
    start_state = _state_evidence(rank, head, optimizer)
    if baseline and not INITIAL.exists():
        with tempfile.TemporaryDirectory(dir=PERF) as temporary:
            staged = Path(temporary) / "initial"
            staged.mkdir()
            model.save_pretrained(staged / "adapter", safe_serialization=True,
                                  save_embedding_layers=False)
            torch.save(rng_state(), staged / "rng.pt")
            write_atomic_json(staged / "start.json", start_state)
            staged.rename(INITIAL)
    if start_state != read_json(INITIAL / "start.json"):
        raise ValueError("Performance paths must start from identical adapter/head bytes")
    restore_rng(torch.load(INITIAL / "rng.pt", weights_only=True, map_location="cpu"))
    pooling = _pooling_probe(model, tokens["policy"]["pad_token_id"])
    stress = _stress(model, rank, head, suite["preflight"]["stress_length"])
    if not all(stress["gradients"][group]["nonzero_tensors"]
               for group in ("lora_b", "head")):
        raise ValueError("Velocity test produced no gradient")
    if not matched_telemetry:
        _, _, initial_scores = _validation(model, data, encoded["validation"], suite)
        _tensor_array(root / "initial-scores.npy", initial_scores)
    progress.emit("FUNCTIONAL", "Official head, native pooling, BCE and 128-token backward passed")
    params = [*rank.values(), *head.values()]
    order = _indices(data, suite)
    effective = suite["training"]["accumulation_steps"]
    train = DevicePartition(encoded["train"], "cuda")
    measured = 2 if matched_telemetry else (0 if baseline else MATCHED_STABLE)
    history = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def update(step, trace=None):
        return _update(model, optimizer, scheduler, suite, train,
                       order[step * effective:(step + 1) * effective], params, trace=trace)

    with (root / "updates.jsonl").open("x", encoding="utf-8") as stream:
        for step in range(MATCHED_WARMUP + measured):
            started = time.perf_counter()
            if step == MATCHED_WARMUP - 1 and not matched_telemetry:
                profile = _profile_update(model, lambda trace, step=step: update(step, trace))
                result = profile["update"]
                write_atomic_json(root / "phase-profile.json", profile)
            else:
                result = update(step)
            row = {"step": step + 1, "seconds": time.perf_counter() - started, **result}
            history.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            progress.emit("UPDATE", f"{backend} {step + 1}/{MATCHED_WARMUP + measured}",
                          seconds=round(row["seconds"], 3), loss=round(row["loss"], 5))
    timing = memory()
    report = {
        "status": "PASS", "backend": backend, "os": platform.platform(), "environment": env,
        "accelerator_packages": _package_versions(), "model_revision": suite["model_revision"],
        "max_length": tokens["selected_length"], "micro_batch_size": 1,
        "rows_per_update": effective, "warmup_updates": MATCHED_WARMUP,
        "stable_updates": measured, "pooling": pooling, "stress": stress,
        "load_memory": load_memory, "start_memory": start_memory, "training_memory": timing,
        "initial": start_state,
        "matched_window_row_ids": [
            data.train[int(index)].sample_id for index in
            order[MATCHED_WARMUP * effective:(MATCHED_WARMUP + MATCHED_STABLE) * effective]
        ],
        "matched_window_lengths": encoded["train"].lengths[
            order[MATCHED_WARMUP * effective:(MATCHED_WARMUP + MATCHED_STABLE) * effective]
        ].tolist(),
    }
    trained = _state_evidence(rank, head, optimizer)
    if any(trained[key] == start_state[key] for key in trained):
        raise ValueError("LoRA, score or optimizer did not update")
    if not matched_telemetry:
        for name in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
            kernel = profile["kernels"][name]
            if kernel["implementation"] != "torch.nn.functional.conv1d" and (
                not kernel["calls"] or not kernel["backward_outputs"]
            ):
                raise ValueError(f"No verified forward/backward calls: {name}")
    if measured:
        seconds = np.asarray([row["seconds"] for row in history[MATCHED_WARMUP:]])
        report |= {"samples_per_second": measured * effective / seconds.sum(),
                   "step_seconds_median": float(np.median(seconds)),
                   "stable_seconds": float(seconds.sum())}
    write_atomic_json(root / "performance.json", report)
    if baseline:
        return
    rows, targets, scores = _validation(model, data, encoded["validation"], suite)
    threshold = threshold_record(targets, scores, False, suite["selection"])
    write_predictions(root / "predictions_validation.jsonl", rows, targets, scores,
                      threshold["threshold"])
    cursor = MATCHED_WARMUP + measured
    files = _publish(root, model, optimizer, scheduler, cursor, order)
    rng_probe = _rng_probe()
    following = update(cursor)
    write_atomic_json(root / "benchmark.json", {
        "pid": os.getpid(), "suite_sha256": sha256_file(SUITE), "options": plan,
        "threshold": threshold, "trained": trained, "checkpoint_files": files,
        "reference": {"outcome": following, "state": _state_evidence(rank, head, optimizer),
                      "rng_probe": rng_probe},
    })
    progress.emit("PERF_COMPLETE", f"{backend} matched window and checkpoint complete",
                  samples_per_second=round(report["samples_per_second"], 3))


def main():
    from nlp_text_classification.experiments.batch_b_s3 import configure_device
    from nlp_text_classification.experiments.batch_scheduler import exclusive_queue

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("fallback", "wsl", "windows-fla"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--matched-telemetry", action="store_true")
    args = parser.parse_args()
    if not args.attempt.isascii() or not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must consist of ASCII letters, digits and hyphens")
    if args.matched_telemetry and (args.backend != "fallback" or args.recover):
        parser.error("matched telemetry replays two Windows fallback updates")
    configure_compute_threads()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    suite = load_yaml(SUITE)
    root = PERF / args.attempt
    root.mkdir(parents=True, exist_ok=True)
    progress = Progress(root / "progress.md", " ".join(os.sys.orig_argv),
                        title=f"S3 performance · {args.backend}")
    with exclusive_queue(ROOT.parent / ".batch-a-gpu.lock"), protect(
        progress, process_limit=suite["preflight"]["process_commit_limit_gib"] * GIB,
    ):
        configure_device()
        if args.recover:
            from nlp_text_classification.experiments.batch_b_s3_preflight import recover

            recover(root, suite, progress)
        else:
            run(root, suite, progress, args.backend, matched_telemetry=args.matched_telemetry)


if __name__ == "__main__":
    main()

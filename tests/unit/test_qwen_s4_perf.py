"""Unit tests for Batch B S4 telemetry, padding statistics, trace parsing, and batching."""

from __future__ import annotations

import json

import numpy as np
import torch

from nlp_text_classification.experiments.batch_b_s4_runner import (
    compare_scores,
    iter_micro_batches,
    snapshot_trainable_state,
)
from nlp_text_classification.experiments.batch_b_s4_telemetry import (
    build_time_budget,
    compute_padding_stats,
    merge_intervals,
    parse_chrome_trace,
)


def test_padding_stats_dynamic_vs_static_and_sorted():
    lengths = np.asarray([3, 31, 4, 29, 8, 25, 6, 30, 5, 28, 7, 27, 9, 24, 10, 26], dtype=np.int32)
    static = compute_padding_stats(lengths, batch_size=8, multiple=64, strategy="static_64")
    unsorted = compute_padding_stats(lengths, batch_size=8, multiple=8, strategy="dynamic_unsorted")
    sorted_eff = compute_padding_stats(
        lengths, batch_size=8, multiple=8, strategy="dynamic_intra_effective_sorted",
    )
    assert static["padded_tokens"] == 16 * 64
    assert unsorted["padded_tokens"] == 16 * 32
    assert sorted_eff["padded_tokens"] == 8 * 16 + 8 * 32
    assert sorted_eff["valid_tokens"] == static["valid_tokens"] == int(lengths.sum())
    assert sorted_eff["valid_token_ratio"] > unsorted["valid_token_ratio"] > (
        static["valid_token_ratio"]
    )


def test_merge_intervals_and_parse_chrome_trace(tmp_path):
    assert merge_intervals([(10.0, 20.0), (15.0, 30.0), (40.0, 50.0)]) == 30.0
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "kDequantizeBlockwise", "ts": 100.0, "dur": 40.0},
            {"ph": "X", "cat": "kernel", "name": "nvjet_tst_128x16", "ts": 120.0, "dur": 30.0},
            {"ph": "X", "cat": "kernel", "name": "chunk_gated_delta_fwd", "ts": 200.0, "dur": 50.0},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 50.0, "dur": 10.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 10.0,
             "dur": 5.0},
        ]
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    parsed = parse_chrome_trace(trace_path, wall_seconds=0.000250)
    assert parsed["kernel_count"] == 3
    assert parsed["memcpy_memset_count"] == 1
    assert abs(parsed["kernel_active_seconds"] - 100e-6) < 1e-12
    assert abs(parsed["gpu_all_active_seconds"] - 110e-6) < 1e-12
    assert abs(parsed["launch_gap_seconds"] - 140e-6) < 1e-12
    assert parsed["buckets"]["bnb_nf4_dequant_or_gemm"]["calls"] == 1
    assert parsed["buckets"]["cublas_gemm"]["calls"] == 1
    assert parsed["buckets"]["fla_gated_delta_rule"]["calls"] == 1
    assert parsed["cuda_sync_calls"] == 1


def test_iter_micro_batches_preserves_sample_set_and_partial_tail_weights():
    indices = np.asarray([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int64)
    lengths = np.zeros(30, dtype=np.int32)
    lengths[indices] = [30, 5, 25, 8, 20, 4, 19, 7, 15, 6, 12]
    chunks = iter_micro_batches(indices, lengths, micro_batch_size=8,
                                sort_mode="intra_update_sorted")
    assert [len(c) for c in chunks] == [8, 3]
    flat = np.concatenate(chunks)
    assert set(flat.tolist()) == set(indices.tolist())
    assert lengths[chunks[0]].max() <= lengths[chunks[1]].min()
    # Partial tail weighting: sum(len(chunk) / len(indices)) == 1.0
    assert abs(sum(len(c) / len(indices) for c in chunks) - 1.0) < 1e-12


def test_compare_scores_and_time_budget():
    ref = np.full((4, 28), 0.49, dtype=np.float32)
    cand = ref.copy()
    cand[0, 0] = 0.51
    diff = compare_scores(ref, cand)
    assert diff["threshold_0_5_flips"] == 1
    assert diff["borderline_flips"] == 1
    assert abs(diff["prob_max_abs"] - 0.02) < 1e-5
    budget = build_time_budget(train_sps=2.0, val_sps=40.0)
    assert budget["1_seed_5_epochs"]["train_hours_neutral"] > 0
    assert budget["3_seeds_5_epochs"]["scenarios_hours"]["conservative"] > (
        budget["3_seeds_5_epochs"]["scenarios_hours"]["optimistic"]
    )
    assert torch.tensor([1.0]).item() == 1.0


def test_snapshot_trainable_state_clones_tensors():
    rank = {"a": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
    head = {"score.weight": torch.nn.Parameter(torch.tensor([[3.0, 4.0]]))}
    snap = snapshot_trainable_state(rank, head)
    with torch.no_grad():
        rank["a"].zero_()
        head["score.weight"].fill_(9.0)
    assert torch.equal(snap["a"], torch.tensor([1.0, 2.0]))
    assert torch.equal(snap["score.weight"], torch.tensor([[3.0, 4.0]]))

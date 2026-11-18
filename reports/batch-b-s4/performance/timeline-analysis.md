# Batch B S4 Timeline, Kernel & Operator Analysis (`timeline-analysis.md`)

**Trace File:** `artifacts/batch-b-s4/performance/traces/s4_1_baseline_update6.json` (`773,778,141` bytes)  
**Trace SHA-256:** `a2f8146d2d874dff5d0923918895deff46ddf41a559a51b590d1960fb80bf0e6`  
**Summary JSON:** [`artifacts/batch-b-s4/performance/profile-summary.json`](../../artifacts/batch-b-s4/performance/profile-summary.json)  
**Top-10 Kernel CSV:** [`operator-top10.csv`](./operator-top10.csv)

> **Instrumentation & Attribution Discipline:** Following Section S4-1 rules, uninstrumented wall time (`14.5624 s/update`), CUDA event spans during Update 5 (`13.2719 s` wall, module-level hooks including checkpoint recomputation), and CUPTI trace measurements during Update 6 (`25.2603 s` profiler wall, `16.9997 s` trace span) are kept strictly separate. Because WSL2 virtualizes the vGPU without bare-metal Nsight Systems/Compute hardware performance counters, SM/Tensor Core active-cycle hardware counters are marked **unavailable** and are not fabricated; instead, CUPTI trace-derived `est. achieved occupancy %`, non-overlapping kernel active intervals, CPU launch gaps, and `nvidia-smi` power/clock metrics are reported.

---

## 1. Timeline Summary: Wall Clock vs. Active GPU Time vs. CPU Launch Gap

| Measurement Aperture (`S4_0_baseline` / `micro=1, accum=16, GC=full`) | Value | Share / Interpretation |
|---|---:|---|
| Uninstrumented steady update mean (`S4_0_baseline`) | **14.5624 s** | Clean reference update wall time (`1.0987 samples/s`) |
| Profiled update 6 wall time (`torch.profiler` active) | **25.2603 s** | Includes ~`10.7 s` profiler collection overhead |
| Profiled trace timestamp span (`trace_span_seconds`) | **16.9997 s** | First-to-last GPU activity window |
| Merged union GPU kernel active time (`kernel_active_seconds`) | **4.0657 s** | Non-overlapping GPU compute intervals across 272,574 kernels |
| Merged union GPU kernel + memcpy/memset (`gpu_all_active_seconds`) | **4.0873 s** | **True active GPU time per optimizer update (`28.1%` of `14.56 s` uninstrumented wall)** |
| Profiled CPU launch/scheduling gap (`wall - gpu_all_active`) | **21.1729 s** | `83.82%` of profiled wall; **`~10.47 s` (~`71.9%`) against uninstrumented `14.56 s`** |
| Total CUDA kernels (`kernel_count`) | **272,574** | `17,036` kernel launches per micro-step (`÷ 16`) |
| Total `gpu_memcpy` / `gpu_memset` (`memcpy_memset_count`) | **9,203** | `9,155` memcpy + `48` memset |
| Total CUDA runtime API calls (`cuda_runtime`) | **318,638** | Host-to-driver dispatch overhead |
| Explicit stream synchronizations (`cudaStreamSynchronize`) | **52 calls** (`15.26 ms` CPU) | Sync count is small; **dispatch queue starvation**, not sync stalls, causes the gap |
| Weighted CUPTI `est. achieved occupancy %` | **35.73%** | Hardware Tensor Core % unavailable in WSL2 |

### Key Clarification on `nvidia-smi` `36.1%–50.0%` Utilization
In S3 (`36.1%`) and S4-0 (`50.0%`), `nvidia-smi` samples at coarse intervals (~250–1000 ms) whether *at least one kernel is active at the instant of sampling*. CUPTI proves two simultaneous facts:
1. Over a `14.56 s` update, **only `4.087 s` (~28%) has any GPU kernel running** — `~72%` of wall time sits inside inter-kernel CPU submission gaps while power stays at **`48.6 W`** and SM clock idles near **`1,467 MHz`**.
2. Even while running, `micro=1` executes small GEMMs at **`4.94%–11.17%` CUPTI est. occupancy**. Raising `micro_batch_size` from `1 -> 16` coalesces 16 passes into 1, raising steady GPU power to **`150.6 W`**, SM clock to **`2,775 MHz`**, and delivering **`18.89×`** higher training throughput.

---

## 2. Update 5 Module CUDA Event Breakdown (`update5_wall_seconds = 13.2719 s`)

> **Note:** Module forward hooks fire during *both* the initial forward pass (`16` micro-steps) and the non-reentrant gradient checkpoint recomputation pass (`16` micro-steps), so forward event totals include recompute passes and CPU enqueue latency inside each hook span; they must not be summed directly with CUPTI exclusive times.

| Module Span (`module_event_spans_update5`) | Calls / Update | Span Total (s) | Role |
|---|---:|---:|---|
| `checkpoint_recompute_total` | 512 | **4.3580 s** (`32.8%`) | 32 layers × 16 micro-steps recomputed |
| `fwd:deltanet_linear` | 3,840 | 2.2346 s | 24 layers × NF4 `in_proj_*` & `out_proj` |
| `fwd:mlp` | 512 | 1.1549 s | 32 layers × `gate`/`up`/`down` NF4 MLPs |
| `fwd:rmsnorm` | 2,560 | 0.8178 s | FP32 internal upcast RMSNorm kernels |
| `fwd:fla_chunk_gdr` | 768 | 0.7231 s | `fla.chunk_gated_delta_rule` (span includes Triton Python launch) |
| `fwd:full_attn_linear` | 1,024 | 0.5412 s | 8 layers × NF4 `q/k/v/o_proj` |
| `fwd:deltanet_norm` | 768 | 0.1376 s | `FusedRMSNormGated` |
| `fwd:full_attn_sdpa` | 256 | 0.0523 s | 8 full attention SDPA calls |
| `fwd:causal_conv1d` | 768 | 0.0065 s | Fast `causal_conv1d_fn` active span |
| `fwd:final_norm` / `rotary_emb` / `score_head` / `embed_tokens` | 16 each | 0.0070 s combined | Embedding, RoPE, classifier head |

---

## 3. Update 6 Kernel Category Buckets (`4,087.34 ms` Total GPU Kernel+Memcpy Sum)

| Bucket | Calls | Total GPU (ms) | Share of GPU Time (%) | Mean / Call (µs) | Optimizable Tier & Notes |
|---|---:|---:|---:|---:|---|
| `cublas_gemm` | 51,392 | **1,851.29** | **45.29%** | 36.02 | High (`micro` batching shifts `16×16` WMMA tiles -> larger `64×64`/`64×128` TensorOp tiles) |
| `bnb_nf4_dequant_or_gemm` | 23,808 | **1,303.97** | **31.90%** | 54.77 | High (amortized `16×` by `micro=16`, or `0` in Category B BF16) |
| `elementwise_cast_norm_act` | 185,762 | **678.66** | **16.60%** | 3.65 | Medium (64,880 FP32<->BF16 casts, FP32 add/mul/dropout; launch overhead huge) |
| `fla_gated_delta_rule` | 8,832 | **182.95** | **4.48%** | 20.71 | GPU fast; **CPU host cost is 3,065.1 ms (`16.7×` GPU)** at `micro=1`! |
| `sdpa_attention` | 432 | 26.05 | 0.64% | 60.29 | Very low (only 8 layers, short `seq<=40`) |
| `memcpy_memset` | 9,203 | 21.68 | 0.53% | 2.36 | Low |
| `causal_conv1d` | 1,152 | 8.63 | 0.21% | 7.49 | Very low |
| `other` | 1,186 | 8.59 | 0.21% | 7.24 | Negligible |
| `optimizer_adamw` | 10 | 5.53 | 0.14% | 552.70 | Negligible (`44.4 ms` op self-CUDA) |

---

## 4. Top-10 Most Expensive CUDA Kernels (`operator-top10.csv`)

| Rank | Kernel Name (Demangled Short) | Bucket | Calls | Total (ms) | Mean (µs) | CUPTI Est. Occ. (%) | Share (%) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, 2>` | `bnb_nf4_dequant_or_gemm` | 11,904 | **1,245.70** | 104.65 | 82.51% | **30.64%** |
| 2 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8` | `cublas_gemm` | 9,904 | **467.98** | 47.25 | **11.17%** | **11.51%** |
| 3 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` | `cublas_gemm` | 3,568 | **245.98** | 68.94 | **4.94%** | **6.05%** |
| 4 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_nn_align8` | `cublas_gemm` | 1,056 | 213.87 | 202.52 | 15.00% | 5.26% |
| 5 | `ampere_bf16_s16816gemm_bf16_64x128_ldg8_f2f_stages_64x3_tn` | `cublas_gemm` | 576 | 197.76 | 343.33 | 0.00%* | 4.86% |
| 6 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8` | `cublas_gemm` | 1,648 | 147.14 | 89.28 | 35.93% | 3.62% |
| 7 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align8` | `cublas_gemm` | 4,176 | 135.28 | 32.39 | 32.86% | 3.33% |
| 8 | `bfloat16_copy_kernel_cuda` (FP32<->BF16 cast) | `elementwise_cast_norm_act` | 40,624 | 123.53 | 3.04 | 15.67% | 3.04% |
| 9 | `LoadWithCast<1> / StoreWithCast<1>` | `elementwise_cast_norm_act` | 24,256 | 108.17 | 4.46 | 23.25% | 2.66% |
| 10 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_nn` | `cublas_gemm` | 480 | 90.46 | 188.46 | 0.00%* | 2.23% |

*\*Note: CUPTI reports `est_occupancy_pct = 0.0` for certain SASS Ampere/Ada GEMM or Triton kernels where static register/shared-mem trace metadata is omitted by the driver; these represent telemetry omissions rather than actual 0% SM occupancy.*

---

## 5. CPU Dispatcher Overhead Top Operators (`top_profiler_ops`)

Comparing self-CPU time vs. self-CUDA time exposes why `micro=1` starves the RTX 4070 Ti SUPER:
- **`ChunkGatedDeltaRuleFunction` (`768` fwd) + `ChunkGatedDeltaRuleFunctionBackward` (`384` bwd):** **`3,065.11 ms` self-CPU** (`1,664.42 + 1,400.69 ms`) vs. only **`182.95 ms` self-CUDA** (`16.75×` more time dispatching Python/Triton wrappers than running GPU math!).
- **`aten::mm` (`43,696` calls):** **`2,661.48 ms` self-CPU** vs. **`1,851.29 ms` self-CUDA**.
- **`bitsandbytes::dequantize_blockwise` + `dequantize_4bit` (`11,904` pairs):** **`944.94 ms` self-CPU** (`614.01 + 330.92 ms`) vs. **`1,303.97 ms` self-CUDA** (`1,245.70 + 58.27 ms`).
- **`aten::copy_` (`79,022` calls), `aten::mul` (`30,993`), `aten::native_dropout` (`7,936`), `aten::add` (`15,435`):** Combined **`1,116.07 ms` self-CPU** (`374.86 + 354.38 + 189.57 + 137.27 ms` + `191.10 ms` `aten::add_`).

**Engineering Conclusion:** Reducing per-update forward+recompute+backward cycles via `micro_batch=16, accum=1` (16× fewer Python/CUDA launches) paired with `gc_mode="partial_16"` (halving recompute passes) directly attacks both the `~10.47 s` CPU launch gap and `45%+` of NF4 dequantize/GEMM/cast volume simultaneously.

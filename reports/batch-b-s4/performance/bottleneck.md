# Batch B S4 Performance Bottleneck Analysis (`bottleneck.md`)

**Baseline Reference:** S3 WSL2 (`1.1586 samples/s`, `13.81 s/update`) / S4-0 Replay (`1.0987 samples/s`, `14.5624 s/update`)  
**Profiling Evidence:** [`artifacts/batch-b-s4/performance/profile-summary.json`](../../artifacts/batch-b-s4/performance/profile-summary.json), [`operator-top10.csv`](./operator-top10.csv), [`experiments.csv`](./experiments.csv)

---

## Top 3 True Hardware & Runtime Bottlenecks (Ranked by Measured Impact)

### Rank #1 — Micro-Batch = 1 Extreme Kernel Fragmentation (~72% CPU Launch Gap, Tiny WMMA GEMMs, 23,808× NF4 Dequantization)
- **Evidence:**
  1. **CUPTI Trace (Update 6):** `micro_batch=1, accum=16` issues **`272,574` CUDA kernels** and `318,638` CUDA runtime API calls per optimizer update (`16` forward + `16` checkpoint-recompute + `16` backward passes).
  2. **Merged GPU Active Time vs Wall Clock:** Across all 272,574 kernels and 9,203 memcpy/memset events, total non-overlapping GPU busy duration (`gpu_all_active_seconds`) is only **`4.0873 s`** (profiled wall `25.2603 s` -> `83.82%` gap; even against uninstrumented wall `14.5624 s`, **`~10.47 s` or ~71.9% is pure CPU dispatch gap between short kernels**).
  3. **Hardware Telemetry Underutilization:** At `micro=1`, mean power draw sits at **`48.6 W`** and mean SM clock throttles to **`1,467 MHz`** because short Python/Triton dispatches cannot feed the GPU continuously (`ChunkGatedDeltaRuleFunction` + backward alone consume **`3,065.11 ms` CPU wall** to issue `182.95 ms` of GPU kernels — a `16.7×` CPU-to-GPU bottleneck).
  4. **NF4 Weight Streaming Amplification:** At `micro=1`, weights are read and dequantized **16 times per optimizer update** (`11,904` calls to `kDequantizeBlockwise<__nv_bfloat16>` taking `1,245.70 ms` + `11,904` double-quant scale dequants taking `58.27 ms` = **`31.90%` of GPU kernel time**), while linear GEMMs shrink to `[16..40, K] @ [K, N]` executing `16×16` WMMA tiles at only **`4.94%–11.17%` CUPTI est. achieved occupancy**.
- **Controlled Experimental Proof (S4-2):** Keeping `effective_batch=16` mathematically identical and raising `micro_batch_size` alone (`GC=full` fixed):
  - `micro=1 -> 2 -> 4 -> 8 -> 16` scales training throughput **`1.099 -> 2.545 -> 4.468 -> 9.813 -> 16.233 samples/s`** (**`14.010×` S3**, **`14.77×` S4-0**), cutting update time from **`14.562 s -> 0.986 s`**, lifting GPU power `48.6 W -> 151.2 W` and SM clock `1,467 -> 2,762 MHz`, with **more** free VRAM (`6,057 MiB` free vs `5,202 MiB`)!

---

### Rank #2 — Full 32-Layer Non-Reentrant Gradient Checkpointing Recomputation (~32.8% of Compute & Re-Launch Volume)
- **Evidence:**
  1. **Update 5 CUDA Event Instrumentation:** Under `micro=1, gc=full`, `checkpoint_recompute_total` spans **`4.3580 s`** out of `13.2719 s` (**`32.8%`** of update wall time), re-running every GatedDeltaNet (`384` extra recompute calls), `causal_conv1d`, `rmsnorm`, `full_attention`, `mlp`, and `bnb.Linear4bit` NF4 dequantize pass (`~1/3` of all `23,808` dequants and `272k` kernel launches!).
- **Controlled Experimental Proof (S4-3):**
  - On `micro=1` (`S4_0 -> GC_micro1_partial16 -> GC_micro1_off`): `1.0987 -> 1.3569 -> 1.9343 samples/s` (`+76.1%` net speedup).
  - On `micro=8`: `9.8127 (full) -> 11.5004 (partial_16) -> 13.2158 (partial_attn8)` (`+34.7%` net speedup while keeping `3,417 MiB` free VRAM); `GC=off` drops to `858 MiB` free (`FAIL_VRAM_HEADROOM`).
  - On `micro=16`: **`16.2329 (full) -> 20.5726 samples/s (partial_16)`** (`+26.7%` additional gain over `B5_micro16`, keeping **`3,094 MiB`** free in multi-run Stage C and **`4,167 MiB`** free in isolated Stage D `A_winner_foreach_evalfast` with `AdamW(foreach=True)` -> **`21.8852 samples/s`**!). Full removal (`GC_micro16_off`, `26.04 samples/s`) and `partial_attn8` (`23.57 samples/s`) fail the `>= 2,048 MiB` VRAM safety gate (`31 MiB` and `842 MiB` free respectively).

---

### Rank #3 — Validation `batch_size=1` Bottleneck & Hidden FP32 Residual Stream / Elementwise Casts
This third tier pairs two verified sources of waste across training and end-to-end 5-epoch runtime:

1. **Validation `eval_batch_size=1` (`S3 = 4.79 samples/s` / `4.71 h` for 15 evals -> `4.95 h` at `S4 eval_b1`):**
   - **Controlled Experimental Proof (`validation-sweep.json` / `validation-matrix.csv`):** Full 5,416-row validation at `eval_batch=32, sort_by_length=True` jumps from `4.558 samples/s` (`b=1`) to **`130.951 samples/s` (`28.7×`)**, slashing all 15 validations (3 seeds × 5 epochs) from **`~4.71–4.95 hours` down to `0.1723 hours` (~10.3 minutes)** with `6,082 MiB` min free VRAM and only `4/1,344` borderline flips (`prob_max_abs = 0.0159`).
2. **FP32 Residual Stream & Cast Kernels (`16.60%` of Baseline GPU Kernel Time, `+2.2 GiB` VRAM):**
   - `peft.prepare_model_for_kbit_training` keeps embeddings, `conv1d`, and norms in FP32, producing `64,880` explicit FP32<->BF16 cast kernels per update (`bfloat16_copy_kernel_cuda` + `LoadWithCast`/`StoreWithCast`, totaling `231.7 ms` directly plus FP32 elementwise adds/multiplies/dropout in `elementwise_cast_norm_act = 678.66 ms` / `16.60%`).
   - **Controlled B1 Probe (`B1_nf4_bf16_stream` vs `A_winner_foreach_evalfast`):** Removing the forced FP32 stream saves **`2,232 MiB` free VRAM** (`6,399 MiB` vs `4,167 MiB` min free), **but does not add net speed at `micro=16`** (`21.3321` vs `21.8852 samples/s`) because `micro=16` is already dominated by BF16 GEMMs and NF4 dequantize memory traffic while `B1` triples S3 score divergence (`15/1,344` flips, `prob_max_abs = 0.0627`).

---

## Bottlenecks Ruled Out as Primary Causes
- **Dynamic Padding Overhead (`H3` falsified):** S3 already used dynamic `multiple=8` (`82.9%` valid ratio at `micro=1`), NOT fixed `64` (`26.0%`). Moreover, intra-update length sorting (`P5_mult8_sorted_micro8`: `7.17 samples/s`) ran **slower** than unsorted (`B4_micro8`: `9.81 samples/s`) due to dynamic tensor shape switching and lower clock stability (`1,994 vs 2,128 MHz`).
- **AdamW Optimizer Step (`0.14%` GPU time) & Tensor Preparation (`~10 ms`):** Negligible at `micro=1`; enabling `AdamW(foreach=True)` combined with clean isolated allocation in `A_winner_foreach_evalfast` yields a modest final improvement (`20.57 -> 21.89 samples/s`, update `0.778 -> 0.731 s`).
- **Pure FLA / `causal_conv1d` Kernel Compute Time:** On GPU alone, `fla_gated_delta_rule` takes only `182.95 ms` (`4.48%`) and `causal_conv1d` takes `8.63 ms` (`0.21%`) per update — the slowdown at `micro=1` came from `3,065 ms` of **CPU wrapper launch overhead**, not slow Triton/CUDA math inside those kernels.

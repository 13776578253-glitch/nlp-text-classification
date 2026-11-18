# Batch B S4 Accuracy Parity, Gradient Checks & Independent Recovery (`accuracy-and-recovery.md`)

**Verification Scope:** 48 paired validation rows (`48 × 28 = 1,344` label predictions), mixed-length 16-row extremes, full 5,416-row validation sweep, 13-step loss/gradient trajectories, and cross-PID checkpoint recovery (`PID 548 -> PID 747`).  
**Frozen Test Split (`test.jsonl`):** Untouched (`test_set_accessed: false`).

---

## 1. Training Parity vs. S3 WSL2 Baseline (`experiments.csv` / `result.json`)

Because `micro_batch_size` changes CUDA reduction order and RNG consumption per forward pass, cross-`micro` configurations are evaluated against the frozen S3 48-row post-13 probabilities using mean/p95/max probability differences, logit differences, threshold `0.5` flips (and whether flips sit within `[0.45, 0.55]` borderline), and gradient finiteness.

| Run ID | Category | Status | Init Prob Max vs S3 | Init Flips (/1,344) | Post-13 Prob Mean | Post-13 Prob Max | Post-13 Flips (/1,344) | Borderline Flips | Step-13 Loss (S3 = 1.0984) | Step-13 Grad Norm (S3 = 10.821) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `S4_0_baseline` (`micro=1`, `eval_b1`) | baseline | PASS | `1.192e-7` | 0 | 0.001888 | 0.015203 | 4 | 4 / 78 | 1.0987 | 10.815 |
| `B2_micro2` (`micro=2`, `GC=full`) | A | PASS | `1.192e-7` | 0 | 0.001908 | 0.014824 | 5 | 5 / 78 | — | — |
| `B3_micro4` (`micro=4`, `GC=full`) | A | PASS | `1.192e-7` | 0 | 0.001938 | 0.013680 | 3 | 3 / 78 | — | — |
| `B4_micro8` (`micro=8`, `GC=full`) | A | PASS | `1.192e-7` | 0 | 0.001913 | 0.017379 | 5 | 5 / 78 | — | — |
| `B5_micro16` (`micro=16`, `GC=full`) | A | PASS | `1.192e-7` | 0 | 0.001905 | 0.013648 | 4 | 4 / 78 | — | — |
| `GC_micro16_partial16` (`eval_b1`) | A | PASS | `1.192e-7` | 0 | 0.001905 | 0.013648 | 4 | 4 / 78 | — | — |
| **`A_winner_foreach_evalfast`** (`eval_b32_sorted`) | **A (Winner)** | **PASS** | `0.015940`* | 4* | **0.002152** | **0.018696** | **5** | **5 / 78** | **1.0988** | **10.826** |
| `P0_static64_micro1` | A | PASS | `1.192e-7` | 0 | 0.001847 | 0.021786 | 4 | 4 / 78 | — | — |
| `P2_static64_micro8` | A | PASS | `1.192e-7` | 0 | 0.001996 | 0.017863 | 3 | 3 / 78 | — | — |
| `P4_mult8_sorted_micro4` | A | PASS | `1.192e-7` | 0 | 0.001918 | 0.015168 | 5 | 5 / 78 | — | — |
| `P5_mult8_sorted_micro8` | A | PASS | `1.192e-7` | 0 | 0.001952 | 0.017883 | 4 | 4 / 78 | — | — |
| `P6_mult1_sorted_micro8` | A | PASS | `1.192e-7` | 0 | 0.001921 | 0.013945 | 4 | 4 / 78 | — | — |
| `B1_nf4_bf16_stream` (`eval_b32_sorted`) | B | PASS | `0.025873` | 8 | **0.006537** | **0.062748** | **15** | 15 / 78 | 1.1191 | 10.940 |
| `B2_bf16_lora_micro1_gcfull` | B | PASS | **`0.366599`** | **68** | **0.047610** | **0.411347** | **76** | 33 / 78 (43 outside) | 1.1284 | 11.074 |
| `B2_bf16_lora_micro8_gcoff` | B | FAIL_VRAM | **`0.366599`** | **68** | **0.047721** | **0.401196** | **75** | 32 / 78 (43 outside) | — | — |

*\*`A_winner_foreach_evalfast` runs initial evaluation with `eval_batch=32, sorted=True` (already verified identically on `eval_b32_sorted` vs `b1`: `prob_max_abs = 0.01594`, `4/1344` borderline flips); when evaluated at `eval_batch=1` (`GC_micro16_partial16`), initial `prob_max` is `1.192e-7` with `0` flips.*

### Key Takeaways on Numerical Accuracy
1. **Category A Drift Is Small & Strictly Borderline:** `S4_0_baseline` itself exhibits `0.001888` mean / `0.015203` max prob drift after 13 updates (`4` borderline flips), consistent with non-deterministic BF16/Triton backward atomics; S3 originally observed `0.00220` mean / `0.01811` max (`3` flips) even across Windows vs WSL backends. `A_winner_foreach_evalfast` sits directly in that exact range (`0.002152` mean, `0.018696` max, `5` borderline flips out of `1,344` predictions; logit mean abs `0.0182`, max `0.1250`), with step-13 loss matching S3 (`1.09879` vs `1.09844`, Δ=`+0.00035`) and step-13 gradient norm (`10.8258` vs `10.8214`).
2. **Category B (`B1` & `B2`) Alters Numerical Trajectory:**
   - `B1_nf4_bf16_stream` raises prob max diff 3.4× (`0.06275`) and flips 3× (`15/1,344`), plus step-13 loss lags S3 (`1.1191` vs `1.0984`).
   - `B2_bf16_lora` produces **substantial initial divergence (`0.3666` prob max, `1.9375` logit max, `68/1,344` step-0 flips including 38 non-borderline flips)** solely from removing 4-bit NF4 quantization noise across `Qwen3.5-4B`'s 24 GatedDeltaNet + 8 FullAttention layers — matching Unsloth's independent documentation warning on Qwen3.5 4-bit quantization differences. Thus `B2` cannot share S3/NF4 comparability without launching an explicitly separate B-class BF16 protocol.

---

## 2. Step-6 Non-Zero Gradient & Module Coverage Gate (`acceptance.json`)

Every run verified zero NaN/Inf across all 13 updates (`all_finite_losses_and_grad_norms = true`). Module gradient norms at Step 6 confirm **all 248 LoRA adapters (including all GatedDeltaNet projections) and `score` head receive healthy updates**:

| Config | `lora_a` Non-Zero / Total (Max Norm) | `lora_b` Non-Zero / Total (Max Norm) | `score` Head Non-Zero (Max Norm) |
|---|---:|---:|---:|
| `S4_0_baseline` | `248 / 248` (`0.00813`) | `248 / 248` (`1.08009`) | `1 / 1` (`11.1137`) |
| `GC_micro16_partial16` | `248 / 248` (`0.00830`) | `248 / 248` (`1.10767`) | `1 / 1` (`11.1191`) |
| **`A_winner_foreach_evalfast`** | **`248 / 248`** (`0.00830`) | **`248 / 248`** (`1.10714`) | **`1 / 1`** (`11.1232`) |
| `B1_nf4_bf16_stream` | `248 / 248` (`0.00839`) | `248 / 248` (`1.10511`) | `1 / 1` (`11.1412`) |
| `B2_bf16_lora_micro1_gcfull` | `248 / 248` (`0.00865`) | `248 / 248` (`1.16255`) | `1 / 1` (`11.0145`) |

---

## 3. Validation Batch & Length-Sort Numerical Parity (`validation-matrix.csv` / `validation-sweep.json`)

Compared against `eval_b1_unsorted` on both 48 fixed paired validation samples (`1,344` predictions) and 16 extreme length-mixed rows (`448` predictions, shortest 8 + longest 8):

| Eval ID | Rows | Speed (s/s) | 48-Row Prob Max vs `b1` | 48-Row Flips (/1,344) | Mixed-16 Prob Max vs `b1` | Mixed-16 Flips (/448) | Min Free VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eval_b1_unsorted` | 96 | 4.558 | 0.00000 | 0 | 0.00000 | 0 | 5,976 |
| `eval_b4_unsorted` | 512 | 19.528 | 0.01594 | 4 | 0.01594 | 0 | 6,070 |
| `eval_b8_unsorted` | 512 | 42.933 | 0.01594 | 4 | 0.01594 | 0 | 6,042 |
| `eval_b16_unsorted` | 512 | 81.082 | 0.01594 | 4 | 0.01594 | 0 | 6,064 |
| `eval_b32_unsorted` | 512 | 93.313 | 0.01594 | 4 | 0.01594 | 0 | 6,053 |
| `eval_b16_sorted_full5416` | 5,416 | 80.002 | 0.01594 | 4 | 0.01594 | 0 | 6,083 |
| **`eval_b32_sorted_full5416`** | **5,416** | **130.951** | **0.01594** | **4** | **0.01594** | **0** | **6,082** |

Notice that once `eval_batch > 1` activates batched BF16 kernels (`b4` through `b32`, whether sorted or unsorted), the 48-row diff vs `b1` stabilizes identically at `0.01594` max prob (`4` borderline flips out of `1,344`) and **`0 / 448` flips** on extreme length-mixed inputs — proving right-padded dynamic batching + last-content-token pooling works cleanly without cross-sample padding leakage.

---

## 4. Independent-PID Recovery Verification (`runs/A_winner_foreach_evalfast/recovery.json`)

`A_winner_foreach_evalfast` saved full state after update 13 (`PID 548`) and launched an independent WSL2 process (`PID 747`) to load `adapter/`, `resume.pt` (head weights + `AdamW(foreach=True)` state + linear scheduler + RNG state), verify validation predictions, RNG states, and run Step 14:

- **Status:** **`PASS`**
- **PIDs:** `original_pid = 548` -> `recovery_pid = 747`
- **Resume Cursor / Next Update:** `resume_cursor = 13`, `next_update = 14` (`exact_next_update = true`)
- **Validation Score Parity on Reload (`48` rows):** **`max_abs_score_error = 0.0`**, **`exact_score_match = true`** (`1.4757 s`)
- **RNG Replay (Python / NumPy / CPU Torch / CUDA Torch):** **`exact_rng_probe = true`**
- **Post-Step-14 State Digest Parity (`loss = 1.13757`, `grad_norm = 10.34149`):**
  - Exact SHA-256 match between original PID Step-14 continuation and recovered PID Step-14 continuation:
    - `lora_sha256`: `e42ef7cc6ced6d151dd32658fa1e6771023ba1434ab2d51187af258e517c3fba`
    - `head_sha256`: `3100127c8dea1370d6a65ea6479eca9ac9d20f1b84c6972727cd1f59fc19512b`
    - `optimizer_sha256`: `a0a8a22415e0e98b6c3cdffb3e3053d4e5cd496e1fb532b3eda563ac70a7a4ab`
- **Recovery Peak Process Memory:** `gpu_peak_allocated = 8,290.8 MiB` (`7.72 GiB`), `gpu_peak_reserved = 8,738.8 MiB`, `gpu_free = 6,633 MiB`, `recovery_seconds = 78.32 s`.

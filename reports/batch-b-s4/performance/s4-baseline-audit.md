# Batch B S4 Baseline Audit (`s4-baseline-audit.md`)

**Date:** 2026-11-18  
**Baseline Commit Frozen:** `124c1a2` (preserved on branch `s3-baseline-124c1a2` and untouched `main`)  
**S4 Working Branch:** `batch-b-s4`  
**Formal Training Started:** `false`  
**Frozen Test Split Touched:** `false` (`data/processed/goemotions/v1/test.jsonl` untouched)

---

## 1. Environment & Isolation Verification

All S4 work was isolated from the frozen S3 baseline (`reports/batch-b-s3/`, `artifacts/batch-b-s3/`, `configs/experiment_suites/batch_b_s3.yaml`, and WSL2 runtime `~/.cache/emotionlab-s3-perf/`):

- **WSL2 Isolated Virtual Environment:** `~/.cache/emotionlab-s4-perf/venv`, installed offline from S3 wheel cache (`~/.cache/emotionlab-s3-perf/uv-cache`), pinned lockfile `configs/batch_b/s3-wsl-runtime.lock.txt` (`sha256: 1494504c576a287f16bc8d69c505832daa73e152602cc96bd6af7da40beb9011`), and local `causal_conv1d-1.7.0` wheel (`uv pip check`: 74 packages compatible).
- **Hardware & Software Stack (observed via `reports/batch-b-s4/performance/environment.txt` & `recommended-training.json`):**
  - GPU: `NVIDIA GeForce RTX 4070 Ti SUPER` (SM `8.9`, 16 GiB GDDR6X), Windows Host Driver `610.88`, CUDA `12.8`
  - OS: `WSL2 Ubuntu 22.04.5 LTS`, Python `3.12.14`
  - Key packages: `torch==2.9.1+cu128`, `transformers==5.10.2`, `peft==0.19.0`, `bitsandbytes==0.49.2`, `triton==3.5.1`, `fla-core==0.5.1`, `flash-linear-attention==0.5.1`, `causal-conv1d==1.7.0`, `accelerate==1.16.0`
- **S3 Frozen Evidence SHA-256 Check (`acceptance.json` / `baseline.json`):** All 7 S3 frozen performance files matched their recorded `reports/batch-b-s3/performance/acceptance.json` digests bit-for-bit before and after S4 execution:
  - `comparison.json`: `14c0596d9ed561a366410c1af21e0d537faddf0999ce369d63e14c21deb6042d`
  - `model-resources.json`: `0df132ac8af94b906c52269613c4610aa53cbdae4b1a76a410dbb5c5f73ce47c`
  - `recommended-training.json`: `d87c49399c2c4d769e1b9ae2a1ecb86c9ec7dd7820ebce04cb2377b4c9505474`
  - `report.md`: `060ba737d2593edacef99be07503c5cf1865b1010ab3a3bec037a12d8531b79e`
  - `tokenization.json`: `7f1b04c67d9d4706b713487ae66cc5d9a41a3835823bfd662eb8ab9b7cf89229`
  - `wsl-load-audit.json`: `18a1d761a5cef0cecdfafcae0f647376c3bda5b1bc3b3b1bea5b5310a24de881`
  - `wsl-lora-audit.json`: `7e716bbc04d744f8129ec0236097522f05725f9c4893051bd30c86ef12c6cbeb`

---

## 2. Source Code Audit Findings

Direct inspection of repository training code (`batch_b_s3.py`, `qwen_s3.py`, `qwen_s3_streaming.py`, `qwen_s3_inputs.py`) and installed pinned packages (`transformers==5.10.2`, `peft==0.19.0`, `bitsandbytes==0.49.2`):

1. **Text-Only Architecture & Classifier Head Verified:**
   - Base class resolved from `Qwen3_5ForSequenceClassification` -> `Qwen3_5TextForSequenceClassification` (`modeling_qwen3_5.py`) inheriting `GenericForSequenceClassification`.
   - 32 transformer blocks in a `3:1` hybrid ratio: **24 GatedDeltaNet linear-attention blocks** (`fla-core 0.5.1` `chunk_gated_delta_rule` + `causal_conv1d 1.7.0`) and **8 FullAttention blocks** (`sdpa`).
   - Vision tower is **not** instantiated or executed; `lm_head` is **not** invoked. Output logits come exclusively from the `score` linear head (`hidden_size=2560 -> num_labels=28`), trained via `BCEWithLogitsLoss`.
2. **Right-Padding, Masking, and Last-Content-Token Pooling Verified:**
   - Preprocessing uses `add_special_tokens=False`, `pad_token_id=248044`, right padding, right truncation (`max_length=64`).
   - `GenericForSequenceClassification.forward` locates the last non-pad index via `last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)`, verifying only the final **content token** enters `score`. Padding tokens receive zeroed attention mask (`attention_mask`) before `causal_conv1d`, DeltaNet `_update_linear_attn_mask`, and SDPA.
3. **Trainable Parameter Parity Verified:**
   - Total trainable tensors: **248 LoRA modules** (`r=32, alpha=64, dropout=0.05` across `q/k/v/o/in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj/gate/up/down_proj`) plus `score` head = **65,001,472** trainable parameters.
   - Initial weights loaded from frozen S3 checkpoint (`artifacts/batch-b-s3/performance/shared-initialization/`) match S3 `lora_sha256` and `head_sha256` digests identically.
4. **Actual Padding Collator (`padding-shapes.csv`) Verified — Refutes Static-64 Hypothesis (H3):**
   - `DevicePartition.batch(indices, multiple=8)` dynamically pads each micro-batch to `min(ceil(max_len_in_micro / 8) * 8, 64)` rather than always padding to 64.
   - At `micro=1` across `train_43211` (`718,506` valid tokens): `866,752` padded tokens -> **`82.90%` valid token ratio** (`mean_padded_len = 20.059`, vs `16.628` valid). Across the `matched_128` profiling window (`2,150` valid tokens): `2,600` padded tokens (`82.69%` valid ratio).
   - Increasing `micro_batch_size` under unsorted dynamic multiple-of-8 causes each batch to take the maximum length of more samples: on `train_43211`, valid token efficiency shifts from `82.90%` (`micro=1`) -> `67.04%` (`micro=2`) -> `57.47%` (`micro=4`) -> `51.83%` (`micro=8`) -> `48.24%` (`micro=16`, `mean_padded_len = 34.468`), still far better than `static_64` (`25.98%`).
5. **Hidden FP32 Residual Stream Audit (`peft` + `bitsandbytes`):**
   - `peft.utils.other.prepare_model_for_kbit_training` upcasts every non-`Params4bit` parameter (`embed_tokens`, `conv1d`, `rmsnorm`, `A_log`, `dt_bias`, etc.) to `float32`.
   - `Qwen3_5RMSNorm` computes internally in `.float()` and casts `.type_as(x)`.
   - Consequently, even inside `torch.autocast("cuda", dtype=torch.bfloat16)`, residual activations stay in **FP32**, forcing `bnb.nn.Linear4bit.forward` (`bitsandbytes/nn/modules.py`) to record `inp_dtype = float32`, cast input down to BF16 for `matmul_4bit`, and `.to(inp_dtype)` cast output back to FP32 on every single quantized linear layer (plus `.clone()` in PEFT `Linear4bit`). This manifests directly as 64,880+ cast kernels per update in S4-1 tracing.

---

## 3. S4-0 Baseline Replay & Multi-Source Memory Audit

| Metric | S3 WSL2 Baseline (`comparison.json`) | S4-0 WSL2 Replay (`baseline.json`) | Deviation / Check |
|---|---:|---:|---|
| Update window (`warmup=5`, `measured=8`) | 110.47 s (13.81 s/upd) | 116.50 s (14.5624 s/upd) | +5.46% |
| Training throughput | 1.1586 samples/s | 1.0987 samples/s | **-5.17%** (PASS `<=10%` gate) |
| Cold step 1 / warmup mean | — | 14.5069 s / 14.0762 s | Warmup stable |
| Update median / p95 | — | 14.4554 s / 15.5960 s | Low jitter |
| Validation (`48` rows, `bs=1`) | 10.0146 s (4.793 s/s) | 9.8860 s (4.855 s/s) | Matched |
| PyTorch `peak_allocated` | 5.42 GiB (5,550 MiB est.) | 5,545.9 MiB (5.42 GiB) | Exact match |
| PyTorch `peak_reserved` | 6.17 GiB | 6,268.0 MiB (6.12 GiB) | Exact match |
| Device peak (`nvidia-smi` total card) | 11,391 MiB | 10,860 MiB | WSL + WDDM host within expectation |
| Device min free VRAM | 4,671 MiB | 5,202 MiB | PASS `>= 2,048 MiB` gate |
| `nvidia-smi` util / power / SM clock | 36.1% / — / — | 50.0% / 48.6 W / 1,467 MHz | Low power, low clock |
| Initial 48×28 prob max abs vs S3 | 0.0 (ref) | `1.192e-7` (`0` / `1,344` flips) | Bitwise-level init parity |
| Post-13 48×28 prob mean / max vs S3 | ref | `0.001888` / `0.015203` (`4` / `1,344` flips, all within `0.5±0.05`) | Within floating-point tolerance |

> **口径说明：** S3 WSL2 原始数据来自冻结 `reports/batch-b-s3/performance/comparison.json`；S4-0 为同硬件同锁仓在新隔离环境 `~/.cache/emotionlab-s4-perf/venv` 下重跑的实测基线；后续所有相对 S3 倍率以冻结 `1.158648 samples/s` 为基准，并与 `S4_0_baseline`（`1.098717 samples/s`）相互印证。

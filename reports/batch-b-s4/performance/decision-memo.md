# Batch B S4 Decision Memo (`decision-memo.md`)

**Decision Recommendation:** **`GO_FULL`** (Proceed with Category A Winner `A_winner_foreach_evalfast` pending human sign-off; formal 3-seed training has **NOT** started)  
**Target Protocol Artifact:** [`recommended-training.json`](./recommended-training.json) (`status: "awaiting_review_s4_candidate"`)  
**Time Budget Data:** [`time-budget.json`](./time-budget.json)

---

## 1. End-to-End Time Budget Recalculation (Actual S4 Training + Actual S4 Validation)

Using `train = 43,211` rows, `validation = 5,416` rows, and exact observed steady throughput (`S3 WSL2 = 1.1586 train / 4.793 val s/s`; `S4 Winner A_winner_foreach_evalfast = 21.8852 train / 130.9508 val s/s`; `B1 = 21.3321 train / 130.9508 val s/s`), with explicit overhead (`load + initial eval + 15 checkpoint saves`: `0.0103 h` for 1 seed, `0.0183–0.0308 h` for 3 seeds) and scenario multipliers (`optimistic = 0.95×`, `neutral = 1.00×`, `conservative = +15%`):

| Experiment Scope | S3 Neutral (Range) | S4 Category A Winner Train + Val | **S4 Winner Neutral** | **S4 Winner Range (Opt – Cons)** | S4 `B1_nf4_bf16` Neutral (Range) |
|---|---:|---:|---:|---:|---:|
| **1 seed × 5 epochs** | 53.38 h (50.7–61.4 h) | `2.742 h` + `0.057 h` | **2.810 h** | **2.67 – 3.23 h** | 2.881 h (2.74–3.31 h) |
| **3 seeds × 2 epochs** | 64.06 h (60.9–73.7 h) | `3.291 h` + `0.069 h` | **3.378 h** | **3.21 – 3.88 h** | 3.463 h (3.29–3.98 h) |
| **3 seeds × 3 epochs** | 96.08 h (91.3–110.5 h) | `4.936 h` + `0.103 h` | **5.062 h** | **4.81 – 5.82 h** | 5.190 h (4.93–5.97 h) |
| **3 seeds × 5 epochs** | **160.13 h** (152.1–184.1 h) | **`8.227 h` + `0.172 h`** | **8.430 h** | **8.01 – 9.69 h** | 8.643 h (8.21–9.94 h) |

*Note: Budgets do not assume early stopping (`patience=2`) triggers; if patience stops any seed around epoch 3–4, total wall time drops further into `~5–7 hours`.*

---

## 2. Evaluation Against Section 6 Decision Thresholds

| Threshold Criterion | Required Budget | S4 Winner Actual | Pass? |
|---|---|---|---|
| **Single-seed feasibility** (`1 seed × 5 epochs`) | `<= ~24 h`, recovery & numerical checks PASS | **2.81 h** neutral (**3.23 h** conservative), exact `0.0` recovery error | **PASS (7.4× below 24h limit)** |
| **Full 3-seed feasibility** (`3 seeds × 3 / 5 epochs`) | `<= 24–36 h` | **5.06 h** (3 ep) / **8.43 h** neutral, **9.69 h** conservative (5 ep) | **PASS (< 10h overnight run!)** |
| **VRAM Safety Gate** (`>= 2,048 MiB` host-level free) | `>= 2,048 MiB` | **`4,167 MiB`** free (`11,895 MiB` peak card usage out of 16 GiB) | **PASS (`2×` safety margin)** |
| **Scientific Protocol Integrity** | Identical model, NF4, 248 LoRA modules, effective batch 16, BCE, epochs, seeds | **100% preserved** (pure Category A system changes) | **PASS** |

---

## 3. Four-Way Option Analysis & Opportunity Costs

### Recommended Choice: **`GO_FULL`** (Using S4 Category A Config `A_winner_foreach_evalfast`)
- **Configuration Changes vs. S3 (minimal diff in `recommended-training.json`):**
  1. `micro_batch_size`: `1 -> 16` (`accumulation_steps`: `16 -> 1`)
  2. `gradient_checkpointing`: `"non_reentrant" (32 layers) -> "partial_16" (16 alternating layers)`
  3. `adamw_foreach`: `False -> True`
  4. `evaluation_micro_batch_size`: `1 -> 32`, `evaluation_sort_by_length`: `False -> True`
- **Why `GO_FULL` Wins:**
  - Compresses `3 seeds × 5 epochs` from an unworkable **`160.13 h` (~6.7 days)** down to **`8.43 h` neutral (`8.01–9.69 h`)**, cleanly under both the 24h and 36h thresholds for full 3-seed Batch B execution.
  - Preserves exact Batch B scientific comparability: keeps NF4 double-quant, BF16 compute, `Qwen/Qwen3.5-4B-Base` rev `1001bb4d`, all 248 LoRA adapters (`r=32, α=64`), full `score` head, `effective_batch=16`, LR schedule, 5 epochs, and all 3 seeds (`42, 123, 2026`).
  - Zero accuracy-degrading tricks: Step-13 diff vs S3 (`prob_mean=0.00215`, `5/1344` borderline flips) stays within S3's own cross-backend hardware drift (`0.00220`, `3/1344` flips), `248/248` LoRA + head gradients remain active, and independent-PID recovery passed with `0.0` score error.
  - Operational safety: Running `seeds=[42, 123, 2026]` sequentially inside the existing resumable trainer naturally provides seed `42` completion in **`~2.8 hours`** as an inspectable checkpoint without narrowing the protocol scope before launch.

### Why the Other Three Options Are Suboptimal
1. **Why `GO_PILOT` (Single Seed Only) Is Suboptimal:** `GO_PILOT` makes sense when 1 seed takes ~18–24 hours and 3 seeds would cost 50–70 hours. With `3 seeds × 5 epochs` costing only **`8.43 hours` (`8.01–9.69 h`)** — less than half of one single 24h window — stopping after 1 seed leaves multi-seed variance unresolved while saving only ~5.6 GPU-hours. Running `GO_FULL` sequentially (`42 -> 123 -> 2026`) already gets seed 42's result in ~2.8h while completing the full scientific protocol overnight. *(If human governance strictly wants seed-42 validation Macro-F1 inspected before running seeds 123 & 2026, setting `--seeds 42` first costs almost nothing in orchestration overhead thanks to verified exact resume.)*
2. **Why `DEFER` Is Suboptimal:** The `DEFER` trigger threshold (`< 2.7 samples/s`) is exceeded by **8.1×** (`21.885 samples/s`). There is no remaining runtime blockage on WSL2 RTX 4070 Ti SUPER: VRAM headroom is `4,167 MiB`, recovery is verified across processes, and isolated venv lockfiles already exist. Deferring wastes a resolved engineering breakthrough.
3. **Why `DROP_4B` Is Suboptimal (for Batch B S4):** Abandoning 4B was warranted only if 4B could not finish within an acceptable GPU budget without breaking Batch B's scientific baseline. Now that S4 achieves `8.43 h` for 3 seeds × 5 epochs under exact Category A constraints, `DROP_4B` would discard cross-scale 4B vs small-model comparisons prematurely. *(Category C alternatives like 0.8B/2B, LoRA subset, or frozen-backbone linear probes remain valuable **companion baselines** later, not replacements forced by slow Qwen3.5 runtime.)*

---

**Formal Human Review Gate:** `recommended-training.json` has `status: "awaiting_review_s4_candidate"` and `formal_training_started: false`. No 3-seed training has begun, and `test.jsonl` remains completely untouched.

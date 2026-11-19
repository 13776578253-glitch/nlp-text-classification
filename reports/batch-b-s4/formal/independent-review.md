# Independent Pre-launch Review

- Reviewer: independent explore agent (read-only; no shell, no file writes, no training)
- Date: 2026-11-19
- Subject: `reports/batch-b-s4/formal/prelaunch-review.md` (Rev 1) and its underlying evidence
- Verdict: **PASS** (B1 withdrawn; the formal gate is WSL `/proc/meminfo`)

The independent reviewer cross-checked the tracked evidence, not the live machine. SHA-256 recomputation, pytest execution and live `resource_status()` / `/api/snapshot` calls were outside the read-only toolset; those checks were run separately by the main agent and are referenced below. The reviewer explicitly did not treat them as independently recomputed. The original B1 was based on the report's Windows `resource_status()`; after inspecting the WSL branch and `/proc/meminfo`, it is withdrawn.

## Checks performed by the independent reviewer

| Check | Result | Evidence |
| --- | --- | --- |
| v5/v6 acceptance status and flags | PASS | both `status: PASS`, `formal_training_started: false` |
| v5→v6 source hash diff | PASS | exactly one of 13 differs: `batch_b_s4_formal.py` `eb8ae85e…` → `690e417d1936…f28e79`; other 12 identical |
| Report hash table vs v6 acceptance | PASS | all 13 prefixes/suffixes match the v6 map |
| v6 acceptance numeric fields | PASS | `device_min_free_mib` 2821, validation 49.253 s, reload 38.769 s, save 11.492 s, rows 5416, plan 16/2701/11/13505/811, step 4 / replayed 5, 497 states |
| Restore flags | PASS | same cursor, RNG probe, exact scores, max abs error 0.0, exact next update |
| 151,648 score claim | PASS | `reproduced_scores: 151648` = 5416×28 |
| v5/v6 checkpoint `files` | PASS | only `adapter/adapter_config.json` differs; weights, `resume.pt`, predictions, metrics, thresholds identical |
| adapter config difference | PASS | same 12 `target_modules`, different order; all other fields identical |
| checkpoint `parameters` / `updates` / `state` / `cursor` | PASS | identical |
| reference stress / tail / validation / `next_update` | PASS | loss 1.146449, grad 14.015074, tail 1.043926/11.413060, per-label metrics and step-5 replay identical |
| memory delta | PASS | 3948 → 2821 MiB, as reported |
| reports vs artifacts v6 acceptance | PASS | content-identical (341 lines, same pids 362/1989) |
| reports checkpoint copy vs artifact | PASS | content-identical |
| data fingerprint / counts (declared) | PASS | train 43,211 `31b27182…`, validation 5,416 `017da8aa…`, dataset manifest `f9f503dc…`; actual bytes not hashed by reviewer |
| plan consistency | PASS | 16/2701/13505/811 in recipe, acceptance, INPUTS_OK, tests |
| runbook | PASS | three train commands + resume all use `prelaunch-v6`; no v5 left |
| report links | PASS | all v4/v5 evidence files exist |
| monitor write scope | PASS | writes only `MONITOR_ROOT/seed-*/digest.*`; run dirs are read-only |
| monitor torch import | PASS | stdlib only; GPU via `nvidia-smi` |
| monitor digest | PASS | `S4 monitor | Idle | no training run found`; `status NO_RUN`, seed 42, elapsed 0 |
| monitor notable-event filter | PASS | TRAIN/VALIDATE/VALIDATION_ROWS/EPOCH/CHECKPOINT excluded |
| memory gate code | PASS | `START_FREE=8 GiB`, `PAUSE_FREE=6 GiB`, `min(physical,commit)<8` raises before model load |
| repo state | PASS | no `runs/`; only `monitor/`, `preflight/`, `watchdog-fixtures/` |
| watchdog thresholds | PASS | 2560 warning, 1024 critical, 15 s, 10240 retry; fixture lock released |
| branch / HEAD | PARTIAL | HEAD `5165ba4`; parent `a55c761` not confirmable with read-only tools |

## Not independently recomputed

- SHA-256 of the 13 bound sources, six pinned configs, reference/checkpoint copies and data files.
- `pytest` 68, `ruff`, `git diff --check`.
- Live host memory (`resource_status`), GPU free, disk free, and `/api/snapshot`.

Main-agent evidence for these: the hash audit output, the WSL `_binding`/`checked_inputs` rerun (`BINDING_EQUAL True`, `INPUTS_OK 43,211/5,416`, encoded shapes `(43211,28)/(5416,28)`), 68 pytest pass, Ruff pass, and the monitor API header. The reviewer found no contradiction between those outputs and the files.

## Blockers

None for the formal entry. The original B1 (Windows host 7.36 GiB) was withdrawn: `training/resources.py` runs inside WSL (`sys.platform == "linux"`) and `memory_available()` reads `/proc/meminfo`; Ubuntu reports 22.4 GiB available, so `protect()` passes the 8 GiB gate. The Windows host `resource_status()` 9.14 GiB is a desktop headroom note, not the gate. `independent-review.md` now exists.

## Discrepancies found and resolution

1. Test count: `reports/batch-b-s4/formal/report.md` said 67; the audit report said 68. Static inventory supports 68. **Fixed in Rev 2** (`report.md` now 68).
2. §2.4 overstated "唯一数值差异是显存最低余量"; validation/reload/save times, samples/s, pids and memory readings also differ. **Fixed in Rev 2**, scoped to training payload and listing runtime differences.
3. Live readings (7.36 GiB, 13,839 MiB VRAM, disk, no worker) were presented without a tracked artifact. **Fixed in Rev 2**, marked live at review time and requiring re-check before launch.
4. "5 次更新" mixed four committed updates with the replayed fifth. **Fixed in Rev 2** as "4 次提交更新 + 第 5 次重放".

## Conclusion

No code, data, or evidence mismatch was found beyond the four wording issues, all corrected. The previous CONDITIONAL PASS was based only on the Windows B1, which was a platform misread; the actual WSL gate passes at 22.4 GiB. Final verdict: **PASS**. The Windows host is at 9.14 GiB, with 7.1 GB held by Docker Desktop's WSL distro; stopping Docker is optional for host headroom. Training is not started.

## Follow-up (Rev 2)

A second read-only pass confirmed **CONDITIONAL PASS** and all four fixes:

- Test count 68 in both reports: PASS.
- §2.4 scoped to training payload with runtime differences listed: PASS; every value matches the v5/v6 acceptances.
- Live readings labeled as live-at-review-time: PASS.
- 4 committed updates + fifth replay in both reports: PASS; committed step remains 4.
- `independent-review.md` exists and is linked from §9: PASS.
- No new contradiction in source hashes, acceptance flags, runbook, monitor scope, checkpoint files or reference signals: PASS.

Two non-blocking wording nits were raised and fixed in Rev 3: the checklist row now says "training payload equivalence (runtime readings differ, see §2.4)", and the "~4.3 GiB desktop GPU" estimate is labeled as derived rather than archived.

Remaining blocker: none for the formal entry. The earlier 7.36 GiB Windows reading is a headroom note, not the WSL gate; the WSL gate is 22.4 GiB available. Training is not started.

## Post-review memory-gate correction

The original B1 compared a Windows host reading with a gate that runs in WSL. Fresh inspection:

```text
Ubuntu free -m:    available 22,420 MB (8 GiB required) -> pass
Windows host:      9.14 GiB free (headroom note, not the gate)
Docker distro:     8.18 GB cached
vmmemWSL:          7.1 GB working set
```

Ubuntu was terminated (`wsl --terminate Ubuntu-22.04`); the host free stayed at 9.14 GiB, proving the 7.1 GB is Docker Desktop's WSL distro, not Ubuntu. Docker Desktop and `com.docker.build` are still running; stopping them is optional for Windows headroom. The formal gate passes; final verdict **PASS**. Training is not started.

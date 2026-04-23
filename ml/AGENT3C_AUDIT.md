# Agent 3C Audit Report

**Auditor:** Agent 3C-Audit (independent)
**Date:** 2026-04-18
**Approach:** All 8 checks run from scratch against live code and data. Prior AGENT3C_AUDIT.md results were not trusted and are fully replaced below.

---

## Check Results

| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | 5 folds, no test overlap | PASS | fold_1: 2024-01-01–2024-06-30, fold_2: 2024-07-01–2024-12-31, fold_3: 2025-01-01–2025-06-30, fold_4: 2025-07-01–2025-12-31, fold_5: 2026-01-01–2026-03-18. No pair overlaps. |
| 2 | Purge gap >= forward horizon | PASS | Last train bar: 2023-06-30 14:35 ET, first val bar: 2023-07-04 11:15 ET. Gap = 71 bars >= 5 (forward horizon). |
| 3 | Embargo size correct | PASS | 1min=330, 3min=110, 5min=66. All match expected one-session values (RTH 09:30–15:00 = 330 min). |
| 4 | Training rows all-bars | PASS | Fold 1 train rows after purge: 38,906 (> 30,000 threshold). No signal-bar gating active. |
| 5 | session_pivot in strategy maps | PASS | STRATEGY_TIMEFRAME_MAP["session_pivot"]="5min", STRATEGY_SIGNAL_COLUMN_MAP["session_pivot"]="session_pivot_signal". |
| 6 | MODEL_GROUPS covers all 9 signals | PASS | 4 groups: model_1 [ifvg_signal, connors_signal], model_2 [ifvg_open_signal, ttm_signal], model_3 [orb_vol_signal, session_pivot_signal, session_pivot_break_signal], model_4 [orb_ib_signal, orb_wick_signal]. No duplicates. |
| 7 | No signal-bar filter in code | PASS | No active signal-bar filter patterns found in dataset_builder.py or train.py. |
| 8 | Full test suite | PASS | 117 passed, 3 skipped, 0 failed (245s). |

---

## Verdict

**3C APPROVED — proceed to 3D**

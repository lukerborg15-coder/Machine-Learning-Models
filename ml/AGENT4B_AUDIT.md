# Agent 4B Audit Report

## Critical Check — Training Row Count
- Fold 1 5min training rows: 38,911
- Expected full-session bars: 45,209
- Expected minimum (80% of full session bars): 36,167
- Result: PASS

Output:
```text
Fold 1 training rows (5min, all bars): 38911
? PASS: Training rows 38911 > threshold 36167
```

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| Critical | Training row count all-bars | PASS | 38,911 rows > 36,167 threshold |
| 1 | No signal-bar filter in code | PASS | No active signal-bar filters found in `dataset_builder.py` or `train.py` |
| 2 | NoTrade label dominates (>90%) | PASS | model_1 97.3%, model_2 98.7%, model_3 97.4%, model_4 98.9%; all long/short rates > 0.1% |
| 3 | Conflict bars labeled NoTrade | PASS | Conflict bar label was 2 |
| 4 | Four model groups, all 9 strategies | PASS | 4 groups; all 9 signal columns covered; no duplicates |
| 5 | Contract limit and position sizing | PASS | max_contracts=50; 10pt/0.60=10, 10pt/0.80=25, 5pt/0.75=40, 20pt/0.75=9 |
| 6 | 5 folds, no overlap | PASS | 5 folds; no test overlap; fold_5 test ends 2026-03-18 |
| 7 | Full test suite | PASS | 116 passed, 3 skipped, 0 failed; all 12 Agent 4B tests passed |

## Raw Check Outputs

### Check 1 — No Signal-Bar Filter in Code
```text
? No active signal-bar filters found in dataset_builder or train
```

### Check 2 — NoTrade Label Dominates
Note: `assign_labels` is exposed from `ml.train` in this codebase, so that import path was used for the provided check logic.

```text
model_1: NoTrade=97.3%, Long=1.46%, Short=1.26%
  ? model_1 label distribution OK
model_2: NoTrade=98.7%, Long=0.75%, Short=0.58%
  ? model_2 label distribution OK
model_3: NoTrade=97.4%, Long=1.38%, Short=1.18%
  ? model_3 label distribution OK
model_4: NoTrade=98.9%, Long=0.59%, Short=0.49%
  ? model_4 label distribution OK
```

### Check 3 — Conflict Bars Labeled NoTrade
```text
Conflict bar label: 2
? PASS: Conflict bar correctly labeled NoTrade
```

### Check 4 — Four Model Groups Defined
```text
? PASS: 4 model groups, all 9 strategies covered, no duplicates
  model_1: ['ifvg_signal', 'connors_signal']
  model_2: ['ifvg_open_signal', 'ttm_signal']
  model_3: ['orb_vol_signal', 'session_pivot_signal', 'session_pivot_break_signal']
  model_4: ['orb_ib_signal', 'orb_wick_signal']
```

### Check 5 — Contract Limit and Position Sizing
```text
? PASS: max_contracts=50, confidence scaling correct, stop scaling correct
  10pt stop, conf=0.60: 10 contracts
  10pt stop, conf=0.80: 25 contracts
  5pt stop,  conf=0.75: 40 contracts
  20pt stop, conf=0.75: 9 contracts
```

### Check 6 — 5 Folds Configured Correctly
```text
? PASS: 5 folds, no test overlap, chronological order
  fold_1: train?2023-06-30, val?2023-12-31, test?2024-06-30
  fold_2: train?2023-12-31, val?2024-06-30, test?2024-12-31
  fold_3: train?2024-06-30, val?2024-12-31, test?2025-06-30
  fold_4: train?2024-12-31, val?2025-06-30, test?2025-12-31
  fold_5: train?2025-06-30, val?2025-12-31, test?2026-03-18
```

### Check 7 — Full Test Suite
Command:
```text
python -m pytest ml/tests -v --tb=short
```

Clean rerun result:
```text
============ 116 passed, 3 skipped, 1 warning in 224.49s (0:03:44) ============
```

The first run printed the same complete pytest summary, but the shell wrapper timed out after the summary and returned 124. It was rerun with a longer timeout and exited 0.

## Verdict
**4B APPROVED — proceed to 3C**

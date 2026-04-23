# Agent 4B-Audit — Architecture Review

**Role:** Independent auditor for Agent 4B's work
**Blocks:** Agent 3C cannot start until this audit passes
**Approach:** Read the spec cold, inspect the code independently. Do not trust Agent 4B's reported counts — verify them by reading the code directly.

---

## Execution Requirement — RUN ALL CHECKS

> **Run every check by executing the exact Python code provided. Do NOT skip any check. Do NOT substitute code inspection for execution.** Record the actual printed output for each check. Mark the audit report with exact pass/fail counts from real output.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\MASTER_CHANGE_PLAN.md` — especially the all-bars guarantee
2. `C:\Users\Luker\strategyLabbrain\Agents\Agent 4B - Grouped Model Architecture.md` — the spec
3. `C:\Users\Luker\strategyLabbrain\ml\train.py` — read in full
4. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — read in full
5. `C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py` — read in full

Read the code before reading Agent 4B's status file.

---

## The Most Important Check

Before running any other check, run this. If it fails, BLOCK immediately:

```python
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")

# Count bars in fold_1 training window
fold1_train = df["2021-03-19":"2023-06-30"]
n_rows = len(fold1_train)
print(f"Fold 1 training rows (5min, all bars): {n_rows}")

# Expected: ~252 trading days/year × 2.3 years × 78 bars/day ≈ 45,396
# Minimum acceptable: 80% of expected = ~36,000
# If you see 750-3000: signal-bar gating is still active → BLOCK
expected = 252 * 2.3 * 78
threshold = expected * 0.80
assert n_rows > threshold, (
    f"CRITICAL FAIL: Training rows ({n_rows}) far below expected ({expected:.0f}). "
    f"Signal-bar gating is still active. This is the root cause of zero deployment candidates. "
    f"Agent 4B must fix before any further work."
)
print(f"✓ PASS: Training rows {n_rows} > threshold {threshold:.0f}")
```

---

## Audit Check 1 — No Signal-Bar Filter in Code

Search the codebase for any remaining signal-bar gating:

```python
import subprocess, sys

files_to_check = [
    r"C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py",
    r"C:\Users\Luker\strategyLabbrain\ml\train.py",
]

filter_patterns = [
    r"signal.*!=.*0",
    r"!= 0.*signal",
    r"\[df\[.*signal.*\] != 0\]",
    r"signal_col.*filter",
]

found_any = False
for filepath in files_to_check:
    with open(filepath) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        for pattern in filter_patterns:
            import re
            if re.search(pattern, line) and not line.strip().startswith("#"):
                print(f"POTENTIAL FILTER at {filepath}:{i}: {line.rstrip()}")
                found_any = True

if found_any:
    print("\n⚠️  Found potential signal filtering. Review each line above.")
    print("    If any of these are active data filters (not comments/label logic), BLOCK.")
else:
    print("✓ No active signal-bar filters found in dataset_builder or train")
```

Record: PASS or BLOCK with exact line numbers of any remaining filters.

---

## Audit Check 2 — NoTrade Label Dominates

```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.dataset_builder import assign_labels  # or however the function is exposed

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
fold1_train = df["2021-03-19":"2023-06-30"].copy()

# Test each model group
model_groups = [
    {"name": "model_1", "signal_cols": ["ifvg_signal", "connors_signal"]},
    {"name": "model_2", "signal_cols": ["ifvg_open_signal", "ttm_signal"]},
    {"name": "model_3", "signal_cols": ["orb_vol_signal", "session_pivot_signal", "session_pivot_break_signal"]},
    {"name": "model_4", "signal_cols": ["orb_ib_signal", "orb_wick_signal"]},
]

for mg in model_groups:
    labeled = assign_labels(fold1_train.copy(), mg["signal_cols"])
    counts = labeled["label"].value_counts(normalize=True)
    notrade_pct = counts.get(2, 0)
    long_pct    = counts.get(0, 0)
    short_pct   = counts.get(1, 0)
    print(f"{mg['name']}: NoTrade={notrade_pct:.1%}, Long={long_pct:.2%}, Short={short_pct:.2%}")

    assert notrade_pct > 0.90, (
        f"FAIL {mg['name']}: NoTrade={notrade_pct:.1%} should be >90%. "
        "Signal-bar gating may still be active."
    )
    assert long_pct > 0.001, f"FAIL {mg['name']}: No Long labels"
    assert short_pct > 0.001, f"FAIL {mg['name']}: No Short labels"
    print(f"  ✓ {mg['name']} label distribution OK")
```

Record: PASS or FAIL per model group.

---

## Audit Check 3 — Conflict Bars Labeled NoTrade

```python
import pandas as pd, numpy as np, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.dataset_builder import assign_labels

# Build synthetic DataFrame with a bar where ifvg fires long AND connors fires short
import pandas as pd
dates = pd.date_range("2024-01-02 10:00", periods=5, freq="5min", tz="America/New_York")
df = pd.DataFrame({
    "open": [100.0]*5, "high": [101.0]*5, "low": [99.0]*5, "close": [100.5]*5,
    "volume": [1000]*5, "atr_14": [2.0]*5,
    "ifvg_signal":    [1, 0, 0, 0, 0],      # long on bar 0
    "connors_signal": [-1, 0, 0, 0, 0],     # short on bar 0 — CONFLICT (model_1)
    "future_return":  [0.005, 0, 0, 0, 0],  # positive return (would be a long win if no conflict)
}, index=dates)

labeled = assign_labels(df, signal_cols=["ifvg_signal", "connors_signal"])
conflict_label = labeled["label"].iloc[0]
print(f"Conflict bar label: {conflict_label}")
assert conflict_label == 2, (
    f"FAIL: Conflict bar (long+short same bar) should be NoTrade (2), got {conflict_label}. "
    "Conflicting direction signals must produce NoTrade label."
)
print("✓ PASS: Conflict bar correctly labeled NoTrade")
```

Record: PASS or FAIL.

---

## Audit Check 4 — Four Model Groups Defined

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import MODEL_GROUPS  # adjust import path as needed

assert len(MODEL_GROUPS) == 4, f"Expected 4 model groups, found {len(MODEL_GROUPS)}"

expected_names = {"model_1", "model_2", "model_3", "model_4"}
actual_names   = {mg["model_name"] for mg in MODEL_GROUPS}
assert actual_names == expected_names, f"Model names mismatch: {actual_names}"

# Verify all 9 strategies are covered
all_signal_cols = []
for mg in MODEL_GROUPS:
    all_signal_cols.extend(mg["signal_cols"])

required = [
    "ifvg_signal", "ifvg_open_signal", "connors_signal", "ttm_signal",
    "orb_ib_signal", "orb_wick_signal", "orb_vol_signal",
    "session_pivot_signal", "session_pivot_break_signal",
]
for col in required:
    assert col in all_signal_cols, f"Strategy signal not covered by any model: {col}"

# Verify no signal col appears in two models
from collections import Counter
counts = Counter(all_signal_cols)
duplicates = {k: v for k, v in counts.items() if v > 1}
assert not duplicates, f"Signal cols assigned to multiple models: {duplicates}"

print("✓ PASS: 4 model groups, all 9 strategies covered, no duplicates")
for mg in MODEL_GROUPS:
    print(f"  {mg['model_name']}: {mg['signal_cols']}")
```

Record: PASS or FAIL.

---

## Audit Check 5 — Contract Limit and Position Sizing

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.topstep_risk import TopStepRiskManager

rm = TopStepRiskManager()

# Verify ceiling
assert rm.max_contracts == 50, f"max_contracts={rm.max_contracts}, expected 50"

# Verify position_size caps at 50
size_at_high_conf = rm.position_size(stop_pts=1.0, confidence=0.99)
assert size_at_high_conf == 50, f"Expected 50 at max confidence, got {size_at_high_conf}"

# Verify confidence scaling
size_low  = rm.position_size(stop_pts=10, confidence=0.60)
size_high = rm.position_size(stop_pts=10, confidence=0.80)
assert size_low < size_high, (
    f"FAIL: Lower confidence should give fewer contracts. "
    f"Got low={size_low}, high={size_high}"
)

# Verify stop scaling (tighter stop = more contracts at same risk)
size_tight = rm.position_size(stop_pts=5,  confidence=0.75)
size_wide  = rm.position_size(stop_pts=20, confidence=0.75)
assert size_tight > size_wide, (
    f"FAIL: Tighter stop should allow more contracts. "
    f"Got tight={size_tight}, wide={size_wide}"
)

# Verify zero/negative stop raises
import pytest
try:
    rm.position_size(stop_pts=0, confidence=0.75)
    assert False, "Should have raised ValueError for stop_pts=0"
except (ValueError, ZeroDivisionError):
    pass  # expected

print("✓ PASS: max_contracts=50, confidence scaling correct, stop scaling correct")
print(f"  10pt stop, conf=0.60: {size_low} contracts")
print(f"  10pt stop, conf=0.80: {size_high} contracts")
print(f"  5pt stop,  conf=0.75: {size_tight} contracts")
print(f"  20pt stop, conf=0.75: {size_wide} contracts")
```

Record: PASS or FAIL.

---

## Audit Check 6 — 5 Folds Configured Correctly

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import WALK_FORWARD_FOLDS

assert len(WALK_FORWARD_FOLDS) == 5, f"Expected 5 folds, got {len(WALK_FORWARD_FOLDS)}"

# Verify TEST windows do not overlap each other.
# NOTE: In rolling walk-forward, fold N+1 val == fold N test is CORRECT BY DESIGN.
# Do NOT assert f1.test_end < f2.val_start — that will always fail intentionally.
for i in range(len(WALK_FORWARD_FOLDS) - 1):
    f1 = WALK_FORWARD_FOLDS[i]
    f2 = WALK_FORWARD_FOLDS[i + 1]
    assert f1.test_end < f2.test_start, (
        f"Fold {i+1} and {i+2} test windows overlap: {f1.test_end} >= {f2.test_start}"
    )

# Verify fold_5 covers the most recent data
last_fold = WALK_FORWARD_FOLDS[-1]
assert "2026" in last_fold.test_end, f"Last fold test should end in 2026, got {last_fold.test_end}"

print("✓ PASS: 5 folds, no test overlap, chronological order")
for fold in WALK_FORWARD_FOLDS:
    print(f"  {fold.name}: train→{fold.train_end[:10]}, val→{fold.val_end[:10]}, test→{fold.test_end[:10]}")
```

Record: PASS or FAIL.

---

## Audit Check 7 — Full Test Suite

```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```

All previously passing tests must still pass. All 12 new Agent 4B tests must pass. Record exact count.

---

## Audit Decision

Write `ml/AGENT4B_AUDIT.md`:

```markdown
# Agent 4B Audit Report

## Critical Check — Training Row Count
- Fold 1 5min training rows: [N]
- Expected minimum (80% of full session bars): [N]
- Result: PASS / BLOCK

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| Critical | Training row count all-bars | PASS/BLOCK | [exact count] |
| 1 | No signal-bar filter in code | PASS/FAIL | [line numbers if found] |
| 2 | NoTrade label dominates (>90%) | PASS/FAIL | [per-model breakdown] |
| 3 | Conflict bars labeled NoTrade | PASS/FAIL | |
| 4 | Four model groups, all 9 strategies | PASS/FAIL | |
| 5 | Contract limit and position sizing | PASS/FAIL | [values] |
| 6 | 5 folds, no overlap | PASS/FAIL | |
| 7 | Full test suite | PASS/FAIL | [N passed, N failed] |

## Verdict
**4B APPROVED — proceed to 3C**
OR
**4B BLOCKED — [list every failing check]**

Do NOT read AGENT4B_STATUS.md until after you have written your verdict.

Do not write APPROVED unless Critical check passes AND all other checks pass.
```

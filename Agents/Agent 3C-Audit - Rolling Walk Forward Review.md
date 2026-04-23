# Agent 3C-Audit — Rolling Walk-Forward Review

**Role:** Independent auditor for Agent 3C's work
**Blocks:** Agent 3D cannot start until this audit passes
**Approach:** Run every check with the exact code below. Do not trust Agent 3C's reported results — verify everything yourself from scratch.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Agents\Agent 3C - Rolling Walk Forward and Purging.md`
2. `C:\Users\Luker\strategyLabbrain\ml\train.py` — read in full
3. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — focus on purge/embargo
4. Do NOT read `AGENT3C_STATUS.md` until after you have run all checks and written your verdict

---

## Absolute Paths

```
Project root: C:\Users\Luker\strategyLabbrain
train.py:     C:\Users\Luker\strategyLabbrain\ml\train.py
tests:        C:\Users\Luker\strategyLabbrain\ml\tests\
```

---

## Check 1 — Five Folds, No Test Overlap

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import WALK_FORWARD_FOLDS
import pandas as pd

assert len(WALK_FORWARD_FOLDS) == 5, f"Expected 5 folds, got {len(WALK_FORWARD_FOLDS)}"

test_windows = []
for fold in WALK_FORWARD_FOLDS:
    ts = pd.Timestamp(fold.test_start, tz="America/New_York")
    te = pd.Timestamp(fold.test_end, tz="America/New_York")
    test_windows.append((fold.name, ts, te))
    print(f"  {fold.name}: test {ts.date()} to {te.date()}")

for i in range(len(test_windows)):
    for j in range(i + 1, len(test_windows)):
        name_i, s_i, e_i = test_windows[i]
        name_j, s_j, e_j = test_windows[j]
        overlap = max(s_i, s_j) <= min(e_i, e_j)
        assert not overlap, (
            f"FAIL: {name_i} and {name_j} test windows overlap. "
            f"{name_i}: {s_i.date()}-{e_i.date()}, {name_j}: {s_j.date()}-{e_j.date()}"
        )

print("PASS: 5 folds, no test window overlap")
```

---

## Check 2 — Purge Gap Removes Label-Leaking Rows

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import WALK_FORWARD_FOLDS
from ml.dataset_builder import apply_purge_embargo, DEFAULT_FORWARD_HORIZON_BARS

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
fold = WALK_FORWARD_FOLDS[0]

splits = apply_purge_embargo(df, fold, forward_horizon_bars=DEFAULT_FORWARD_HORIZON_BARS)
train = splits["train"]
val   = splits["val"]

last_train_bar = train.index[-1]
first_val_bar  = val.index[0]
gap_bars = len(df.loc[last_train_bar:first_val_bar]) - 2

print(f"Last train bar:       {last_train_bar}")
print(f"First val bar:        {first_val_bar}")
print(f"Gap bars:             {gap_bars}")
print(f"Forward horizon bars: {DEFAULT_FORWARD_HORIZON_BARS}")

assert gap_bars >= DEFAULT_FORWARD_HORIZON_BARS, (
    f"FAIL: Purge gap ({gap_bars}) < forward horizon ({DEFAULT_FORWARD_HORIZON_BARS}). "
    "Label leakage present."
)
print("PASS: Purge gap removes label-leaking train rows")
```

---

## Check 3 — Embargo Size Matches Timeframe

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.dataset_builder import embargo_bars_for_timeframe

# RTH session 09:30-15:00 = 330 min. One full session per timeframe:
EXPECTED = {"1min": 330, "3min": 110, "5min": 66}

for tf, expected in EXPECTED.items():
    actual = embargo_bars_for_timeframe(tf)
    assert actual == expected, f"FAIL: {tf} embargo={actual}, expected {expected}"
    print(f"  PASS {tf}: embargo={actual} bars")

print("PASS: Embargo sizes correct for all timeframes")
```

---

## Check 4 — Training Row Count Is All-Bars (No Signal Gating)

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import WALK_FORWARD_FOLDS
from ml.dataset_builder import apply_purge_embargo, DEFAULT_FORWARD_HORIZON_BARS

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
fold = WALK_FORWARD_FOLDS[0]
splits = apply_purge_embargo(df, fold, forward_horizon_bars=DEFAULT_FORWARD_HORIZON_BARS)
train = splits["train"]

print(f"Fold 1 train rows after purge: {len(train)}")

# 5min bars: ~252 days/yr * 2.3 yrs * 66 RTH bars/day ~ 38,000 minimum
assert len(train) > 30_000, (
    f"FAIL: Only {len(train)} training rows. Signal-bar gating may be active."
)
print("PASS: Training rows confirm all-bars (no signal-bar gating)")
```

---

## Check 5 — session_pivot in Strategy Maps

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import STRATEGY_TIMEFRAME_MAP, STRATEGY_SIGNAL_COLUMN_MAP

assert "session_pivot" in STRATEGY_TIMEFRAME_MAP, (
    "FAIL: session_pivot missing from STRATEGY_TIMEFRAME_MAP"
)
assert STRATEGY_TIMEFRAME_MAP["session_pivot"] == "5min", (
    f"FAIL: session_pivot timeframe should be 5min, got {STRATEGY_TIMEFRAME_MAP['session_pivot']}"
)
assert "session_pivot" in STRATEGY_SIGNAL_COLUMN_MAP, (
    "FAIL: session_pivot missing from STRATEGY_SIGNAL_COLUMN_MAP"
)
assert STRATEGY_SIGNAL_COLUMN_MAP["session_pivot"] == "session_pivot_signal", (
    f"FAIL: wrong signal column for session_pivot: {STRATEGY_SIGNAL_COLUMN_MAP['session_pivot']}"
)
print("PASS: session_pivot correctly in both strategy maps")
```

---

## Check 6 — MODEL_GROUPS Covers All 9 Signals, No Duplicates

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.train import MODEL_GROUPS
from collections import Counter

assert len(MODEL_GROUPS) == 4, f"FAIL: Expected 4 model groups, got {len(MODEL_GROUPS)}"

all_signal_cols = []
for mg in MODEL_GROUPS:
    all_signal_cols.extend(mg["signal_cols"])

required = [
    "ifvg_signal", "ifvg_open_signal", "connors_signal", "ttm_signal",
    "orb_ib_signal", "orb_wick_signal", "orb_vol_signal",
    "session_pivot_signal", "session_pivot_break_signal",
]
for col in required:
    assert col in all_signal_cols, f"FAIL: {col} not covered by any model group"

counts = Counter(all_signal_cols)
dupes = {k: v for k, v in counts.items() if v > 1}
assert not dupes, f"FAIL: Signal cols in multiple groups: {dupes}"

print(f"PASS: {len(MODEL_GROUPS)} model groups, all 9 signals covered, no duplicates")
for mg in MODEL_GROUPS:
    print(f"  {mg['model_name']}: {mg['signal_cols']}")
```

---

## Check 7 — No Active Signal-Bar Filter in Code

```python
import re, sys
from pathlib import Path

root = Path(r"C:\Users\Luker\strategyLabbrain")
files_to_check = [
    root / "ml" / "dataset_builder.py",
    root / "ml" / "train.py",
]

filter_patterns = [
    r"\[df\[.*signal.*\]\s*!=\s*0\]",
    r"\.loc\[.*signal.*!=\s*0",
    r"signal.*!=.*0.*filter",
]

found_any = False
for filepath in files_to_check:
    lines = filepath.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in filter_patterns:
            if re.search(pattern, line):
                print(f"POTENTIAL FILTER at {filepath.name}:{i}: {line.rstrip()}")
                found_any = True

if found_any:
    print("FAIL: Active signal-bar filter found. Review lines above.")
else:
    print("PASS: No active signal-bar filters found")
```

---

## Check 8 — Full Test Suite

```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```

Record exact pass/fail counts.

---

## Audit Decision

Write `ml/AGENT3C_AUDIT.md`:

```markdown
# Agent 3C Audit Report

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | 5 folds, no test overlap | PASS/FAIL | [fold dates] |
| 2 | Purge gap >= forward horizon | PASS/FAIL | [gap bars] |
| 3 | Embargo size correct | PASS/FAIL | [values] |
| 4 | Training rows all-bars | PASS/FAIL | [row count] |
| 5 | session_pivot in strategy maps | PASS/FAIL | |
| 6 | MODEL_GROUPS covers all 9 signals | PASS/FAIL | |
| 7 | No signal-bar filter in code | PASS/FAIL | |
| 8 | Full test suite | PASS/FAIL | [N passed] |

## Verdict
**3C APPROVED — proceed to 3D**
OR
**3C BLOCKED — [exact reasons]**
```

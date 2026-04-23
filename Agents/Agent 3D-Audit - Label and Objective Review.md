# Agent 3D-Audit — Label and Objective Review

**Role:** Independent auditor for Agent 3D's work
**Blocks:** Agent 3E cannot start until this audit passes
**Approach:** The triple-barrier label generator has the highest error-surface in the pipeline. Run every check. Do not trust Agent 3D's reported results.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Agents\Agent 3D - Triple Barrier Labels and Meta Labeling.md`
2. `C:\Users\Luker\strategyLabbrain\ml\labels.py` — read in full
3. `C:\Users\Luker\strategyLabbrain\ml\train.py` — focus on `_load_strategy_frame()`
4. Do NOT read `AGENT3D_STATUS.md` until after you have run all checks and written your verdict

---

## Absolute Paths

```
Project root: C:\Users\Luker\strategyLabbrain
labels.py:    C:\Users\Luker\strategyLabbrain\ml\labels.py
train.py:     C:\Users\Luker\strategyLabbrain\ml\train.py
```

---

## Check 1 — Stop Hit First Labels Loss

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label

dates = pd.date_range("2024-01-02 10:00", periods=15, freq="5min", tz="America/New_York")
close_prices = [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5,
                96.0, 95.5, 95.0, 94.5, 94.0, 93.5, 93.0]
df = pd.DataFrame({
    "open":  [p + 0.1 for p in close_prices],
    "high":  [p + 0.3 for p in close_prices],
    "low":   [p - 0.3 for p in close_prices],
    "close": close_prices,
    "atr_14": [2.0] * 15,
}, index=dates)

# Long at bar 0. Entry=100, stop=100-1.5*2=97.0, target=100+1.0*1.5*2=103.0
# Price falls to 97.0 at bar 6 — stop hit first
signal = pd.Series([1] + [0]*14, index=dates)
result = triple_barrier_label(df, signal, df["atr_14"])
label = result["label"].iloc[0]
print(f"Stop-hit label: {label} (expected 0)")
assert label == 0, f"FAIL: Stop hit should give label=0 (loss), got {label}"
print("PASS: Stop hit first correctly labels loss (0)")
```

---

## Check 2 — Target Hit First Labels Win

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label

dates = pd.date_range("2024-01-02 10:00", periods=15, freq="5min", tz="America/New_York")
close_prices = [100.0, 101.0, 102.0, 103.5, 104.0, 105.0, 106.0, 107.0,
                108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]
df = pd.DataFrame({
    "open":  [p - 0.1 for p in close_prices],
    "high":  [p + 0.5 for p in close_prices],
    "low":   [p - 0.5 for p in close_prices],
    "close": close_prices,
    "atr_14": [2.0] * 15,
}, index=dates)

# Long at bar 0. Entry=100, stop=97.0, target=103.0
# Price reaches 103.5 at bar 3 — target hit first
signal = pd.Series([1] + [0]*14, index=dates)
result = triple_barrier_label(df, signal, df["atr_14"])
label = result["label"].iloc[0]
print(f"Target-hit label: {label} (expected 1)")
assert label == 1, f"FAIL: Target hit should give label=1 (win), got {label}"
print("PASS: Target hit first correctly labels win (1)")
```

---

## Check 3 — Short Signal Symmetry

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label

dates = pd.date_range("2024-01-02 10:00", periods=15, freq="5min", tz="America/New_York")
close_prices = [100.0, 99.0, 98.0, 97.0, 96.5, 96.0, 95.5, 95.0,
                94.5, 94.0, 93.5, 93.0, 92.5, 92.0, 91.5]
df = pd.DataFrame({
    "open":  [p + 0.1 for p in close_prices],
    "high":  [p + 0.3 for p in close_prices],
    "low":   [p - 0.3 for p in close_prices],
    "close": close_prices,
    "atr_14": [2.0] * 15,
}, index=dates)

# Short at bar 0. Entry=100, stop=103.0, target=97.0
# Price falls to 97.0 at bar 3 — target hit first → win
signal = pd.Series([-1] + [0]*14, index=dates)
result = triple_barrier_label(df, signal, df["atr_14"])
label = result["label"].iloc[0]
print(f"Short target-hit label: {label} (expected 1)")
assert label == 1, f"FAIL: Short target hit should give label=1 (win), got {label}"
print("PASS: Short signal target hit correctly labels win (1)")
```

---

## Check 4 — Session Boundary Exit at 15:00

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label

# Signal at 14:45. With max_bars=60 that would go well past 15:00.
# Exit must be forced at 15:00.
dates = pd.date_range("2024-01-02 14:45", periods=10, freq="5min", tz="America/New_York")
close_prices = [100.0] * 10  # flat — no barrier hit, only time/session exit
df = pd.DataFrame({
    "open":  close_prices,
    "high":  [p + 0.1 for p in close_prices],
    "low":   [p - 0.1 for p in close_prices],
    "close": close_prices,
    "atr_14": [2.0] * 10,
}, index=dates)

signal = pd.Series([1] + [0]*9, index=dates)
result = triple_barrier_label(df, signal, df["atr_14"], max_bars=60)
exit_time = result["exit_time"].iloc[0]
print(f"Exit time: {exit_time}")

if pd.notna(exit_time):
    exit_ts = pd.Timestamp(exit_time)
    if exit_ts.tzinfo is None:
        exit_ts = exit_ts.tz_localize("America/New_York")
    assert exit_ts.hour <= 15, (
        f"FAIL: Exit time {exit_ts} is after 15:00. Session boundary not enforced."
    )
    print("PASS: Exit correctly capped at session boundary (<=15:00)")
else:
    print("PASS: No exit recorded (signal at end of session)")
```

---

## Check 5 — Transaction Cost Makes Flat Path a Loss

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label

# Flat price — neither stop nor target hit within max_bars.
# After transaction cost the PnL must be negative → label=0.
dates = pd.date_range("2024-01-02 10:00", periods=10, freq="5min", tz="America/New_York")
close_prices = [100.0] * 10
df = pd.DataFrame({
    "open":  close_prices,
    "high":  [p + 0.01 for p in close_prices],
    "low":   [p - 0.01 for p in close_prices],
    "close": close_prices,
    "atr_14": [2.0] * 10,
}, index=dates)

signal = pd.Series([1] + [0]*9, index=dates)
result = triple_barrier_label(df, signal, df["atr_14"],
                              max_bars=5, transaction_cost_pts=0.07)
label = result["label"].iloc[0]
r_mult = result["r_multiple"].iloc[0]
print(f"Flat path label: {label}, r_multiple: {r_mult}")
assert label == 0, (
    f"FAIL: Flat path with transaction cost should be a loss (label=0), got {label}"
)
print("PASS: Transaction cost correctly makes flat path a loss")
```

---

## Check 6 — Non-Signal Bars Get NoTrade Label (2)

```python
import pandas as pd, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.labels import triple_barrier_label
import numpy as np

dates = pd.date_range("2024-01-02 10:00", periods=5, freq="5min", tz="America/New_York")
df = pd.DataFrame({
    "open":  [100.0]*5,
    "high":  [101.0]*5,
    "low":   [99.0]*5,
    "close": [100.5]*5,
    "atr_14": [2.0]*5,
}, index=dates)

# Signal only on bar 0 — bars 1-4 are non-signal
signal = pd.Series([1, 0, 0, 0, 0], index=dates)
result = triple_barrier_label(df, signal, df["atr_14"])

# Non-signal bars should have NaN label (filled to 2 by train.py)
for i in range(1, 5):
    lbl = result["label"].iloc[i]
    assert pd.isna(lbl), (
        f"FAIL: Non-signal bar {i} should have NaN label, got {lbl}"
    )
print("PASS: Non-signal bars have NaN label (correctly filled to NoTrade=2 in training)")
```

---

## Check 7 — _load_strategy_frame Uses Triple-Barrier Labels

Read `ml/train.py` and find `_load_strategy_frame()`. Verify:

1. It accesses a per-strategy label column like `label_{strategy_name}` (e.g. `label_ifvg`)
2. It fills NaN values in that column with `2` (NoTrade)
3. It does NOT use the generic `label` column (shift-N return label) as the training target

Print the relevant lines from `_load_strategy_frame()` in your audit report. Write PASS if the above three conditions are met, FAIL if any are not.

```python
# Read and print the _load_strategy_frame section
from pathlib import Path
content = Path(r"C:\Users\Luker\strategyLabbrain\ml\train.py").read_text(encoding="utf-8")

import re
match = re.search(r"def _load_strategy_frame.*?(?=\ndef |\Z)", content, re.DOTALL)
if match:
    snippet = match.group(0)[:1500]
    print(snippet)

    uses_per_strategy_label = "label_" in snippet
    fills_notrade = "fillna(2)" in snippet or "fill_value=2" in snippet
    print(f"\nUses per-strategy label column: {uses_per_strategy_label}")
    print(f"Fills NaN with NoTrade (2):     {fills_notrade}")

    assert uses_per_strategy_label, "FAIL: Does not use per-strategy label column"
    assert fills_notrade, "FAIL: Does not fill NaN labels with NoTrade (2)"
    print("PASS: _load_strategy_frame correctly wired to triple-barrier labels")
else:
    print("FAIL: Could not find _load_strategy_frame in train.py")
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

Write `ml/AGENT3D_AUDIT.md`:

```markdown
# Agent 3D Audit Report

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | Stop hit first labels loss (0) | PASS/FAIL | |
| 2 | Target hit first labels win (1) | PASS/FAIL | |
| 3 | Short signal symmetry | PASS/FAIL | |
| 4 | Session boundary exit at 15:00 | PASS/FAIL | |
| 5 | Transaction cost makes flat path a loss | PASS/FAIL | |
| 6 | Non-signal bars get NaN (filled to 2) | PASS/FAIL | |
| 7 | _load_strategy_frame wired to triple-barrier | PASS/FAIL | [paste relevant lines] |
| 8 | Full test suite | PASS/FAIL | [N passed] |

## Verdict
**3D APPROVED — proceed to 3E**
OR
**3D BLOCKED — [exact reasons]**
```

# Agent 4A-Audit — Pivot Features and Session Signal Review

**Role:** Independent auditor for Agent 4A's work
**Blocks:** Agent 4B cannot start until this audit passes
**Approach:** Read the spec and the code cold. Do not read Agent 4A's status file first. Form your own opinion from the code and data, then compare.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Strategies\Session Level Pivots.md` — the spec
2. `C:\Users\Luker\strategyLabbrain\ml\signal_generators.py` — look for session_pivot_signal()
3. `C:\Users\Luker\strategyLabbrain\Implementation\camarilla_pivot_generator.py`
4. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py`

Read the code before reading Agent 4A's status file. Your job is to find what the builder missed, not to confirm what they reported.

---

## Absolute Paths

```
Project root:      C:\Users\Luker\strategyLabbrain
Parquets:          C:\Users\Luker\strategyLabbrain\ml\data\
Signal generators: C:\Users\Luker\strategyLabbrain\ml\signal_generators.py
Tests:             C:\Users\Luker\strategyLabbrain\ml\tests\
```

---

## Audit Check 1 — Camarilla Uses Prior Day Data

Open `dataset_builder.py` and `camarilla_pivot_generator.py`. Find where H3/H4/S3/S4 are computed.

Answer these questions and print your findings:
- What is the source of `prev_high`, `prev_low`, `prev_close` used in the Camarilla formula?
- Is this source from the prior calendar day or from the current session?
- If the parquet only contains 09:30–15:00 session bars, how does the generator access yesterday's OHLC?

**Pass condition:** Generator explicitly uses data from the prior calendar day. The word "prior", "yesterday", or `shift(1)` on a daily groupby must appear in the level computation logic.

**Fail condition:** Generator uses `df["high"]`, `df["low"]`, `df["close"]` without a daily offset — that means it uses today's intraday data, which is lookahead.

Write code to spot-check:
```python
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")

# Take a known date, manually compute Camarilla from prior day raw CSV
# and compare to the value in the parquet
test_date = "2024-03-15"
raw = pd.read_csv(r"C:\Users\Luker\strategyLabbrain\data\mnq_5min_databento.csv",
                  index_col=0, parse_dates=True)
raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
prior_day = "2024-03-14"
prior = raw[raw.index.date == pd.Timestamp(prior_day).date()]
if len(prior) > 0:
    prev_high  = prior["high"].max()
    prev_low   = prior["low"].min()
    prev_close = prior["close"].iloc[-1]
    expected_h4 = prev_close + (prev_high - prev_low) * 0.55
    expected_s4 = prev_close - (prev_high - prev_low) * 0.55
    parquet_bars = df[df.index.date == pd.Timestamp(test_date).date()]
    if len(parquet_bars) > 0:
        actual_h4_dist = parquet_bars["camarilla_h4_dist"].iloc[0]
        actual_price   = parquet_bars["close"].iloc[0]
        actual_atr     = parquet_bars["atr_14"].iloc[0]
        implied_h4     = actual_price - actual_h4_dist * actual_atr
        print(f"Expected H4: {expected_h4:.2f}")
        print(f"Implied H4 from parquet: {implied_h4:.2f}")
        diff = abs(expected_h4 - implied_h4)
        print(f"Difference: {diff:.4f} {'✓ PASS' if diff < 1.0 else '✗ FAIL'}")
```

Record result: PASS or FAIL with exact numbers.

---

## Audit Check 2 — Rejection Candle Logic

Open `signal_generators.py`. Find `session_pivot_signal()`. Read the long signal logic carefully.

**The critical question:** Does the code check that `close > level` for long signals AND `close < level` for short signals?

Look for a bar that TOUCHES S4 but closes BELOW S4. This must NOT generate a long signal.

Write a synthetic test:
```python
import pandas as pd
import numpy as np
from ml.signal_generators import session_pivot_signal

# Synthetic DataFrame — one bar that touches S4 but closes below it
dates = pd.date_range("2024-01-02 10:00", periods=20, freq="5min",
                       tz="America/New_York")
df = pd.DataFrame({
    "open":  [100.0] * 20,
    "high":  [101.0] * 20,
    "low":   [98.5]  + [100.0] * 19,   # bar 0 dips to 98.5
    "close": [98.2]  + [100.0] * 19,   # bar 0 closes BELOW S4 (98.5) → NOT a rejection
    "volume": [1000] * 20,
    "atr_14": [2.0]  * 20,
    # Camarilla levels — S4 at 98.5
    "camarilla_s4": [98.5] * 20,
    "camarilla_s3": [99.2] * 20,
    "camarilla_h3": [100.8] * 20,
    "camarilla_h4": [101.5] * 20,
    "prev_day_close": [101.0] * 20,    # close < prev_day_close = mean reversion context satisfied
    # Session levels — set far away so they don't trigger
    "asia_high": [110.0] * 20, "asia_low": [90.0] * 20,
    "london_high": [110.0] * 20, "london_low": [90.0] * 20,
    "premarket_high": [110.0] * 20, "premarket_low": [90.0] * 20,
    "prev_day_high": [110.0] * 20, "prev_day_low": [90.0] * 20,
}, index=dates)

signals = session_pivot_signal(df)
result = signals.iloc[0]
print(f"Signal on breakdown bar: {result}")
assert result == 0, (
    f"FAIL: Bar closing below S4 generated signal={result}. "
    "Breakdown bars must NOT generate long signals. Rejection logic is broken."
)
print("✓ PASS: Rejection candle correctly requires close above level")
```

Record result: PASS or FAIL.

---

## Audit Check 3 — Daily Cap Is Shared Across All Levels

Verify the 2-signal-per-day cap applies across ALL levels combined, not 2 per level.

```python
import pandas as pd
from ml.signal_generators import session_pivot_signal

# Build a DataFrame where 3 different levels each offer a valid signal on the same day
# Only the first 2 should fire, the 3rd should be blocked
# This requires 3 bars on the same date, each touching a different level cleanly

dates = pd.date_range("2024-01-02 09:35", periods=3, freq="30min",
                       tz="America/New_York")
df = pd.DataFrame({
    "open":  [100.0, 100.0, 100.0],
    "high":  [102.0, 102.0, 102.0],
    "low":   [98.4,  99.1,  98.4],    # bar 0 touches S4, bar 1 touches S3, bar 2 touches S4 again
    "close": [99.5,  99.5,  99.5],    # all close above S4/S3 = valid rejection
    "volume": [1000, 1000, 1000],
    "atr_14": [2.0,  2.0,  2.0],
    "camarilla_s4": [98.5, 98.5, 98.5],
    "camarilla_s3": [99.2, 99.2, 99.2],
    "camarilla_h3": [100.8]*3,
    "camarilla_h4": [101.5]*3,
    "prev_day_close": [101.0]*3,
    "asia_high": [110.0]*3, "asia_low": [90.0]*3,
    "london_high": [110.0]*3, "london_low": [90.0]*3,
    "premarket_high": [110.0]*3, "premarket_low": [90.0]*3,
    "prev_day_high": [110.0]*3, "prev_day_low": [90.0]*3,
}, index=dates)

signals = session_pivot_signal(df)
print(f"Bar 0: {signals.iloc[0]}")
print(f"Bar 1: {signals.iloc[1]}")
print(f"Bar 2: {signals.iloc[2]}")
total = (signals != 0).sum()
assert total == 2, (
    f"FAIL: Expected 2 signals (daily cap), got {total}. "
    "The 2/day cap may be per-level rather than shared."
)
print("✓ PASS: Daily cap correctly shared across all levels")
```

Record result: PASS or FAIL.

---

## Audit Check 4 — NY AM Running High Excludes Current Bar

Find where NY AM running high/low is computed in `dataset_builder.py`.

Look for `.expanding().max()`. Confirm `.shift(1)` is applied after the max. Without `shift(1)`, bar N's own high is included in bar N's feature — 1-bar lookahead.

```python
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")

# Find a bar where the session high is a new intraday high
# The feature at that bar should NOT include that bar's high
# It should equal the max of all prior session bars

if "nyam_high_dist" in df.columns:
    # Take first NY session day available
    first_date = df.index.date[0]
    day_bars = df[df.index.date == first_date]

    # Compute expected running high WITHOUT including current bar
    expected_running_high = day_bars["high"].expanding().max().shift(1)

    # The feature is stored as distance: (close - nyam_high) / atr_14
    # Recover implied level: nyam_high = close - nyam_high_dist * atr_14
    implied_nyam_high = day_bars["close"] - day_bars["nyam_high_dist"] * day_bars["atr_14"]

    # First bar: nyam_high should be NaN (no prior bars) or set to a sentinel
    # Remaining bars: implied level should match expected running max
    for i in range(1, min(5, len(day_bars))):
        exp = expected_running_high.iloc[i]
        act = implied_nyam_high.iloc[i]
        print(f"  Bar {i}: expected={exp:.2f}, implied={act:.2f}, match={abs(exp-act)<0.5}")
else:
    print("nyam_high_dist column not found — may use different column name")
    print("Available columns with 'nyam' or 'ny_am':", 
          [c for c in df.columns if "nyam" in c.lower() or "ny_am" in c.lower()])
```

Record result: PASS or FAIL.

---

## Audit Check 5 — Parquet Column Integrity

Confirm the original feature columns are present and unchanged after rebuild.

```python
import pandas as pd

df5 = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
df1 = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_1min.parquet")

# Verify core columns
core_required = [
    "open", "high", "low", "close", "volume",
    "atr_14", "synthetic_delta",
    "ifvg_signal", "ifvg_open_signal",
    "orb_ib_signal", "orb_vol_signal", "orb_wick_signal", "orb_va_signal",
    "ttm_signal", "connors_signal",
    "session_pivot_signal",
    "camarilla_h4_dist", "camarilla_h3_dist",
    "camarilla_s4_dist", "camarilla_s3_dist",
]

for col in core_required:
    in5 = col in df5.columns
    in1 = col in df1.columns
    status = "✓" if (in5 and in1) else "✗ MISSING"
    print(f"  {col}: 5min={in5}, 1min={in1} {status}")

# Verify consistent columns across all parquets
df2 = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_2min.parquet")
df3 = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_3min.parquet")
cols_match = (set(df1.columns) == set(df2.columns) == set(df3.columns) == set(df5.columns))
print(f"\nAll parquets have identical columns: {'✓ YES' if cols_match else '✗ NO'}")
if not cols_match:
    print("5min columns:", sorted(df5.columns.tolist()))
    print("1min columns:", sorted(df1.columns.tolist()))
```

Record result: PASS or FAIL.

---

## Audit Check 6 — Run Full Test Suite Independently

```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```

Every test that passed before Agent 4A must still pass. All new Agent 4A tests must pass. Report the full count.

---

## Audit Check 7 — Signal Rate Sanity

```python
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
vc = df["session_pivot_signal"].value_counts()
n_bars = len(df)
n_signals = vc.get(1, 0) + vc.get(-1, 0)
rate = n_signals / n_bars

print(f"Total bars: {n_bars}")
print(f"Long signals: {vc.get(1,0)}")
print(f"Short signals: {vc.get(-1,0)}")
print(f"Signal rate: {rate:.4f} ({rate*100:.2f}%)")

# Sanity bounds: MNQ trades 5 years × 252 days × 78 5min bars ≈ 98,280 bars
# Max 2 signals/day × 1,260 days = 2,520 max possible signals
# Expected signal rate: 0.5%–2.5%
assert 0.001 < rate < 0.05, (
    f"Signal rate {rate:.4f} is outside expected range 0.1%–5%. "
    "Either the signal fires too rarely (logic too strict) or too often (proximity too wide)."
)
print("✓ PASS: Signal rate in expected range")
```

Record result: PASS or FAIL with exact numbers.

---

## Audit Decision

After completing all 7 checks, write `ml/AGENT4A_AUDIT.md`:

```markdown
# Agent 4A Audit Report

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | Camarilla uses prior day data | PASS/FAIL | [exact values] |
| 2 | Rejection candle logic | PASS/FAIL | [test output] |
| 3 | Daily cap shared across levels | PASS/FAIL | [test output] |
| 4 | NY AM running high excludes current bar | PASS/FAIL | |
| 5 | Parquet column integrity | PASS/FAIL | |
| 6 | Full test suite | PASS/FAIL | [N passed, N failed] |
| 7 | Signal rate sanity | PASS/FAIL | [rate, counts] |

## Verdict
**4A APPROVED — proceed to 4B**
OR
**4A BLOCKED — [list every failing check with exact reason]**

Do not write APPROVED unless ALL 7 checks pass.
If BLOCKED: list exactly what must be fixed. Agent 4A must fix and re-submit for re-audit.
```

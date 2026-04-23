# Agent 4A — Pivot Features and Session Pivot Signal

**Phase:** Data layer fix
**Deliverable:** Rebuilt feature parquets with verified non-zero Camarilla pivot features + working `session_pivot_signal` column
**Estimated session length:** 1 full context session
**Prerequisite:** Agents 1A, 1B, 2, 3A, 3B all complete. Do not modify any existing artifact or checkpoint — this agent only touches `dataset_builder.py`, `signal_generators.py`, and rebuilds parquets.

---

## Why This Agent Exists

Two problems found in the existing data layer:

**Problem 1 — Camarilla pivot features may be missing or all-zero in the parquets.**
`Implementation/camarilla_pivot_generator.py` exists but may never have been correctly wired into `ml/dataset_builder.py`. If the pivot columns (camarilla_h3_dist, camarilla_h4_dist, camarilla_s3_dist, camarilla_s4_dist, session high/low distances) are missing or all-zero, the CNN trained on 5 years of data has been blind to the most important structural price levels.

**Problem 2 — Session Level Pivots has no signal column.**
`session_pivot_signal` does not exist in any parquet. Session Level Pivots was previously "feature engineering only." It is now an active strategy paired with ORB IB in Model 3. Its signal column must be added before any retraining.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Home.md`
2. `C:\Users\Luker\strategyLabbrain\MASTER_CHANGE_PLAN.md`
3. `C:\Users\Luker\strategyLabbrain\Strategies\Session Level Pivots.md` — full signal spec
4. `C:\Users\Luker\strategyLabbrain\Implementation\camarilla_pivot_generator.py` — read in full
5. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — read in full
6. `C:\Users\Luker\strategyLabbrain\ml\signal_generators.py` — read in full
7. `C:\Users\Luker\strategyLabbrain\ml\AGENT3B_AUDIT.md`

---

## Absolute Paths

```
Project root:            C:\Users\Luker\strategyLabbrain
camarilla generator:     C:\Users\Luker\strategyLabbrain\Implementation\camarilla_pivot_generator.py
signal generators:       C:\Users\Luker\strategyLabbrain\ml\signal_generators.py
dataset builder:         C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py
Feature parquets:        C:\Users\Luker\strategyLabbrain\ml\data\
Raw CSV data:            C:\Users\Luker\strategyLabbrain\data\
```

---

## Task 1 — Verify Current State

Run the full test suite first:
```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```
Record the exact pass/fail/skip counts. Do not proceed if tests are not green.

Then run the parquet inspection script to confirm what is and is not in the current parquets:

```python
import pandas as pd, os

parquet_dir = r"C:\Users\Luker\strategyLabbrain\ml\data"
for fname in os.listdir(parquet_dir):
    if not fname.endswith(".parquet"):
        continue
    df = pd.read_parquet(os.path.join(parquet_dir, fname))
    print(f"\n=== {fname} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check pivot columns
    pivot_cols = [c for c in df.columns if any(x in c for x in
                  ["camarilla", "asia_", "london_", "premarket_", "prev_day_", "prev_week_"])]
    print(f"Pivot columns found ({len(pivot_cols)}): {pivot_cols}")

    if pivot_cols:
        desc = df[pivot_cols].describe()
        zero_cols = [c for c in pivot_cols if df[c].abs().max() < 1e-6]
        nan_cols  = [c for c in pivot_cols if df[c].isna().all()]
        print(f"All-zero pivot cols: {zero_cols}")
        print(f"All-NaN pivot cols:  {nan_cols}")

    # Check session pivot signal
    if "session_pivot_signal" in df.columns:
        vc = df["session_pivot_signal"].value_counts()
        print(f"session_pivot_signal counts: {vc.to_dict()}")
    else:
        print("session_pivot_signal: MISSING")
```

Record output verbatim in AGENT4A_STATUS.md before proceeding.

---

## Task 2 — Wire Camarilla Pivot Generator into dataset_builder.py

Open `Implementation/camarilla_pivot_generator.py` and understand its interface. Then wire it into `ml/dataset_builder.py` so Camarilla levels are computed and distance features are added to the feature matrix.

**Critical requirements:**

- Camarilla levels (H3, H4, S3, S4) must be computed from the **prior day's OHLC only**. The generator must have access to the full raw DataFrame (not session-filtered) so it can look back to the previous day's close/high/low. Session filtering happens AFTER level computation.
- The raw prior-day close/high/low must come from bars outside the 09:30–15:00 session window. Confirm the generator loads or receives the full daily data before applying the session filter.
- Distance features are: `(close - level) / atr_14` for each of H3, H4, S3, S4. Positive = price above level, negative = price below. ATR must be the same `atr_14` column already in the parquet.
- If `atr_14` is NaN or zero on a bar, set all distance features for that bar to 0.0. Do not divide by zero.
- Session high/low distance features (Asia, London, Pre-Market) are computed the same way. These sessions end before the 09:30 NY open, so they are fully known at session start — no lookahead.
- NY AM running high/low: use `.expanding().max().shift(1)` so bar N uses the max of bars 0 through N-1. The `shift(1)` is mandatory — it prevents bar N's high from appearing in bar N's feature.
- All new columns must be added to the feature matrix without changing the order or values of existing columns. The existing 35-column feature contract must remain intact. New pivot columns are additive.

After wiring, verify:
```python
df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
pivot_cols = [c for c in df.columns if "camarilla" in c]
assert len(pivot_cols) >= 4, "Missing camarilla columns"
assert df[pivot_cols].abs().max().max() > 0, "All camarilla distances are zero"
assert df[pivot_cols].isna().mean().max() < 0.15, "Too many NaN values in pivot cols"
print("Camarilla columns verified:", pivot_cols)
```

---

## Task 3 — Add session_pivot_signal to signal_generators.py

Add the `session_pivot_signal()` function to `ml/signal_generators.py` following the exact specification in `Strategies/Session Level Pivots.md`.

**Logic requirements (read the strategy spec first, then implement):**

Long signal (+1) — ALL of the following on the same bar:
1. `bar.low <= level + (proximity_atr * atr_14)` where level is S4 or S3 or a session/prev-day low
2. `bar.close > level` — close is ABOVE the level (rejection candle). If close <= level → no signal, price broke through
3. `bar.close < prev_day_close` — mean reversion context
4. Daily count < max_per_day (2)

Short signal (-1) — ALL of the following on the same bar:
1. `bar.high >= level - (proximity_atr * atr_14)` where level is H4 or H3 or a session/prev-day high
2. `bar.close < level` — close is BELOW the level. If close >= level → no signal, price broke out
3. `bar.close > prev_day_close` — mean reversion context
4. Daily count < max_per_day (2)

Level priority (if multiple levels touched same bar): H4/S4 first, H3/S3 second, prev_day high/low third, session highs/lows last. Only ONE signal fires per bar.

**Pitfalls to avoid:**
- Do NOT fire a signal if the bar closes through the level (that is a breakout, not a rejection)
- Do NOT fire a signal during ATR warmup (atr_14 is NaN) — return 0 for those bars
- The daily cap counter MUST be keyed by calendar date, not bar index. Reset at session open.
- prev_day_close must come from the prior calendar day's close, not yesterday's last session bar (which might be 14:59 ET). Use the same source that computes Camarilla levels.

After implementing, verify on real data:
```python
from ml.signal_generators import session_pivot_signal
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
sig = session_pivot_signal(df)
vc = sig.value_counts()
print("Signal distribution:", vc.to_dict())
assert 1 in vc.index, "No long signals generated — check logic"
assert -1 in vc.index, "No short signals generated — check logic"
n_signals = vc.get(1, 0) + vc.get(-1, 0)
n_bars = len(df)
rate = n_signals / n_bars
assert 0.001 < rate < 0.05, f"Signal rate {rate:.4f} is outside expected 0.1%-5% range"
print(f"Signal rate: {rate:.4f} ({n_signals} signals in {n_bars} bars)")
```

If zero signals: the rejection candle condition or the proximity threshold is too strict. Debug by printing bars where the level was touched and checking why the close condition failed.

---

## Task 4 — Add session_pivot_signal to dataset_builder.py

Wire `session_pivot_signal()` into `dataset_builder.py` alongside the other signal generators so it is computed and included in the parquet rebuild.

The column name must be exactly: `session_pivot_signal`

This column must appear in the feature matrix before parquet export.

---

## Task 5 — Rebuild All Parquets

```
python ml/dataset_builder.py --rebuild
```

Or however the rebuild command is structured in the existing code. Rebuild all 4 timeframe parquets (1min, 2min, 3min, 5min).

After rebuild, run verification on each parquet:

```python
import pandas as pd, os

required_pivot_cols = [
    "camarilla_h3_dist", "camarilla_h4_dist",
    "camarilla_s3_dist", "camarilla_s4_dist",
]
required_signal_col = "session_pivot_signal"

parquet_dir = r"C:\Users\Luker\strategyLabbrain\ml\data"
for fname in os.listdir(parquet_dir):
    if not fname.endswith(".parquet"):
        continue
    df = pd.read_parquet(os.path.join(parquet_dir, fname))
    print(f"\n=== {fname} ===")

    # Verify pivot columns present and non-zero
    for col in required_pivot_cols:
        assert col in df.columns, f"MISSING: {col}"
        assert df[col].abs().max() > 0, f"ALL ZERO: {col}"
        nan_pct = df[col].isna().mean()
        assert nan_pct < 0.15, f"TOO MANY NANS ({nan_pct:.1%}): {col}"
        print(f"  {col}: max={df[col].abs().max():.4f}, nan%={nan_pct:.1%} ✓")

    # Verify signal column present
    assert required_signal_col in df.columns, f"MISSING: {required_signal_col}"
    vc = df[required_signal_col].value_counts()
    assert 1 in vc.index and -1 in vc.index, "Signal missing longs or shorts"
    print(f"  {required_signal_col}: {vc.to_dict()} ✓")

    # Verify existing 35 feature columns still present and unchanged
    original_35 = [  # List all 35 original feature column names here
        "open", "high", "low", "close", "volume", "log_volume",
        "synthetic_delta", "log_ret_1", "log_ret_5", "atr_14",
        "orb_vol_signal", "orb_wick_signal", "orb_ib_signal",
        "ifvg_signal", "ifvg_open_signal", "ttm_signal", "connors_signal",
        # ... add all 35
    ]
    for col in original_35:
        assert col in df.columns, f"ORIGINAL COLUMN MISSING AFTER REBUILD: {col}"
    print(f"  All original feature columns present ✓")
```

---

## Task 6 — Run Full Test Suite

```
python -m pytest ml/tests -v --tb=short
```

All tests that were passing before this agent started must still pass. If any previously-passing test fails, the parquet rebuild broke something — fix it before proceeding.

Add to the test suite (`ml/tests/test_pipeline.py` or a new `test_agent4a.py`):

- `test_camarilla_h4_above_h3` — H4 > H3 always (formula property)
- `test_camarilla_s3_above_s4` — S3 > S4 always
- `test_camarilla_uses_prior_day` — feeding in a modified prior day changes current day's levels, feeding in a modified CURRENT day does not
- `test_session_pivot_signal_rejection_only` — a bar that touches S4 but closes below S4 does NOT generate a long signal
- `test_session_pivot_signal_daily_cap` — after 2 signals on a calendar day, no more signals fire that day regardless of conditions
- `test_session_pivot_signal_no_signal_at_atr_warmup` — first 14 bars of any parquet return 0 signal
- `test_ny_am_high_excludes_current_bar` — NY AM running high feature at bar N equals max(high[0:N-1]), not max(high[0:N])

---

## Task 7 — Write AGENT4A_STATUS.md

Location: `C:\Users\Luker\strategyLabbrain\ml\AGENT4A_STATUS.md`

```markdown
## Completed
- [list tasks]

## Parquet State Before Rebuild
- [paste output of Task 1 inspection script]

## Parquet State After Rebuild
- [paste output of Task 5 verification script]

## Session Pivot Signal Distribution (5min parquet)
- Long signals: [N]
- Short signals: [N]
- Signal rate: [%]

## Camarilla Values Sample (first 5 signal bars)
- [sample of H3/H4/S3/S4 values with prior-day OHLC used]

## Test Suite Results
- Before: [N passed, N failed]
- After: [N passed, N failed]
- New tests added: [N]

## Feature Column Count
- Original: 35 columns
- After rebuild: [N] columns (35 original + [N] new pivot/signal columns)
- Column order preserved: yes/no

## Known Issues
- [any]

## Next Agent
- Agent 4A-Audit reads this file and inspects the parquets independently.
- Agent 4B starts after 4A-Audit passes.
```

---

## Deliverables

```
ml/
├── signal_generators.py          ← session_pivot_signal() added
├── dataset_builder.py            ← camarilla generator wired in, session pivot signal added
├── data/
│   ├── features_mnq_1min.parquet ← rebuilt with pivot cols + session_pivot_signal
│   ├── features_mnq_2min.parquet ← rebuilt
│   ├── features_mnq_3min.parquet ← rebuilt
│   └── features_mnq_5min.parquet ← rebuilt
├── tests/
│   └── test_agent4a.py           ← 7 new tests
└── AGENT4A_STATUS.md
```

---

## Logic Gaps to Guard Against

1. **Camarilla from today's data:** Using today's OHLC for pivot computation is a silent lookahead bug. The generator must explicitly load or receive prior-day data. If the parquet only contains session bars (09:30–15:00), the generator must separately access the full raw CSV or pre-computed daily OHLC to get yesterday's close/high/low.

2. **Rejection vs breakout confusion:** A bar touching S4 and closing BELOW S4 is a bearish breakout, NOT a long signal. The close condition (`close > level` for longs) is the critical gate. Many implementations check only proximity without checking the close side.

3. **Daily cap across all levels:** The 2-signal cap is shared across all pivot levels combined — not 2 per level. If H4 fires a short and then S4 setup appears, the second one is allowed (count=1 → 2), but a third setup of any kind is blocked.

4. **NY AM running high shift:** `.expanding().max().shift(1)` is mandatory. Without `.shift(1)`, bar N's own high is included in the running max used as bar N's feature. That is a 1-bar lookahead.

5. **NaN handling in pivot distances:** Early bars and session boundaries may have NaN ATR or NaN level values. All NaN distances must be filled with 0.0, not dropped. Dropping rows changes the parquet index and breaks the sliding window dataset.

6. **Column order preservation:** The existing 35 feature columns must remain in the same positions in the parquet. New columns are appended. The feature_order JSON files and scaler objects reference column positions — reordering breaks them.

7. **Parquet schema consistency:** All 4 rebuilt parquets must have identical column names (though different row counts). Mixed column sets across timeframes will cause silent failures downstream when the model config specifies feature_cols.

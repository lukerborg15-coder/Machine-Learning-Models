# Agent 1A — Data Loader & Signal Generators

**Phase:** A
**Deliverable:** Working data loader + all 7 signal generator functions with passing tests
**Estimated session length:** 1 full context session
**Prerequisite:** None — this is the first agent

---

## Absolute Paths (use these for all file references)

```
Project root:       C:\Users\Luker\strategyLabbrain
Data (CSV files):   C:\Users\Luker\strategyLabbrain\data
ML output folder:   C:\Users\Luker\strategyLabbrain\ml
Strategy specs:     C:\Users\Luker\strategyLabbrain\Strategies\
```

Everything is built from scratch inside `strategyLabbrain\ml\`. Do not reference StrategyLab.

---

## Context to Read First (in this order)

1. [[Home]]
2. [[Architecture Overview]]
3. [[ML Operators Guide]]
4. [[Testing Requirements]]
5. [[Tick Data - Delta Features]]
6. Strategy specs: [[IFVG]], [[IFVG - Open Variant]], [[ORB IB - Initial Balance]], [[ORB Volatility Filtered]], [[ORB Wick Rejection]], [[ORB Volume Adaptive]], [[TTMSqueeze]], [[ConnorsRSI2]]

---

## Folder Structure to Create

```
strategyLabbrain\ml\
├── signal_generators.py     ← build from scratch using vault strategy specs
├── dataset_builder.py       ← stub: canonical load_data() only, Agent 1B completes
├── topstep_risk.py          ← stub only, Agent 2 builds
├── model.py                 ← stub only, Agent 2 builds
├── train.py                 ← stub only, Agent 2 builds
├── evaluate.py              ← stub only, Agent 2 builds
├── export_onnx.py           ← stub only, Agent 3 builds
├── tests\
│   ├── __init__.py
│   └── test_pipeline.py     ← Agent 1A writes data + signal tests
├── artifacts\               ← empty, for model checkpoints and scalers
└── data\                    ← empty, Agent 1B populates with parquets
```

---

## Task List

### Task 1 — Create the ml\ folder structure
Create all folders and stub files listed above. Stubs just need a module docstring — no implementation yet.

### Task 2 — Implement data loader
In `dataset_builder.py`, implement `load_data(instrument, timeframe, session_only=True)`:
- Reads `C:\Users\Luker\strategyLabbrain\data\{instrument}_{timeframe}_databento.csv`
- Converts datetime column to Eastern time (tz-aware, `America/New_York`)
- If `session_only=True`, session filters to 09:30–15:00 ET only (use `df.between_time('09:30', '15:00')` — inclusive on both ends)
  - **Standardized**: all files use `'15:00'` as end time, NOT `'14:59'`
- If `session_only=False`, returns the full tz-aware MNQ dataset with no RTH filter applied
- Returns clean DataFrame with DatetimeIndex in both cases
- Available files for the current build: `mnq_1min`, `mnq_2min`, `mnq_3min`, `mnq_5min`, `mnq_15min`, `mnq_30min`, `mnq_1h`

**Canonical usage:**
- Agent 1A signal generation uses `session_only=True`
- Agent 1B label generation uses `session_only=False`, then applies session filtering after forward labels are aligned

### Task 3 — Build signal_generators.py

**Three generators are pre-built. Do NOT rewrite them. Import them directly from `Implementation/`:**

**IFVG and IFVG Open Variant** — import from:
`C:\Users\Luker\strategyLabbrain\Implementation\ifvg_generator.py`

**TTMSqueeze** — import from:
`C:\Users\Luker\strategyLabbrain\Implementation\ttm_squeeze_generator.py`

Run the standalone sanity check on each pre-built file first:
```bash
python C:\Users\Luker\strategyLabbrain\Implementation\ifvg_generator.py
python C:\Users\Luker\strategyLabbrain\Implementation\ttm_squeeze_generator.py
```

Do NOT rewrite the IFVG or TTMSqueeze logic. Do NOT copy those implementations into `ml/`. `ml\signal_generators.py` should wrap or re-export the imported functions plus the locally implemented ORB/Connors generators.

**HTF data for IFVG confluence:** When calling `ifvg_combined()`, pass the HTF DataFrame:
- For 1min entry TF: load 5min or 15min data as `htf_df`
- For 5min entry TF: load 15min or 1H data as `htf_df`
- Load HTF data using the same `load_data()` function with the appropriate timeframe and `session_only=True`
- Example: `ifvg_combined(df_1min, timeframe_minutes=1, htf_df=load_data('mnq', '5min', session_only=True))`
- If `htf_df=None`, the HTF confluence gate is skipped — this reduces signal quality significantly

**Five functions to write from spec:**

```python
def orb_volatility_filtered(df, or_minutes=10, atr_period=14,
                             atr_lookback=100, min_atr_pct=25,
                             max_atr_pct=85, max_signals_per_day=1) -> pd.Series:
    # Returns Series of 1 (long), -1 (short), 0 (no signal)
    # Column name in feature matrix: orb_vol_signal
    # Rules: 10-min opening range, ATR percentile filter 25th-85th pct,
    #        breakout on close above/below OR, max 1 signal/day
    #        IMPORTANT: first valid breakout bar is the bar AFTER the OR closes
    #        (e.g., on 5min chart with 10min OR: OR uses 9:30+9:35 bars,
    #         first breakout candidate is the 9:40 bar)
    # Spec: strategyLabbrain\Strategies\ORB Volatility Filtered.md

def orb_wick_rejection(df, or_minutes=10, min_body_pct=0.55,
                        atr_period=14, max_signals_per_day=1) -> pd.Series:
    # Returns Series of 1/-1/0
    # Column name in feature matrix: orb_wick_signal
    # Rules: 10-min opening range, breakout candle body >= 55% of bar range,
    #        max 1 signal/day, guard: skip if high == low
    #        IMPORTANT: first valid breakout bar is the bar AFTER the OR closes
    # Spec: strategyLabbrain\Strategies\ORB Wick Rejection.md

def orb_initial_balance(df, atr_period=14,
                         max_signals_per_day=1) -> pd.Series:
    # Returns Series of 1/-1/0
    # Column name in feature matrix: orb_ib_signal
    # Rules: 60-min opening range (IB: 9:30-10:30), breakout targets at IB extension,
    #        max 1 signal/day
    #        IMPORTANT: first valid breakout bar is the bar AFTER 10:30
    #        (e.g., on 5min chart: first candidate is the 10:30 bar close)
    # Spec: strategyLabbrain\Strategies\ORB IB - Initial Balance.md

def orb_volume_adaptive(df, or_minutes=10, vol_multiplier=1.5,
                         atr_period=14, max_signals_per_day=1) -> pd.Series:
    # Returns Series of 1/-1/0
    # Column name in feature matrix: orb_va_signal
    # Rules: 10-min opening range, breakout bar volume >= or_avg_volume × vol_multiplier,
    #        max 1 signal/day, guard: skip if or_avg_volume == 0
    #        IMPORTANT: first valid breakout bar is the bar AFTER the OR closes
    # Spec: strategyLabbrain\Strategies\ORB Volume Adaptive.md

def connors_rsi2(df, rsi_period=2, rsi_entry=10, rsi_exit=90,
                  exit_ma=5, trend_ma=200, atr_period=14) -> pd.Series:
    # Returns Series of 1/-1/0
    # Column name in feature matrix: connors_signal
    # Rules: RSI(2) < 10 long entry, > 90 short entry, SMA(200) trend filter,
    #        exit on RSI recovery or SMA(5) cross
    #        Only take FIRST entry in each consecutive RSI sequence
    #        No signals before bar 200 (SMA warmup)
    # Spec: strategyLabbrain\Strategies\ConnorsRSI2.md
```

**Integration in signal_generators.py:**
```python
from Implementation.ifvg_generator import ifvg_combined
from Implementation.ttm_squeeze_generator import ttm_squeeze

# All functions available as:
# orb_volatility_filtered, orb_wick_rejection, orb_initial_balance,
# orb_volume_adaptive, connors_rsi2, ifvg_combined, ttm_squeeze
```

**IFVG shared daily limit:** The `ifvg` and `ifvg_open` generators must share a counter — combined signals from both cannot exceed 2 per day. This is already handled inside `ifvg_combined()`.

### Task 4 — Write and run signal-specific tests
In `ml\tests\test_pipeline.py`, implement:
- `test_timezone_is_eastern()` — verify index is tz-aware Eastern
- `test_session_hours_only()` — no bars outside 09:30–15:00
- `test_no_duplicate_timestamps()` — index is unique
- `test_strategy_signal_no_lookahead()` — randomize a single row, signals before it don't change
- `test_max_signals_per_day()` — ORB strategies: max 1/day
- `test_ifvg_shared_daily_limit()` — IFVG + IFVG Open combined ≤ 2/day

**All tests must pass before writing AGENT1A_STATUS.md.**

---

## Deliverables

```
strategyLabbrain\ml\
├── signal_generators.py          ← all 7 signal functions implemented
├── dataset_builder.py            ← canonical load_data() implemented (rest stubbed for Agent 1B)
├── tests\test_pipeline.py        ← data + signal tests written and passing
└── AGENT1A_STATUS.md
```

---

## AGENT1A_STATUS.md Format

```markdown
## Completed
- [list all tasks completed]

## Artifacts
- signal_generators.py: [list of functions implemented]
- dataset_builder.py: load_data(instrument, timeframe, session_only=True) implemented
- Signal counts per strategy on mnq_5min sample: [signals per strategy]

## Known Issues
- [any issues found during sanity checks]

## Next Agent Instructions (for Agent 1B)
- data loader is ready: use `from dataset_builder import load_data`
- signal generators are ready: use `from signal_generators import *`
- Pre-built generators are imported from `Implementation/` inside signal_generators.py
- All signal tests pass: python -m pytest ml\tests\ -v --tb=short
- Agent 1B implements feature engineering, labels, and parquet assembly
```

---

## Logic Gaps to Guard Against

1. `high == low` on any bar → wick rejection must skip, not divide by zero
2. ConnorsRSI2 SMA(200) → no signals before bar 200
3. ORB breakout bar → first valid bar is AFTER the OR closes, not the closing bar itself
4. IFVG 2/day shared limit → one counter for both ifvg + ifvg_open combined
5. `load_data(..., session_only=True)` → exactly 09:30–15:00 ET, tz-aware, inclusive on both ends
6. `load_data(..., session_only=False)` → full tz-aware data, no session filter
7. TTMSqueeze momentum → uses `linreg()` (pre-built), NOT diff/rolling_mean/slope

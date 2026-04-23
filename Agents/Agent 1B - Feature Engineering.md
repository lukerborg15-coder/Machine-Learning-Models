# Agent 1B — Feature Engineering & Assembly

**Phase:** B
**Deliverable:** Verified feature matrix parquet files for the required `MNQ` timeframes
**Estimated session length:** 1 full context session
**Prerequisite:** Agent 1A complete — if `ml\AGENT1A_STATUS.md` exists, read it before continuing

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
2. **`ml\AGENT1A_STATUS.md`** — if present, read it before proceeding
3. [[Architecture Overview]]
4. [[Testing Requirements]]
5. [[Tick Data - Delta Features]] — for synthetic delta normalization decision
6. [[Session Level Pivots]] — for pivot feature spec
7. [[NinjaScript Integration]] — for the exported `feature_order_{strategy}.json` deployment contract

---

## Starting Point

Agent 1A has already created:
- `ml\signal_generators.py` — all 7 signal functions working, with pre-built IFVG/TTM logic imported from `Implementation/`
- `ml\dataset_builder.py` — `load_data(instrument, timeframe, session_only=True)` implemented, rest stubbed
- `ml\tests\test_pipeline.py` — data + signal tests passing
- Pre-built generators verified and imported directly from `Implementation/`

**First step:** Run `python -m pytest ml\tests\ -v --tb=short` to confirm everything passes.

---

## Task List

### Task 1 — Implement OHLCV feature computation
In `dataset_builder.py`, implement `compute_ohlcv_features(df)`:
- Normalize OHLCV: `(price - rolling_mean(20)) / rolling_std(20)` — trailing window only, no lookahead
- Log volume: `np.log1p(volume)` → column name: `volume_log`
- Synthetic delta: `volume × (close - open) / (high - low)` — **guard: return 0.0 if high == low**
  - **Normalization:** Use raw synthetic delta as the feature column `synthetic_delta`. Log-volume is a separate feature (`volume_log`). Do NOT apply Option 2 or Option 3 from Tick Data spec — use raw delta + separate log volume.
- Log returns: `np.log(close / close.shift(1))` → `return_1` and `np.log(close / close.shift(5))` → `return_5`
- ATR(14) normalized by rolling mean → `atr_norm`

### Task 2 — Implement session level pivot features
In `dataset_builder.py`, implement `compute_pivot_features(df)`:

**Camarilla pivots and session levels are pre-built.** Import them from:
`C:\Users\Luker\strategyLabbrain\Implementation\camarilla_pivot_generator.py`

Do NOT rewrite the Camarilla or session level logic. Do NOT copy it into `ml/`. Import `compute_pivot_features` directly and call it from `dataset_builder.py`. Run the standalone sanity check first: `python camarilla_pivot_generator.py`

The pre-built generator handles:
- Camarilla H3/H4/S3/S4 from **prior calendar day's** OHLC only — never current day
- Running session H/L for Asia, London, Pre-market, NY AM (backward-looking only — at bar N uses max/min of bars 0..N-1)
- Previous day H/L/C, previous week H/L
- All distances: `(close - level) / ATR(14)` — ATR-normalized
- Binary flag per level: 1 if close > level, else 0 (Camarilla levels only)
- Spec: `strategyLabbrain\Strategies\Session Level Pivots.md`

**Dependency rule:** pivot distances depend on ATR already being available on the working DataFrame. Compute OHLCV features, including ATR-derived columns, before calling `compute_pivot_features`.

### Task 3 — Implement time features
In `dataset_builder.py`, implement `compute_time_features(df)`:
- `time_of_day`: minutes since 09:30, normalized 0–1 (330 min window to 15:00)
- `day_of_week`: cyclical encoding — `sin(2π × dow / 5)`, `cos(2π × dow / 5)` → produces 2 columns: `dow_sin`, `dow_cos`
  - **Do NOT use one-hot encoding** (that would produce 5 columns and break the feature order contract)
- `is_news_day`: 0/1 flag — create a static CSV of FOMC/NFP/CPI dates in `ml\data\news_dates.csv`
  - Source: Federal Reserve calendar (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) for FOMC dates
  - Source: BLS release schedule for NFP (first Friday each month) and CPI dates
  - Cover 2021-01-01 through 2026-12-31. Format: `date,event_type` (e.g., `2024-01-31,FOMC`)
  - If unable to scrape, create a minimal placeholder CSV with at least known 2024-2026 dates and flag it in AGENT1B_STATUS.md

### Task 4 — Implement label generation
In `dataset_builder.py`, implement `compute_labels(df, n_forward=5)`:
- `future_return = df['close'].shift(-n_forward) / df['close'] - 1`
- `shift(-n_forward)` is forward-looking — verify this is correct before proceeding
- Threshold = `ATR(14) × 0.5`
- Labels: 0 = Long (return > threshold), 1 = Short (return < -threshold), 2 = No Trade
- Drop the last `n_forward` rows (NaN labels)

**Critical order of operations for labeling:**
- Load raw data with `load_data(instrument, timeframe, session_only=False)`
- Compute labels (`shift(-n_forward)`) on the raw data BEFORE session filtering
- Build the working feature frame from `load_data(instrument, timeframe, session_only=True)`
- Align raw-data labels back to the session-filtered frame by timestamp
- After alignment to 09:30–15:00, rows whose label references a bar outside the session will have NaN labels
- Drop these NaN-label rows
- Do NOT compute labels after session filtering (that would make shift(-5) at 14:55 reference the next day's 9:30 bar, crossing the overnight gap)

### Task 5 — Assemble full feature matrix
In `dataset_builder.py`, implement `build_feature_matrix(instrument, timeframe)`:
- Use this exact build order:
  1. raw load with `load_data(..., session_only=False)`
  2. session-filtered working frame with `load_data(..., session_only=True)`
  3. OHLCV features including ATR on the working frame
  4. strategy signals on the working frame
  5. pivot/session features on the ATR-ready working frame
  6. time features on the working frame
  7. labels from the raw frame aligned back to the working frame
  8. warmup trimming
  9. parquet export
- Drop first `max(200, other_warmup_periods)` rows to eliminate NaN from indicator warmup
- Save to `ml\data\features_{instrument}_{timeframe}.parquet`
- Record the exact feature column order in `AGENT1B_STATUS.md`
- This recorded feature order is the source of truth until Agent 3 writes per-strategy export manifests

**Run for:** MNQ 1min, 2min, 3min, 5min

**Strategy-to-timeframe mapping for Agent 2 training:**
- ORB strategies (IB, Vol, Wick, VolAdapt): use `mnq_5min`
- IFVG and IFVG Open: use `mnq_1min` (primary) — 1min gives best FVG detection granularity
- TTMSqueeze: use `mnq_5min`
- ConnorsRSI2: use `mnq_5min`

### Task 6 — Write and run remaining tests
In `ml\tests\test_pipeline.py`, add:
- `test_synthetic_delta_no_division_by_zero()` — no NaN/Inf in synthetic delta
- `test_no_nan_in_features_after_warmup()` — after dropping warmup rows, no NaN
- `test_forward_return_uses_future_bar()` — verify shift(-n) produces forward labels
- `test_camarilla_uses_prior_day()` — H3 constant within each day
- `test_camarilla_correct_prior_day()` — H3 values match prior day's OHLC formula
- `test_no_train_test_overlap()` — train ≤ 2024-12-31, val = 2025, test = 2026
- `test_scaler_fit_only_on_train()` — when a scaler is later used, it must not be fit on val/test

**All tests must pass before writing AGENT1B_STATUS.md.**

---

## Deliverables

```
strategyLabbrain\ml\
├── dataset_builder.py            ← fully implemented (load_data + all compute functions + build_feature_matrix)
├── tests\test_pipeline.py        ← all tests written and passing (Agent 1A's + Agent 1B's)
├── data\
│   ├── features_mnq_1min.parquet
│   ├── features_mnq_2min.parquet
│   ├── features_mnq_3min.parquet
│   ├── features_mnq_5min.parquet
│   └── news_dates.csv
└── AGENT1B_STATUS.md
```

---

## AGENT1B_STATUS.md Format

```markdown
## Completed
- [list all tasks completed]

## Artifacts
- Feature column order: N_FEATURES = [exact count], columns: [list]
- Parquet files: [each file with shape (rows, cols)]
- news_dates.csv: [coverage dates, source]

## Known Issues
- [NaN patterns, data gaps, edge cases found]

## Next Agent Instructions (for Agent 2)
- Feature matrix parquet files: strategyLabbrain\ml\data\
- Feature column order: N_FEATURES = [count]
- Recommended seq_len: 30
- Train ≤ 2024-12-31 | Val = 2025 | Test = 2026
- Strategy-to-timeframe mapping:
  - ORB strategies: mnq_5min
  - IFVG/IFVG Open: mnq_1min
  - TTMSqueeze: mnq_5min
  - ConnorsRSI2: mnq_5min
- All tests pass: python -m pytest ml\tests\ -v --tb=short
```

---

## Logic Gaps to Guard Against

1. `high == low` on any bar → synthetic delta = 0.0, never divide
2. Camarilla levels → must use prior calendar day's OHLC, not today's (pre-built handles this)
3. Running session H/L → at bar N, use `max(high[:N-1])` not `max(high[:N])` (pre-built handles this)
4. Labels → `shift(-n)` is forward (correct), `shift(+n)` is backward (wrong)
5. `load_data(..., session_only=False)` → use for raw forward-label computation
6. `load_data(..., session_only=True)` → use for signal generation and the working feature frame
7. Label cross-day → compute labels on raw data, align to the session-filtered frame, then drop NaN labels
8. Pivot distances depend on ATR → compute OHLCV/ATR features before pivot features
9. Day-of-week → cyclical encoding (2 features), NOT one-hot (5 features)

# Agent 4A Audit Report

Independent audit performed cold from the source code before reading any Agent 4A status file.

## Camarilla Source Findings

- `prev_high`, `prev_low`, and `prev_close` are computed in `Implementation/camarilla_pivot_generator.py` from `daily = df.groupby(df.index.date).agg({"high": "max", "low": "min", "close": "last"})`.
- The Camarilla formula uses `daily["high"].shift(1)`, `daily["low"].shift(1)`, and `daily["close"].shift(1)`, so the source is the prior grouped calendar/trading day, not the current session.
- The parquet can contain only 09:30-15:00 rows because `ml/dataset_builder.py` loads `raw_df = load_data(..., session_only=False)`, passes it as `level_source_df=raw_df`, computes levels from that full raw source, and then reindexes those levels to the session-filtered working bars.

## Check Results

| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | Camarilla uses prior day data | PASS | Source uses daily groupby with `shift(1)` for high/low/close. Spot-check: Expected H4 `18483.86`; implied H4 from parquet `18483.86`; difference `0.0000`. |
| 2 | Rejection candle logic | PASS | Synthetic breakdown bar output: `Signal on breakdown bar: 0`; assertion passed. Long condition requires `close.gt(level)` and short condition requires `close.lt(level)`. |
| 3 | Daily cap shared across levels | PASS | Synthetic same-day outputs: Bar 0 `1`, Bar 1 `1`, Bar 2 `0`; total signals `2`, so the cap is shared across levels. |
| 4 | NY AM running high excludes current bar | PASS | Source uses `ny_am_bars["high"].expanding().max().shift(1)` and matching low shift. Exact snippet reported `nyam_high_dist` not found and listed actual columns `ny_am_high_dist`, `ny_am_low_dist`; adapted check on `2021-03-24` matched bars 1-4 expected/implied prior-bar high exactly. |
| 5 | Parquet column integrity | PASS | All required core columns are present in 1min and 5min parquets. 1min, 2min, 3min, and 5min parquets have identical column sets. |
| 6 | Full test suite | PASS | `python -m pytest ml/tests -v --tb=short` exited 0 with `105 passed, 2 skipped, 1 warning in 220.12s`. |
| 7 | Signal rate sanity | PASS | Total bars `85093`; long signals `1181`; short signals `1305`; signal rate `0.0292` (`2.92%`), within the required sanity bounds. |

## Verdict

**4A APPROVED - proceed to 4B**

All seven audit checks passed. No blocking fixes are required.

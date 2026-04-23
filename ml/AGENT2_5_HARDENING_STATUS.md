# Agent 2.5 Hardening Status

## Completed

- Added `ml/tests/test_hardening.py` with 24 targeted hardening tests.
- Added all-strategy no-lookahead coverage for ORB Vol, ORB Wick, ORB IB, ORB Volume Adaptive, ConnorsRSI2, IFVG, IFVG Open, and TTM.
- Added direct TopStep tests for daily loss limit boundaries, fixed 5 MNQ P&L math, and `$1.40` round-trip commission per contract.
- Added synthetic known-outcome `simulate_trading()` tests for long/short trade P&L, daily-loss blocking, and consistency-rule failure.
- Added an Agent 2 artifact contract test for all 8 strategy scalers, checkpoints, and eval CSVs.
- Replaced `ml/data/news_dates.csv` with a 190-row local 2021-2026 FOMC/NFP/CPI calendar.
- Fixed `compute_time_features()` news-date matching so tz-aware Eastern bar dates match local CSV calendar dates.
- Rebuilt MNQ feature parquet files for `1min`, `2min`, `3min`, and `5min` so `is_news_day` is no longer all zero.

## Calendar Sources

- Federal Reserve FOMC calendars: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
- BLS annual release schedules: `https://www.bls.gov/bls/archived_sched.htm`
- BLS current Employment Situation release schedule: `https://www.bls.gov/schedule/news_release/empsit.htm`
- BLS current CPI release schedule: `https://www.bls.gov/schedule/news_release/cpi.htm`

## Verification

- Focused hardening suite: `python -m pytest ml/tests/test_hardening.py -v --tb=short`
- Result: `24 passed, 1 warning`
- Full suite: `python -m pytest ml/tests -v --tb=short`
- Result: `51 passed, 1 warning in 500.74s (0:08:20)`
- Warning: pytest still cannot create a cache path under `.pytest_cache` due workspace `WinError 5` permissions; test behavior is unaffected.

## Feature Parquet Refresh

- `features_mnq_1min.parquet`: `(413426, 37)`, `is_news_day` sum `50269`
- `features_mnq_2min.parquet`: `(203477, 37)`, `is_news_day` sum `24747`
- `features_mnq_3min.parquet`: `(133442, 37)`, `is_news_day` sum `16240`
- `features_mnq_5min.parquet`: `(77420, 37)`, `is_news_day` sum `9434`

## Important Agent 3 Note

- Agent 2 checkpoints and scalers in `ml/artifacts/` remain the pre-hardening baseline artifacts.
- Because `is_news_day` is now real and the feature parquets were rebuilt, Agent 3 should retrain/tune from the refreshed parquets before treating any model as final exportable.
- Do not export the existing Agent 2 checkpoints as final ONNX deliverables without consciously accepting that they were trained before the real news-calendar feature was active.

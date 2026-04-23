# Agent 1B Status

## Completed Work
- Verified the current Agent 1A + 1B pipeline with `python -m pytest ml/tests/ -v --tb=short`: `14 passed, 1 warning`.
- Confirmed `build_feature_matrix()` is producing MNQ parquet feature sets for `1min`, `2min`, `3min`, and `5min`.
- Confirmed the current feature stack includes OHLCV/ATR features, strategy signal features, pivot/session features, time features, and aligned labels.

## Parquet Outputs
- `ml/data/features_mnq_1min.parquet`: `(413426, 37)`
- `ml/data/features_mnq_2min.parquet`: `(203477, 37)`
- `ml/data/features_mnq_3min.parquet`: `(133442, 37)`
- `ml/data/features_mnq_5min.parquet`: `(77420, 37)`

## Feature Column Order
- `open_norm`
- `high_norm`
- `low_norm`
- `close_norm`
- `volume_log`
- `synthetic_delta`
- `return_1`
- `return_5`
- `atr_norm`
- `orb_vol_signal`
- `orb_wick_signal`
- `orb_ib_signal`
- `ifvg_signal`
- `ifvg_open_signal`
- `ttm_signal`
- `connors_signal`
- `orb_va_signal`
- `h3_dist`
- `h4_dist`
- `s3_dist`
- `s4_dist`
- `h3_above`
- `h4_above`
- `s3_above`
- `s4_above`
- `ny_am_high_dist`
- `ny_am_low_dist`
- `prev_day_high_dist`
- `prev_day_low_dist`
- `prev_week_high_dist`
- `prev_week_low_dist`
- `time_of_day`
- `dow_sin`
- `dow_cos`
- `is_news_day`

## Known Issues / Blockers
- Accepted fallback: no fabricated Asia/London/premarket features are included because the available raw data does not support those windows safely.
- Post-Agent 2.5 update: `ml/data/news_dates.csv` now contains a curated 2021-2026 FOMC/NFP/CPI calendar, and rebuilt feature parquets have nonzero `is_news_day` flags.
- Accepted fallback: labels are masked to remain within the `09:30-15:00 ET` training session.
- Full `build_feature_matrix()` runs are slowest on `mnq_1min`.
- Pytest reports one non-blocking `.pytest_cache` permission warning on this machine.

## Next Instructions for Agent 2
- Treat the feature column order above as the source of truth until per-strategy export manifests exist.
- Use `features_mnq_1min.parquet` for `ifvg` and `ifvg_open`.
- Use `features_mnq_5min.parquet` for `orb_ib`, `orb_vol`, `orb_wick`, `orb_va`, `ttm`, and `connors`.
- Preserve temporal splits and fit scalers on training data only.
- Keep strategy artifacts isolated per strategy name when training in parallel.

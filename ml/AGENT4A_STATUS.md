## Completed
- Read the Agent 4A spec and every required context file before code changes.
- Ran baseline test suite before changes: `97 passed, 2 skipped, 1 warning`.
- Inspected all existing feature parquets before rebuild and confirmed `session_pivot_signal` was missing.
- Wired pivot computation in `ml/dataset_builder.py` to use the full raw dataframe before session filtering, so prior-day Camarilla levels use the 16:00 prior-day close/high/low when available.
- Added additive `camarilla_h3_dist`, `camarilla_h4_dist`, `camarilla_s3_dist`, and `camarilla_s4_dist` columns while preserving the first 35 feature columns in their existing order.
- Added `session_pivot_signal()` to `ml/signal_generators.py` with rejection-only logic, ATR warmup guard, level priority, prior-day-close context, and shared 2-signal daily cap.
- Wired `session_pivot_signal` into `ml/dataset_builder.py` and generated session-pivot triple-barrier metadata columns.
- Added 7 Agent 4A tests in `ml/tests/test_agent4a.py`.
- Rebuilt all four parquets: 1min, 2min, 3min, 5min.
- Verified all rebuilt parquets have non-zero Camarilla distance columns, long and short session pivot signals, and preserved first-35 feature order.
- Ran full post-change test suite: `104 passed, 2 skipped, 1 warning`.

## Parquet State Before Rebuild
```text
=== features_mnq_1min.parquet ===
Shape: (419866, 86)
Columns: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_log', 'synthetic_delta', 'return_1', 'return_5', 'atr_norm', 'orb_vol_signal', 'orb_wick_signal', 'orb_ib_signal', 'ifvg_signal', 'ifvg_open_signal', 'ttm_signal', 'connors_signal', 'orb_va_signal', 'h3_dist', 'h4_dist', 's3_dist', 's4_dist', 'h3_above', 'h4_above', 's3_above', 's4_above', 'ny_am_high_dist', 'ny_am_low_dist', 'prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist', 'time_of_day', 'dow_sin', 'dow_cos', 'is_news_day', 'future_return', 'label', 'atr_14', 'label_ifvg', 'exit_bar_ifvg', 'exit_time_ifvg', 'exit_price_ifvg', 'r_multiple_ifvg', 'barrier_hit_ifvg', 'label_ifvg_open', 'exit_bar_ifvg_open', 'exit_time_ifvg_open', 'exit_price_ifvg_open', 'r_multiple_ifvg_open', 'barrier_hit_ifvg_open', 'label_orb_ib', 'exit_bar_orb_ib', 'exit_time_orb_ib', 'exit_price_orb_ib', 'r_multiple_orb_ib', 'barrier_hit_orb_ib', 'label_orb_vol', 'exit_bar_orb_vol', 'exit_time_orb_vol', 'exit_price_orb_vol', 'r_multiple_orb_vol', 'barrier_hit_orb_vol', 'label_orb_wick', 'exit_bar_orb_wick', 'exit_time_orb_wick', 'exit_price_orb_wick', 'r_multiple_orb_wick', 'barrier_hit_orb_wick', 'label_orb_va', 'exit_bar_orb_va', 'exit_time_orb_va', 'exit_price_orb_va', 'r_multiple_orb_va', 'barrier_hit_orb_va', 'label_ttm', 'exit_bar_ttm', 'exit_time_ttm', 'exit_price_ttm', 'r_multiple_ttm', 'barrier_hit_ttm', 'label_connors', 'exit_bar_connors', 'exit_time_connors', 'exit_price_connors', 'r_multiple_connors', 'barrier_hit_connors']
Pivot columns found (4): ['prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist']
       prev_day_high_dist  ...  prev_week_low_dist
count       419866.000000  ...       419866.000000
mean            -6.826219  ...           47.519306
std             19.876748  ...           57.723713
min           -154.000000  ...          -83.340164
25%            -17.671062  ...            9.257422
50%             -6.819249  ...           36.038262
75%              3.137931  ...           74.611099
max            118.023256  ...         1022.373333

[8 rows x 4 columns]
All-zero pivot cols: []
All-NaN pivot cols:  []
session_pivot_signal: MISSING

=== features_mnq_2min.parquet ===
Shape: (209917, 86)
Columns: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_log', 'synthetic_delta', 'return_1', 'return_5', 'atr_norm', 'orb_vol_signal', 'orb_wick_signal', 'orb_ib_signal', 'ifvg_signal', 'ifvg_open_signal', 'ttm_signal', 'connors_signal', 'orb_va_signal', 'h3_dist', 'h4_dist', 's3_dist', 's4_dist', 'h3_above', 'h4_above', 's3_above', 's4_above', 'ny_am_high_dist', 'ny_am_low_dist', 'prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist', 'time_of_day', 'dow_sin', 'dow_cos', 'is_news_day', 'future_return', 'label', 'atr_14', 'label_ifvg', 'exit_bar_ifvg', 'exit_time_ifvg', 'exit_price_ifvg', 'r_multiple_ifvg', 'barrier_hit_ifvg', 'label_ifvg_open', 'exit_bar_ifvg_open', 'exit_time_ifvg_open', 'exit_price_ifvg_open', 'r_multiple_ifvg_open', 'barrier_hit_ifvg_open', 'label_orb_ib', 'exit_bar_orb_ib', 'exit_time_orb_ib', 'exit_price_orb_ib', 'r_multiple_orb_ib', 'barrier_hit_orb_ib', 'label_orb_vol', 'exit_bar_orb_vol', 'exit_time_orb_vol', 'exit_price_orb_vol', 'r_multiple_orb_vol', 'barrier_hit_orb_vol', 'label_orb_wick', 'exit_bar_orb_wick', 'exit_time_orb_wick', 'exit_price_orb_wick', 'r_multiple_orb_wick', 'barrier_hit_orb_wick', 'label_orb_va', 'exit_bar_orb_va', 'exit_time_orb_va', 'exit_price_orb_va', 'r_multiple_orb_va', 'barrier_hit_orb_va', 'label_ttm', 'exit_bar_ttm', 'exit_time_ttm', 'exit_price_ttm', 'r_multiple_ttm', 'barrier_hit_ttm', 'label_connors', 'exit_bar_connors', 'exit_time_connors', 'exit_price_connors', 'r_multiple_connors', 'barrier_hit_connors']
Pivot columns found (4): ['prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist']
       prev_day_high_dist  ...  prev_week_low_dist
count       209917.000000  ...       209917.000000
mean            -4.763741  ...           32.932421
std             13.795324  ...           39.768717
min            -89.000000  ...          -57.457103
25%            -12.253829  ...            6.443836
50%             -4.789474  ...           25.125000
75%              2.177778  ...           51.947967
max             88.034934  ...          661.620690

[8 rows x 4 columns]
All-zero pivot cols: []
All-NaN pivot cols:  []
session_pivot_signal: MISSING

=== features_mnq_3min.parquet ===
Shape: (139882, 86)
Columns: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_log', 'synthetic_delta', 'return_1', 'return_5', 'atr_norm', 'orb_vol_signal', 'orb_wick_signal', 'orb_ib_signal', 'ifvg_signal', 'ifvg_open_signal', 'ttm_signal', 'connors_signal', 'orb_va_signal', 'h3_dist', 'h4_dist', 's3_dist', 's4_dist', 'h3_above', 'h4_above', 's3_above', 's4_above', 'ny_am_high_dist', 'ny_am_low_dist', 'prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist', 'time_of_day', 'dow_sin', 'dow_cos', 'is_news_day', 'future_return', 'label', 'atr_14', 'label_ifvg', 'exit_bar_ifvg', 'exit_time_ifvg', 'exit_price_ifvg', 'r_multiple_ifvg', 'barrier_hit_ifvg', 'label_ifvg_open', 'exit_bar_ifvg_open', 'exit_time_ifvg_open', 'exit_price_ifvg_open', 'r_multiple_ifvg_open', 'barrier_hit_ifvg_open', 'label_orb_ib', 'exit_bar_orb_ib', 'exit_time_orb_ib', 'exit_price_orb_ib', 'r_multiple_orb_ib', 'barrier_hit_orb_ib', 'label_orb_vol', 'exit_bar_orb_vol', 'exit_time_orb_vol', 'exit_price_orb_vol', 'r_multiple_orb_vol', 'barrier_hit_orb_vol', 'label_orb_wick', 'exit_bar_orb_wick', 'exit_time_orb_wick', 'exit_price_orb_wick', 'r_multiple_orb_wick', 'barrier_hit_orb_wick', 'label_orb_va', 'exit_bar_orb_va', 'exit_time_orb_va', 'exit_price_orb_va', 'r_multiple_orb_va', 'barrier_hit_orb_va', 'label_ttm', 'exit_bar_ttm', 'exit_time_ttm', 'exit_price_ttm', 'r_multiple_ttm', 'barrier_hit_ttm', 'label_connors', 'exit_bar_connors', 'exit_time_connors', 'exit_price_connors', 'r_multiple_connors', 'barrier_hit_connors']
Pivot columns found (4): ['prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist']
       prev_day_high_dist  ...  prev_week_low_dist
count       139882.000000  ...       139882.000000
mean            -3.861297  ...           26.491789
std             11.083255  ...           31.788931
min            -73.128492  ...          -46.954796
25%             -9.832519  ...            5.210212
50%             -3.918190  ...           20.317853
75%              1.761340  ...           41.850286
max             68.825503  ...          471.533742

[8 rows x 4 columns]
All-zero pivot cols: []
All-NaN pivot cols:  []
session_pivot_signal: MISSING

=== features_mnq_5min.parquet ===
Shape: (83851, 86)
Columns: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_log', 'synthetic_delta', 'return_1', 'return_5', 'atr_norm', 'orb_vol_signal', 'orb_wick_signal', 'orb_ib_signal', 'ifvg_signal', 'ifvg_open_signal', 'ttm_signal', 'connors_signal', 'orb_va_signal', 'h3_dist', 'h4_dist', 's3_dist', 's4_dist', 'h3_above', 'h4_above', 's3_above', 's4_above', 'ny_am_high_dist', 'ny_am_low_dist', 'prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist', 'time_of_day', 'dow_sin', 'dow_cos', 'is_news_day', 'future_return', 'label', 'atr_14', 'label_ifvg', 'exit_bar_ifvg', 'exit_time_ifvg', 'exit_price_ifvg', 'r_multiple_ifvg', 'barrier_hit_ifvg', 'label_ifvg_open', 'exit_bar_ifvg_open', 'exit_time_ifvg_open', 'exit_price_ifvg_open', 'r_multiple_ifvg_open', 'barrier_hit_ifvg_open', 'label_orb_ib', 'exit_bar_orb_ib', 'exit_time_orb_ib', 'exit_price_orb_ib', 'r_multiple_orb_ib', 'barrier_hit_orb_ib', 'label_orb_vol', 'exit_bar_orb_vol', 'exit_time_orb_vol', 'exit_price_orb_vol', 'r_multiple_orb_vol', 'barrier_hit_orb_vol', 'label_orb_wick', 'exit_bar_orb_wick', 'exit_time_orb_wick', 'exit_price_orb_wick', 'r_multiple_orb_wick', 'barrier_hit_orb_wick', 'label_orb_va', 'exit_bar_orb_va', 'exit_time_orb_va', 'exit_price_orb_va', 'r_multiple_orb_va', 'barrier_hit_orb_va', 'label_ttm', 'exit_bar_ttm', 'exit_time_ttm', 'exit_price_ttm', 'r_multiple_ttm', 'barrier_hit_ttm', 'label_connors', 'exit_bar_connors', 'exit_time_connors', 'exit_price_connors', 'r_multiple_connors', 'barrier_hit_connors']
Pivot columns found (4): ['prev_day_high_dist', 'prev_day_low_dist', 'prev_week_high_dist', 'prev_week_low_dist']
       prev_day_high_dist  ...  prev_week_low_dist
count        83851.000000  ...        83851.000000
mean            -2.960875  ...           19.990465
std              8.337051  ...           23.746857
min            -52.857143  ...          -35.577670
25%             -7.517398  ...            3.995213
50%             -3.017728  ...           15.476411
75%              1.337886  ...           31.770084
max             45.762115  ...          340.778846

[8 rows x 4 columns]
All-zero pivot cols: []
All-NaN pivot cols:  []
session_pivot_signal: MISSING
```

## Parquet State After Rebuild
```text
=== features_mnq_1min.parquet ===
Shape: (421285, 97)
  camarilla_h3_dist: max=125.6012, nan%=0.0% OK
  camarilla_h4_dist: max=121.9600, nan%=0.0% OK
  camarilla_s3_dist: max=186.8160, nan%=0.0% OK
  camarilla_s4_dist: max=258.2720, nan%=0.0% OK
  session_pivot_signal: {0: 418974, -1: 1278, 1: 1033} OK
  All original feature columns present and first-35 order preserved OK

=== features_mnq_2min.parquet ===
Shape: (211204, 97)
  camarilla_h3_dist: max=93.7266, nan%=0.0% OK
  camarilla_h4_dist: max=84.3790, nan%=0.0% OK
  camarilla_s3_dist: max=112.4218, nan%=0.0% OK
  camarilla_s4_dist: max=142.8000, nan%=0.0% OK
  session_pivot_signal: {0: 208907, -1: 1260, 1: 1037} OK
  All original feature columns present and first-35 order preserved OK

=== features_mnq_3min.parquet ===
Shape: (141169, 97)
  camarilla_h3_dist: max=73.1993, nan%=0.0% OK
  camarilla_h4_dist: max=66.0161, nan%=0.0% OK
  camarilla_s3_dist: max=87.5658, nan%=0.0% OK
  camarilla_s4_dist: max=94.7490, nan%=0.0% OK
  session_pivot_signal: {0: 138883, -1: 1255, 1: 1031} OK
  All original feature columns present and first-35 order preserved OK

=== features_mnq_5min.parquet ===
Shape: (85137, 97)
  camarilla_h3_dist: max=48.6330, nan%=0.0% OK
  camarilla_h4_dist: max=43.9181, nan%=0.0% OK
  camarilla_s3_dist: max=58.0630, nan%=0.0% OK
  camarilla_s4_dist: max=63.9287, nan%=0.0% OK
  session_pivot_signal: {0: 82871, -1: 1229, 1: 1037} OK
  All original feature columns present and first-35 order preserved OK
```

## Session Pivot Signal Distribution (5min parquet)
- Long signals: 1037
- Short signals: 1229
- Signal rate: 2.6616% (2266 signals in 85137 bars)

## Camarilla Values Sample (first 5 signal bars)
```text
                timestamp  signal  prev_date  prev_high  prev_low  prev_close          H3         H4          S3         S4    close
2021-03-24T09:30:00-04:00      -1 2021-03-23    13172.0  12978.75    13019.75 13072.89375 13126.0375 12966.60625 12913.4625 13052.00
2021-03-24T09:45:00-04:00       1 2021-03-23    13172.0  12978.75    13019.75 13072.89375 13126.0375 12966.60625 12913.4625 13019.00
2021-03-25T09:35:00-04:00       1 2021-03-24    13072.5  12784.25    12789.00 12868.26875 12947.5375 12709.73125 12630.4625 12722.75
2021-03-25T09:50:00-04:00       1 2021-03-24    13072.5  12784.25    12789.00 12868.26875 12947.5375 12709.73125 12630.4625 12730.50
2021-03-26T09:40:00-04:00      -1 2021-03-25    12837.0  12609.75    12781.25 12843.74375 12906.2375 12718.75625 12656.2625 12803.00
```

## Test Suite Results
- Before: 97 passed, 0 failed, 2 skipped, 1 warning.
- After: 104 passed, 0 failed, 2 skipped, 1 warning.
- New tests added: 7.

## Feature Column Count
- Original: 35 feature columns.
- After rebuild: 40 feature columns (35 original + 4 additive Camarilla distance columns + 1 additive `session_pivot_signal` column).
- Total parquet columns after rebuild: 97 (feature columns + targets + ATR + meta-label columns, including session-pivot meta-label outputs).
- Column order preserved: yes. The first 35 columns match the original feature contract exactly.

## Known Issues
- The raw MNQ CSVs contain 09:30-16:00 ET bars only. They provide prior-day 16:00 OHLC context outside the 09:30-15:00 training window, but they do not contain Asia, London, or true premarket bars. Because of that data limitation, Asia/London/Premarket distance columns remain omitted rather than populated from the 09:30 bar.
- `python ml/dataset_builder.py --rebuild` printed all four output paths but the command wrapper reported a timeout after the paths were emitted. Direct parquet verification passed for all four rebuilt files.

## Next Agent
- Agent 4A-Audit reads this file and inspects the parquets independently.
- Agent 4B starts after 4A-Audit passes.

## Follow-Up Fix: Completed Session Levels
- Updated `ml/dataset_builder.py` so Asia, London, and premarket highs/lows are computed from explicit full-source timestamp windows before joining onto RTH-filtered bars.
- Asia window: prior trading date 20:00 ET through current date before 02:00 ET.
- London window: current date 02:00 ET through before 07:00 ET.
- Premarket window: current date 07:00 ET through before 09:30 ET; the 09:30 RTH bar is excluded.
- Added `test_completed_session_levels_use_full_source_before_rth_filter` to prove overnight source bars populate session distances while the RTH 09:30 bar is not used as premarket.
- Rebuilt all four parquets with `python ml/dataset_builder.py --rebuild`.
- Post-fix test suite: `105 passed, 2 skipped, 1 warning`.
- Note: the local CSV files currently readable at `data/mnq_*_databento.csv` contain 09:30-16:00 ET rows only, so the rebuilt parquets in this workspace still omit Asia/London/premarket columns because there are no overnight source rows to aggregate. The fixed code path will populate them when the raw input files include those 24h bars.

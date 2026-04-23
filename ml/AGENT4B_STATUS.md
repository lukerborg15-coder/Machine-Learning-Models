# Agent 4B Status

**Status:** CODE COMPLETE — NOT EXECUTED

## Completed
- Verified `ml/AGENT4A_AUDIT.md` says **4A APPROVED - proceed to 4B**.
- Added `session_pivot_break_signal()` for Camarilla H4/S4 continuation breaks.
- Wired `session_pivot_break_signal` into `dataset_builder.py`.
- Removed ORB VA from the active training strategy universe and replaced it with Camarilla continuation.
- Defined 4 grouped model configs in `train.py`.
- Added all-bars grouped label assignment with conflict bars forced to NoTrade.
- Removed signal-gated training windows from `build_window_batch()`.
- Updated grouped training, fold loop, checkpoint/scaler/eval naming, and dry-run save suppression.
- Updated HPO call sites to use grouped model jobs and combined group signal direction.
- Updated `TopStepRiskManager` to `max_contracts=50` with confidence-based `position_size()`.
- Updated evaluation and funded simulation to size trades from stop distance and confidence.
- Added `ml/tests/test_agent4b.py` with 12 Agent 4B tests.
- Updated existing tests that asserted signal-gated windows or fixed 5-contract trade math.

## Not Executed
- Parquet rebuild was **not executed** because it requires `python ml/dataset_builder.py --rebuild`.
- Pytest was **not executed** by instruction.
- Training and dry-run training were **not executed** by instruction.
- No Python commands were executed.

## Model Groups
| Model | Strategies | Signal Cols | Parquet |
|---|---|---|---|
| model_1 | ifvg, connors | ifvg_signal, connors_signal | ml/data/features_mnq_5min.parquet |
| model_2 | ifvg_open, ttm | ifvg_open_signal, ttm_signal | ml/data/features_mnq_5min.parquet |
| model_3 | orb_vol, session_pivot, session_pivot_break | orb_vol_signal, session_pivot_signal, session_pivot_break_signal | ml/data/features_mnq_5min.parquet |
| model_4 | orb_ib, orb_wick | orb_ib_signal, orb_wick_signal | ml/data/features_mnq_5min.parquet |

## Training Row Counts (per model, per fold)
Not measured. Measuring requires loading parquet files with Python, which was explicitly prohibited.

| Model | Fold | Rows | Expected | % of Expected |
|---|---|---:|---:|---:|
| model_1 | fold_1 | NOT EXECUTED | >40,000 | NOT EXECUTED |
| model_1 | fold_2 | NOT EXECUTED | >50,000 | NOT EXECUTED |
| model_1 | fold_3 | NOT EXECUTED | >60,000 | NOT EXECUTED |
| model_1 | fold_4 | NOT EXECUTED | >70,000 | NOT EXECUTED |
| model_1 | fold_5 | NOT EXECUTED | >75,000 | NOT EXECUTED |
| model_2 | fold_1 | NOT EXECUTED | >40,000 | NOT EXECUTED |
| model_2 | fold_2 | NOT EXECUTED | >50,000 | NOT EXECUTED |
| model_2 | fold_3 | NOT EXECUTED | >60,000 | NOT EXECUTED |
| model_2 | fold_4 | NOT EXECUTED | >70,000 | NOT EXECUTED |
| model_2 | fold_5 | NOT EXECUTED | >75,000 | NOT EXECUTED |
| model_3 | fold_1 | NOT EXECUTED | >40,000 | NOT EXECUTED |
| model_3 | fold_2 | NOT EXECUTED | >50,000 | NOT EXECUTED |
| model_3 | fold_3 | NOT EXECUTED | >60,000 | NOT EXECUTED |
| model_3 | fold_4 | NOT EXECUTED | >70,000 | NOT EXECUTED |
| model_3 | fold_5 | NOT EXECUTED | >75,000 | NOT EXECUTED |
| model_4 | fold_1 | NOT EXECUTED | >40,000 | NOT EXECUTED |
| model_4 | fold_2 | NOT EXECUTED | >50,000 | NOT EXECUTED |
| model_4 | fold_3 | NOT EXECUTED | >60,000 | NOT EXECUTED |
| model_4 | fold_4 | NOT EXECUTED | >70,000 | NOT EXECUTED |
| model_4 | fold_5 | NOT EXECUTED | >75,000 | NOT EXECUTED |

## Label Distribution (per model)
Not measured. Label distribution checks require Python/parquet execution.

| Model | Long% | Short% | NoTrade% |
|---|---:|---:|---:|
| model_1 | NOT EXECUTED | NOT EXECUTED | NOT EXECUTED |
| model_2 | NOT EXECUTED | NOT EXECUTED | NOT EXECUTED |
| model_3 | NOT EXECUTED | NOT EXECUTED | NOT EXECUTED |
| model_4 | NOT EXECUTED | NOT EXECUTED | NOT EXECUTED |

## Position Sizing
- max_contracts: 50
- point_value: 2.0
- target_risk: $500 per trade
- zero or negative stop distance raises `ValueError`

| Confidence | Scale |
|---:|---:|
| >= 0.80 | 1.00 |
| >= 0.70 | 0.80 |
| >= 0.65 | 0.60 |
| < 0.65 | 0.40 |

## Fold Configuration
- 5 folds confirmed in `WALK_FORWARD_FOLDS`.
- Date windows match the Agent 4B spec.
- Date-only fold end values are handled as end-of-day by `dataset_builder._slice_fold_range()`.

## Test Suite
- Before: 105 passed, 2 skipped from `AGENT4A_AUDIT.md`.
- After: NOT EXECUTED.
- New tests: 12 in `ml/tests/test_agent4b.py`.

## Known Issues
- `ml/data/*.parquet` were not rebuilt in this session because rebuilding requires Python execution.
- `session_pivot_break_signal` column will not be present in parquets until the human runs `python ml/dataset_builder.py --rebuild`.
- Row counts and label distributions are intentionally unmeasured until the human executes the rebuild/dry-run checks.
- No checkpoints, scalers, eval CSVs, funded outputs, or ONNX exports were generated.

## Next Agent
- Agent 4B-Audit reviews this.
- After approval: 3C -> 3C-Audit -> 3D -> 3D-Audit -> 3E -> 3E-Audit.

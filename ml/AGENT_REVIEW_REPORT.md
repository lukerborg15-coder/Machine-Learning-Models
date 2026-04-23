# Agent 1 and Agent 2 Review Report

Review date: 2026-04-10
Reviewer scope: Agent 1A, Agent 1B, Agent 2, and the current pytest suite
Review mode: read-only first pass

## Hardening Update

The targeted Agent 2.5 hardening pass addressed the major test gaps in this first-pass report:

- Added all-strategy no-lookahead tests.
- Added direct TopStep daily-loss and commission/P&L tests.
- Added synthetic known-outcome evaluation tests.
- Added an Agent 2 artifact contract test.
- Replaced the placeholder news calendar with a real 2021-2026 FOMC/NFP/CPI calendar and fixed `is_news_day` date matching.
- Final verification after hardening: `51 passed, 1 warning` from `python -m pytest ml/tests -v --tb=short`.

The remaining strategic note is that Agent 2 checkpoints/scalers are still pre-hardening baseline artifacts. Agent 3 should retrain/tune from refreshed feature parquets before final export.

## Findings

### Test gap - no-lookahead signal coverage is too narrow

- Source of truth: `Implementation\Testing Requirements.md` and `Agents\Agent 1A - Signal Generators.md`
- Current behavior: `ml\tests\test_pipeline.py::test_strategy_signal_no_lookahead` mutates one row and verifies earlier rows for `orb_wick_rejection` only.
- Impact: this is a useful causal smoke test, but it does not prove all ORB generators, ConnorsRSI2, IFVG, IFVG Open, and TTM are no-lookahead safe. A future regression in another signal generator could pass the suite.
- Recommended fix: add a parameterized no-lookahead test across every signal generator, ideally using both mutation-before/after checks and full-vs-truncated dataset checks.

### Test gap - TopStep daily loss and trade-cost rules are not directly tested

- Source of truth: `Reference\Prop Firm Rules.md` and `Implementation\Testing Requirements.md`
- Current behavior: tests cover EOD trailing drawdown from peak and the 40% consistency rule, but do not directly test `check_intraday()` daily loss behavior or `simulate_trade()` commission handling.
- Impact: Agent 2's TopStep simulator could regress daily-loss blocking or transaction-cost math without a focused unit test catching it.
- Recommended fix: add tests for DLL breach at exactly `$1,000`, no breach below `$1,000`, fixed 5 MNQ contract P&L, and `$1.40` round-trip cost per contract.

### Test gap - evaluation realism is weak for low-trade strategies

- Source of truth: `Implementation\ML Operators Guide.md`, Phase E
- Current behavior: Agent 2 eval CSVs exist for all 8 strategies, but several ORB strategies have summary `trade_count=0.0` and `test_sharpe=NaN`.
- Impact: the pipeline reports artifacts correctly, but a zero-trade evaluation is not a strong usability signal. It can hide whether the trading simulation behaves correctly under real trade flow.
- Recommended fix: add a tiny synthetic evaluation smoke test that forces one long winner, one short loser, one DLL-blocked day, and one consistency failure. Keep the real data eval separate.

### Test gap - artifact contract integrity is only partially covered

- Source of truth: `Home.md`, `Implementation\ML Operators Guide.md`, and `Agents\Agent 2 - Model and Training.md`
- Current behavior: `ml\tests\test_parallel_runner.py` checks expected job paths and duplicate artifact stems, while this review confirmed 8 scalers, 8 checkpoints, and 8 eval CSVs exist in `ml\artifacts`.
- Impact: the suite does not currently fail if a completed baseline is missing one expected artifact or if an eval CSV is missing a required metric column.
- Recommended fix: add an artifact-contract test that verifies all 8 `scaler_*.pkl`, `best_model_*.pt`, and `eval_*.csv` files exist and that each eval CSV has fold rows plus the expected metric columns.

### Accepted deviation - extended-hours pivot features are intentionally omitted

- Source of truth: `Agents\Agent 1B - Feature Engineering.md` requested Asia, London, and premarket distances; `ml\AGENT1B_STATUS.md` documents the fallback.
- Current behavior: `ml\dataset_builder.py` drops Asia/London/premarket distance columns when the loaded data does not safely contain those sessions. `ml\tests\test_pipeline.py::test_session_feature_fallback_excludes_unavailable_extended_hours_columns` locks this fallback.
- Impact: this is an intentional safety tradeoff, not an implementation defect. The resulting feature contract is 35 inputs instead of the longer future NinjaScript example.
- Recommended fix: keep this documented until true extended-hours data is added. When extended-hours data is available, reintroduce these features and update the feature-order contract everywhere.

### Accepted deviation - `is_news_day` is inactive because the macro calendar is a placeholder

- Source of truth: `Agents\Agent 1B - Feature Engineering.md` requested a 2021-2026 FOMC/NFP/CPI calendar; `ml\AGENT1B_STATUS.md` and `Implementation\ML Cleanup Backlog.md` document the placeholder.
- Current behavior: `ml\data\news_dates.csv` contains a placeholder row with no date. All current feature parquet files have `is_news_day` sum equal to `0`.
- Impact: no live leakage or crash risk, but the feature is currently non-informative.
- Recommended fix: replace the placeholder with a curated local 2021-2026 macro calendar before treating news-day behavior as modeled.

## Confirmed Compliance

- Agent 1 data loading: `load_data()` converts to `America/New_York`, sorts by timestamp, and applies `09:30-15:00` filtering only for `session_only=True`.
- Agent 1 label direction: `compute_labels()` uses `shift(-n_forward)`, aligns raw labels back to the session-filtered frame, and masks labels that would cross the same-day training session boundary.
- Agent 1 feature contract: current parquet files have 37 columns total: 35 model input features plus `future_return` and `label`.
- Agent 1 parquets: `features_mnq_1min.parquet`, `features_mnq_2min.parquet`, `features_mnq_3min.parquet`, and `features_mnq_5min.parquet` exist.
- Agent 2 model architecture: `TradingCNN` uses `nn.ConstantPad1d((kernel - 1, 0), 0.0)` before `nn.Conv1d(..., padding=0)` and no `padding='same'` usage was found in `ml\model.py`.
- Agent 2 walk-forward: `WALK_FORWARD_FOLDS` matches the locked two-fold anchored baseline: train through 2023 / validate 2024 / test 2025, then train through 2024 / validate 2025 / test through 2026-03-18.
- Agent 2 scaling: `_train_one_fold()` fits `SimpleStandardScaler` on the fold train frame and applies it to validation/test frames.
- Agent 2 gating: `build_window_batch()` keeps only windows whose endpoint raw signal is non-zero, and the test suite verifies raw setup-row gating for `orb_wick`.
- Agent 2 artifacts: `ml\artifacts` contains 8 scalers, 8 checkpoints, and 8 eval CSVs.
- Agent 2 TopStep core rules: `TopStepRiskManager` implements fixed 5 MNQ contracts, EOD trailing drawdown from max EOD equity, daily loss limit checks, and the 40% consistency rule.

## Test Viability

- Strong checks: timezone/session filtering, duplicate timestamps, synthetic-delta zero-range guard, post-warmup NaN checks, forward-label direction, Camarilla prior-day logic, temporal split ordering, causal convolution padding, class-weight non-uniformity, EOD trailing drawdown, and consistency rule.
- Medium checks: scaler train-only checks are directionally useful and the production training path does fit on the train frame, but the tests mostly compare helper outputs rather than loading saved scaler artifacts per strategy.
- Weak checks: signal no-lookahead currently covers one ORB strategy only; setup-row gating is tested with one strategy only; artifact creation is covered by orchestration path strings but not by a completed-artifact invariant test.
- Missing checks: TopStep daily loss boundary, trade commission math, synthetic trading eval with forced trade outcomes, all-strategy no-lookahead, `news_dates.csv` real coverage, and eval CSV metric schema.

## Verification Run

Command run:

```powershell
python -m pytest ml/tests -v --tb=short
```

Observed result:

- Pytest printed `27 passed, 1 warning in 1025.06s (0:17:05)`.
- Warning: pytest cache path creation failed with `WinError 5` under `strategyLabbrain\.pytest_cache`.
- Tool wrapper returned exit code `124` because the command-level timeout fired immediately after pytest printed the pass summary. If a zero shell exit code is required for release bookkeeping, rerun the same command with a timeout above 20 minutes.

Additional checks:

- `ml\artifacts`: 8 `best_model_*.pt`, 8 `scaler_*.pkl`, and 8 `eval_*.csv` files found.
- Feature parquets: all four required MNQ feature parquets found with 35 input features and 2 target columns.
- `is_news_day`: sum is `0` in all current feature parquets, matching the documented placeholder-calendar fallback.

## Next Actions

1. Add parameterized no-lookahead tests across all signal generators.
2. Add TopStep daily-loss, commission, and synthetic evaluation smoke tests.
3. Add a completed-artifact contract test for all 8 baseline strategies.
4. Replace `ml\data\news_dates.csv` with a curated 2021-2026 FOMC/NFP/CPI calendar.
5. Decide a formal policy for `NaN` Sharpe and zero-trade strategy eval rows before Agent 3 uses validation Sharpe for selection.

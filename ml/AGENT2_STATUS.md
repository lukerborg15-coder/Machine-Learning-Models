## Post-Agent 2.5 Hardening Note

- This file is the historical Agent 2 baseline status from 2026-04-09.
- Agent 2.5 hardening on 2026-04-10 replaced the placeholder `news_dates.csv`, rebuilt feature parquets with active `is_news_day` flags, and raised the verified suite to `51 passed, 1 warning`.
- The checkpoints and scalers listed here remain pre-hardening baseline references. Retrain/tune from the refreshed feature parquets before final ONNX export.

## Completed
- Implemented `TradingCNN` in `ml/model.py` with manual left-only causal padding and 3-class logits.
- Implemented anchored walk-forward training in `ml/train.py` while preserving the existing job-orchestration API used by `ml/tests/test_parallel_runner.py`.
- Implemented the canonical `TopStepRiskManager` in `ml/topstep_risk.py`.
- Implemented `ml/evaluate.py` for classification metrics plus TopStep-style trade simulation on setup rows only.
- Added Agent 2 tests to `ml/tests/test_pipeline.py`.
- Installed PyTorch in the active Python environment so the model path and training loop could run locally.
- Re-ran the full test suite: `python -m pytest ml/tests/ -v --tb=short`
- Final test result: `27 passed, 1 warning`
- Trained all 8 baseline strategy models and wrote scaler/checkpoint/evaluation artifacts to `ml/artifacts/`.

## Model Config Used
- n_features per strategy: `ifvg=35`, `ifvg_open=35`, `orb_ib=35`, `orb_vol=35`, `orb_wick=35`, `orb_va=35`, `ttm=35`, `connors=35`
- seq_len: `30`
- n_filters / n_layers / dropout: `64 / 2 / 0.3`
- kernel_size: `3`
- batch_size: `128`
- learning_rate: `3e-4`
- early stopping patience: `10`
- max_epochs: `25`
- device: `cpu`

## Results per Strategy
| Strategy | Val F1 | Val Sharpe | Test F1 | Test Sharpe | Combine Pass Rate |
|---|---|---|---|---|---|
| ifvg | 0.2624 | n/a | 0.3115 | 0.0316 | 0.50 |
| ifvg_open | 0.2575 | n/a | 0.2548 | -2.8779 | 0.00 |
| orb_ib | 0.2946 | n/a | 0.2801 | n/a | 0.00 |
| orb_vol | 0.3455 | n/a | 0.3494 | n/a | 0.00 |
| orb_wick | 0.2895 | n/a | 0.3260 | n/a | 0.00 |
| orb_va | 0.3659 | n/a | 0.3462 | n/a | 0.00 |
| ttm | 0.3406 | n/a | 0.2992 | 0.3073 | 0.00 |
| connors | 0.2601 | n/a | 0.3668 | -1.4328 | 0.00 |

## Known Issues
- Several strategies produced `NaN` Sharpe in the summary row because the post-gating trade sample was too small or too flat for a stable daily-return standard deviation.
- Baseline ROC-AUC values are near random for most strategies; this is an unoptimized Agent 2 baseline, not a tuned result set.
- `ml/data/news_dates.csv` is still a placeholder, so `is_news_day` remains effectively inactive.
- `Implementation/ttm_squeeze_generator.py` still has the standalone Windows `cp1252` print issue noted in Agent 1A status; pipeline behavior is unaffected.
- Pytest still emits one non-blocking cache warning due the workspace temp-folder permissions on this machine.

## Next Agent Instructions
- Agent 3 should treat `ml/train.py`, `ml/model.py`, `ml/evaluate.py`, and `ml/topstep_risk.py` as the baseline training stack.
- Use the existing artifacts in `ml/artifacts/` as baseline references before any hyperparameter search or ONNX export work.
- Preserve the current feature order contract: 35 feature columns, with `future_return` and `label` treated as non-feature target columns.
- Keep setup-row gating semantics: each strategy model evaluates only bars where its own raw signal column is non-zero.
- Re-run `python -m pytest ml/tests/ -v --tb=short` before and after Agent 3 changes.

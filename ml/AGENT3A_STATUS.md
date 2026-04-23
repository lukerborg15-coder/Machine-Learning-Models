# Agent 3A Status - Funded Sim, HPO Guardrails, ONNX Export

**Date:** 2026-04-12 local session  
**Status:** Targeted Agent 3A scaffold implemented and tested. Not deployment-ready yet.

## Completed

- Added `ml/funded_sim.py` for post-Combine Express Funded simulation with Standard and Consistency paths.
- Added 50K XFA defaults: `$0` start, `$2,000` MLL, `$1,000` DLL, MLL trailing to `$0`, MLL lock at `$0` after first payout, `$1,000` max-safe payout buffer, `$125` minimum payout request, and 90/10 payout split for current-policy new-trader assumptions.
- Added a conservative 50K XFA scaling-plan cap for funded simulation: requested strategy size defaults to 5 MNQ, but session contracts are capped to 2, then 3, then 5 as EOD balance reaches the configured tiers.
- Fixed the Consistency XFA logic to use largest winning day divided by total net profit in the payout window, not only the sum of positive days.
- Added `ml/hyperparam_search.py` with staged validation-only HPO helpers and a dry-run manifest.
- Added `ml/export_onnx.py` with opset 12 ONNX export, `feature_order_*.json`, `scaler_params_*.json`, `model_config_*.json`, and deterministic PyTorch-vs-ONNX parity checks.
- Exported a research-only TTM ONNX smoke artifact from the pre-hardening Agent 2 checkpoint. It is not a deployment candidate.
- Added `ml/tests/test_agent3.py` for funded simulation, payout rules, scaling caps, HPO leakage guardrails, and ONNX artifact contracts.
- Added Agent 3A/3B Obsidian handoff docs under `Agents/`.

## Verification

- `python -m pytest ml/tests/test_agent3.py -v --tb=short`: `13 passed, 2 skipped, 1 warning`.
- `py -3.13 -m pytest ml/tests/test_agent3.py -v --tb=short`: `15 passed, 1 warning`.
- `py -3.13 -c "from ml.export_onnx import verify_onnx_matches_pytorch; print(verify_onnx_matches_pytorch('ttm'))"`: max diff `1.9744038581848145e-07`, under `1e-5`, using 3 parity samples.
- `python -m pytest ml/tests -v --tb=short`: `64 passed, 2 skipped, 1 warning`.

The warning is the known Windows pytest cache permission warning under `.pytest_cache` / `pytest-cache-files-*`.

## Current Artifacts

- Research-only ONNX smoke export:
  - `ml/artifacts/model_ttm.onnx`
  - `ml/artifacts/feature_order_ttm.json`
  - `ml/artifacts/scaler_params_ttm.json`
  - `ml/artifacts/model_config_ttm.json`
- Existing Agent 2 baseline artifacts remain pre-hardening references and should not be treated as final deployable Agent 3 outputs.
- No real `hyperparam_{strategy}.csv` files have been produced by a long HPO run.
- HPO test output is configured to use `output_dir=None` so tests do not create fake artifact CSVs.

## Not Complete

- Full staged HPO has not been run.
- Models have not been retrained from refreshed feature parquets after the `is_news_day` calendar fix.
- No strategy has been marked deployment-approved.
- ONNX export has only been smoke-tested with TTM; all-strategy exports should happen only after real HPO/retraining.
- Bar-level DLL/MLL checks still evaluate after simulated exits, not intratrade unrealized liquidation thresholds. This is acceptable for a first bar-level research scaffold, but it is not exact live rule emulation.

## Next Required Step

Run a one-strategy real timing benchmark, then decide the HPO trial count before launching full Agent 3A training. Agent 3B should audit the real HPO output and all exported artifacts before any NinjaScript deployment planning.

## Final Agent 3A Completion Update

**Date:** 2026-04-12 late session

- Completed the real Agent 3A run with `py -3.13`.
- One-trial TTM benchmark took `2.1891082` seconds, so the run selected `20` HPO trials per strategy.
- HPO completed for all 8 strategies and wrote `hyperparam_{strategy}.csv` plus `hpo_summary.csv`.
- Final retraining refreshed all 8 checkpoints/scalers/eval CSVs from the selected configs and saved the deployment/reporting checkpoint from `fold_2`.
- `agent3_rankings.csv` and `FINAL_EVAL_REPORT.md` show `0` research candidates and `0` deployment candidates because no strategy passed the Combine gate on the final held-out fold.
- No ONNX artifacts are exported after the final run; stale TTM smoke-export ONNX/sidecar files were removed to avoid mismatch with refreshed checkpoints.
- Agent 3B found no deployment blocker in the completed run. The medium future-run issue around missing `save_fold_name` fallback was fixed, and symmetric long/short DLL/MLL tests were added.

Final verification:

- `python -m pytest ml/tests -v --tb=short`: `68 passed, 2 skipped, 1 warning` before the final audit-cleanup patch.
- `python -m pytest ml/tests/test_agent3.py -v --tb=short`: `19 passed, 2 skipped, 1 warning` after the final audit-cleanup patch.
- `py -3.13 -m pytest ml/tests/test_agent3.py -v --tb=short`: `20 passed, 1 skipped, 1 warning` after the final audit-cleanup patch.

The remaining warning is still the known Windows pytest cache permission warning.

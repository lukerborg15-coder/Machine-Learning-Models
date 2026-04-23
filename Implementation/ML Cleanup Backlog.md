# ML Cleanup Backlog

**Purpose:** Cleanup and hardening tasks discovered after the Agent 1 + Agent 2 baseline run completed on 2026-04-09.

---

## Agent 1 Cleanup

- Update `ml/AGENT1B_STATUS.md` so it clearly distinguishes 35 feature columns from the two target columns (`future_return`, `label`).
- Refresh `ml/AGENT1B_STATUS.md` test counts so the handoff note matches the current suite size.
- Fix the Windows `cp1252` console-print failure in `Implementation/ttm_squeeze_generator.py` by removing or replacing the non-ASCII arrow in the standalone script output.
- Clean up or quarantine the `pytest-cache-files-*` temp folders that keep causing the recurring pytest cache warning in this workspace.

## Agent 2 Cleanup

- Split `ml/train.py` into smaller modules once the behavior is considered stable:
  - sequence dataset and window building
  - scaling and split utilities
  - training loop
  - orchestration and artifact management
- Add deterministic seeding for numpy and torch so baseline comparisons are repeatable across reruns.
- Add a project-local dependency manifest for the training stack (`torch`, and any future ONNX/runtime deps) so the environment is reproducible.
- Improve evaluation artifact output so confusion matrices and detailed fold metadata are written as structured JSON sidecars instead of CSV string blobs.
- Decide and document a formal policy for cases where Sharpe is undefined because trade counts or return variance are too small.
- Add richer runtime logs during training so the `run_{strategy}.log` paths correspond to actual saved logs instead of reserved artifact paths only.

## Cross-Cutting Documentation Cleanup

- Keep `Home.md`, `Implementation/ML Operators Guide.md`, and the Agent notes aligned whenever a workflow decision is locked in code.
- Preserve the shared feature contract explicitly everywhere:
  - 35 model input columns
  - `future_return` and `label` are target-only columns
- Keep the setup-row gating rule documented everywhere:
  - a strategy model trains and evaluates only on rows where that strategy's raw signal column is non-zero
- Revisit the Agent 3 export note after ONNX is implemented so the feature-order and scaler-export docs match the final code exactly.

## Nice-To-Have Hardening

- Add a small integration test that trains one strategy end-to-end and verifies the expected artifacts are created.
- Consider moving baseline run summaries into a dedicated report note inside the vault after Agent 3 is complete.

## Completed Hardening

- Replaced `ml/data/news_dates.csv` with a real 2021-2026 FOMC/NFP/CPI calendar in Agent 2.5 hardening.
- Added synthetic evaluation smoke tests for TopStep trade outcomes, daily-loss blocking, and consistency failure in `ml/tests/test_hardening.py`.
- Added all-strategy no-lookahead tests, direct TopStep risk math tests, and Agent 2 artifact contract tests in `ml/tests/test_hardening.py`.

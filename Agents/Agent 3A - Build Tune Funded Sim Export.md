# Agent 3A - Build, Tune, Funded Sim, Export

**Phase:** Agent 3 implementation
**Role:** Builder
**Companion:** `Agent 3B - Independent Audit`
**Primary goal:** Produce models and deployment artifacts ranked by payout-adjusted survival, not classification metrics alone.

---

## Required Starting Context

Read these before making changes:

1. `Home.md`
2. `ml\AGENT2_5_HARDENING_STATUS.md`
3. `ml\AGENT_REVIEW_REPORT.md`
4. `ml\AGENT2_STATUS.md`
5. `Implementation\ML Operators Guide.md`
6. `Implementation\NinjaScript Integration.md`
7. `Reference\Prop Firm Rules.md`

Important: Agent 2 checkpoints and scalers are pre-hardening baseline references. The refreshed feature parquets now have active `is_news_day`, so final deployable models must be retrained/tuned from refreshed parquets before ONNX export.

---

## Build Scope

Implement and run the Agent 3 pipeline in this order:

1. Run `python -m pytest ml/tests -v --tb=short`.
2. Run a one-strategy timing benchmark before full search.
3. Use `ml\hyperparam_search.py` for staged validation-only search.
4. Use `ml\funded_sim.py` to simulate Express Funded payout paths after Combine pass.
5. Use `ml\export_onnx.py` to export ONNX with opset 12 and verify PyTorch-vs-ONNX parity.
6. Write `ml\artifacts\FINAL_EVAL_REPORT.md`.
7. Write `ml\AGENT3A_STATUS.md` with exact commands, artifacts, and known limitations.

---

## Ranking Rule

Use payout-adjusted survival as the primary ranking idea:

- First: strategy passes the Combine without rule failure.
- Second: strategy survives Express Funded simulation after passing.
- Third: strategy generates simulated trader payouts under Standard and/or Consistency path.
- Fourth: strategy remains consistent, avoids MLL/DLL failure, has enough trades, and keeps drawdown controlled.
- Last: Sharpe, profit factor, win rate, avg R, and F1 are diagnostics, not the primary objective.

Export ONNX artifacts for research inspection, but only mark models as deployment candidates when the report shows enough payout-adjusted evidence.

---

## Express Funded Simulation Defaults

- Scope: Express Funded only. Defer Live Funded.
- 50K XFA-style defaults: starts at `$0`, MLL starts at `-$2,000`, MLL trails up to `$0`, first payout locks MLL at `$0`.
- DLL: apply `$1,000` day lockout under the conservative non-TopstepX/NinjaTrader assumption unless rule config changes.
- Scaling plan: cap requested strategy size by the funded account's session-start contract tier. The default 50K scaffold uses `2` contracts at `$0`, `3` contracts at `$1,500+`, and `5` contracts at `$2,000+`.
- Standard path: `5` winning days of `$150+`.
- Consistency path: `3` traded days and `40%` consistency target, calculated as largest winning day divided by total net profit in the payout window.
- Payout request: max-safe payout, keeping the default `$1,000` buffer.
- Minimum payout request: `$125`.
- Sim duration: after a Combine pass, continue until account failure or data exhaustion.

---

## Deliverables

- `ml\funded_sim.py` complete and tested.
- `ml\hyperparam_search.py` produces validation-only hyperparameter CSVs.
- `ml\export_onnx.py` exports ONNX, feature order, scaler params, and model config.
- `ml\artifacts\hyperparam_{strategy}.csv` for all searched strategies.
- `ml\artifacts\model_{strategy}.onnx` and JSON sidecars for exported strategies.
- `ml\artifacts\FINAL_EVAL_REPORT.md`.
- `ml\AGENT3A_STATUS.md`.

Do not mark Agent 3 complete until Agent 3B finishes its independent audit.

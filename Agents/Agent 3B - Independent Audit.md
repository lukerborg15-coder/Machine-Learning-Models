# Agent 3B - Independent Audit

**Phase:** Agent 3 independent review
**Role:** Auditor
**Companion:** `Agent 3A - Build, Tune, Funded Sim, Export`
**Primary goal:** Verify Agent 3A artifacts before anything is trusted for deployment.

---

## Required Starting Context

Read these before reviewing:

1. `Home.md`
2. `Agents\Agent 3A - Build Tune Funded Sim Export.md`
3. `ml\AGENT3A_STATUS.md`
4. `ml\AGENT2_5_HARDENING_STATUS.md`
5. `ml\artifacts\FINAL_EVAL_REPORT.md`
6. `ml\artifacts\hyperparam_*.csv`
7. `ml\artifacts\model_config_*.json`
8. `ml\artifacts\feature_order_*.json`
9. `ml\artifacts\scaler_params_*.json`

---

## Audit Checklist

Validate:

- Hyperparameter search selected configs using validation metrics only.
- Test metrics were not used during hyperparameter selection.
- Express Funded simulation starts only after a Combine pass and continues until failure or data exhaustion.
- Standard and Consistency payout paths are both reported.
- Payout ledger records payouts, MLL floor, DLL lockouts, failure reasons, and account balances.
- Consistency uses total net profit, not just positive-day profit.
- Funded simulation respects the configured scaling-plan cap at session start.
- Payout requests enforce the configured minimum payout request.
- Max-safe payout buffer is applied.
- Deployment candidates are justified by payout-adjusted survival, not F1 alone.
- Sparse or insufficient-data strategies are marked research-only.
- ONNX files use opset 12 and match PyTorch outputs within `1e-5`.
- Feature order has exactly 35 features and excludes `future_return` and `label`.
- Scaler JSON arrays have exactly 35 means and 35 std values in the same feature order.

---

## Required Commands

Run:

```powershell
python -m pytest ml/tests -v --tb=short
py -3.13 -m pytest ml/tests/test_agent3.py -v --tb=short
```

If ONNX artifacts were exported, also verify parity for every exported strategy with `py -3.13`.

---

## Output

Write `ml\AGENT3B_AUDIT.md` with:

- Findings first, ordered by severity.
- Any deployment blockers.
- Any research-only strategies and why.
- ONNX parity results.
- Whether `FINAL_EVAL_REPORT.md` is safe to use for NinjaScript deployment planning.

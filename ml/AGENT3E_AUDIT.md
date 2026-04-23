# Agent 3E Audit — Bootstrap Confidence Intervals

**Date:** 2026-04-16  
**Verdict: 3E APPROVED — proceed to Agent 3F**  
**Test suite:** 97 passed, 2 skipped in 490.43s (no regressions; +10 from 3E tests)

---

## Adversarial Checks (10/10 PASS)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Block bootstrap widens CIs vs iid on autocorrelated series | **PASS** | block p95-p05 = 20.908, iid p95-p05 = 5.643 (3.7× wider) |
| 2 | Reproducibility: random_state=123 repeat is bit-identical | **PASS** | all 6 summary keys identical |
| 3 | Percentile ordering p05 ≤ p50 ≤ p95 on every disk artifact | **PASS** | 8 files scanned, 0 violations |
| 4 | Pass-rate bootstrap replays TopStepRiskManager on day sequences | **PASS** | winning seq: point=1.0 mean=1.000; losing seq: point=0.0 mean=0.000 |
| 5 | Degenerate inputs stay finite, respect PF/Calmar caps | **PASS** | 0 violations across all-win / all-lose / zero-pnl scenarios |
| 6 | Unreliable flag fires for n<30, not for n≥30 | **PASS** | n=3 unreliable=True, n=50 unreliable=False |
| 7 | Gate rejects on each failure axis and accepts clean case | **PASS** | 4 rejection paths + 1 acceptance path all verified |
| 8 | Multiple-testing filter toggles correctly | **PASS** | with filter: rejected (p50=0.350<0.50); without: approved |
| 9 | Aggregated n_trades matches sum of per-fold n_trades | **PASS** | 8 strategies, 0 mismatches |
| 10 | Full pytest suite is green | **PASS** | 97 passed, 2 skipped, returncode=0 |

---

## Deployment gate results (TopStep 50K, 2026-04-16)

**0 of 8 strategies approved.** All fail on p05 Sharpe ≤ 0. Root cause: small trade counts (~33–85 trades total across 5 folds) produce very wide bootstrap CIs. The gate correctly refuses to approve strategies where the 5th-percentile scenario is negative.

| Strategy | n_trades | p05 Sharpe | p50 Sharpe | p50 PF | p05 pass-rate | Decision |
|----------|---------|------------|------------|--------|---------------|----------|
| ttm | 33 | −1.011 | 4.121 | 1.809 | 0.0 | REJECT (p05 Sharpe) |
| connors | 85 | −2.487 | −0.289 | 0.958 | 0.0 | REJECT (p05 Sharpe) |
| ifvg | 51 | −4.434 | −1.664 | 0.773 | 0.0 | REJECT (p05 Sharpe) |
| ifvg_open | 0 | NaN | NaN | NaN | NaN | REJECT (no trades) |
| orb_ib | 0 | NaN | NaN | NaN | NaN | REJECT (no trades) |
| orb_vol | 0 | NaN | NaN | NaN | NaN | REJECT (no trades) |
| orb_wick | 0 | NaN | NaN | NaN | NaN | REJECT (no trades) |
| orb_va | 0 | NaN | NaN | NaN | NaN | REJECT (no trades) |

---

## Implementation notes

- **`_sharpe` fix:** `np.full(n, -0.3).std()` is ~1e-17 (not 0) due to floating-point mean error. Fixed with `np.ptp(trades) == 0.0` constant-array guard before std division.
- **`pass_rate` bootstrap:** Resamples complete day sequences (not individual trades) and replays each sequence through `TopStepRiskManager`. Correctly binary (0.0 or 1.0 per resample), never fractional.
- **PF_CAP / CALMAR_CAP:** Enforced in `_profit_factor` and `_calmar` helpers; all-winning and all-losing scenarios confirmed finite.
- **Circular wrap:** `_resample_stationary_block` wraps modulo `n` so every index has equal probability.
- **Legacy gate:** `--legacy-gate` / `--gate-version=3c` routes to the existing Agent 3C per-fold-Sharpe gate unchanged.

---

## Files delivered

```
ml/bootstrap.py                         NEW
ml/evaluate.py                          UPDATED (+compute_bootstrap_cis, --bootstrap)
ml/funded_sim.py                        UPDATED (+4-condition gate, --gate-version=3e)
ml/train.py                             UPDATED (+per-fold CSV persistence)
ml/tests/test_agent3e.py               NEW (10 tests)
ml/tests/_audit_3e.py                  NEW (10 adversarial checks)
ml/artifacts/eval_*_bootstrap.json     GENERATED (8 strategies)
ml/artifacts/agent3e_deployment_decisions.csv  GENERATED
ml/artifacts/agent3e_audit_log.txt     GENERATED
ml/AGENT3E_STATUS.md                   NEW
ml/AGENT3E_AUDIT.md                    NEW (this file)
```

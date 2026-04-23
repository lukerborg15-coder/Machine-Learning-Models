# Agent 3E — Bootstrap Confidence Intervals + Deployment Gate

**Phase:** Statistical rigor and contract fix
**Prerequisites:**
- Agent 3D complete and 3D-Audit APPROVED (`ml/AGENT3D_AUDIT.md` shows "3D APPROVED")
- Triple-barrier labels wired into training
- All 5 folds trained and evaluated

---

## Execution Constraint — CODE COMPLETE, DO NOT EXECUTE

> **Write all code changes completely. Do NOT run pytest. Do NOT run bootstrap simulations. Do NOT execute any Python.** Mark your status file as "CODE COMPLETE — NOT EXECUTED". If you find yourself about to run a command, stop.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\ml\bootstrap.py` — read in full
2. `C:\Users\Luker\strategyLabbrain\ml\funded_sim.py` — read in full
3. `C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py` — read in full
4. `C:\Users\Luker\strategyLabbrain\ml\tests\test_agent3e.py` — read existing tests
5. `C:\Users\Luker\strategyLabbrain\ml\AGENT3D_STATUS.md`

---

## What Already Exists — Do Not Rebuild

**`ml/bootstrap.py`** — fully implemented. Contains:
- `stationary_block_bootstrap()` — block bootstrap preserving autocorrelation
- `bootstrap_trade_metrics()` — computes Sharpe, Calmar, profit_factor, win_rate, avg_r with CIs
- `bootstrap_pass_rate()` — resamples day sequences through TopStepRiskManager
- All functions return `{point, mean, std, p05, p50, p95}`

**`ml/funded_sim.py`** — fully implemented. Contains:
- `evaluate_bootstrap_deployment_gate()` — 4-condition gate

---

## Fix Tasks

### Task 1 — Fix topstep_risk.py contract limit (CRITICAL)

In `ml/topstep_risk.py`, the current code has:
```python
self.contracts = 5
```

This is wrong. TopStep 50K allows 5 NQ mini = **50 MNQ micro**. Point value = $2/point.

Replace the fixed contract approach with dynamic confidence-based sizing:

```python
self.max_contracts = 50
self.point_value = 2.0

def position_size(self, stop_pts: float, confidence: float) -> int:
    """Compute contracts based on ATR stop and model confidence.
    Targets $500 risk per trade, scaled by confidence tier.
    """
    if stop_pts <= 0:
        raise ValueError(f"stop_pts must be positive, got {stop_pts}")
    target_risk = 500.0
    base = int(target_risk / (stop_pts * self.point_value))
    base = max(1, base)
    if confidence >= 0.80:
        scale = 1.00
    elif confidence >= 0.70:
        scale = 0.80
    elif confidence >= 0.65:
        scale = 0.60
    else:
        scale = 0.40
    sized = max(1, int(base * scale))
    return min(sized, self.max_contracts)
```

Update `simulate_trade()` to accept a `contracts` parameter:
```python
def simulate_trade(self, entry, stop, target, exit_price,
                   contracts: int = 1, commission_per_rt=1.40):
    direction = 1 if target > entry else -1
    pnl_per_point = contracts * self.point_value
    raw_pnl = (exit_price - entry) * direction * pnl_per_point
    cost = commission_per_rt * contracts
    return raw_pnl - cost
```

Remove `self.contracts` and `self.pnl_per_point` from `__init__` — they are no longer fixed.

Update the comment in `__init__` to reflect the new architecture.

### Task 2 — Verify deployment gate thresholds

In `funded_sim.py`, find `evaluate_bootstrap_deployment_gate()`. Confirm the 4 rejection conditions match exactly:

| Condition | Threshold |
|---|---|
| Aggregated p05 Sharpe | must be > 0 |
| Any single fold point Sharpe | must be > -0.3 |
| Aggregated p50 profit factor | must be >= 1.2 |
| Aggregated p05 pass rate | must be >= 0.30 |

If any threshold differs, update it.

### Task 3 — Verify bootstrap_pass_rate uses TopStepRiskManager

In `bootstrap.py`, find `bootstrap_pass_rate()`. Confirm it:
- Resamples sequences of **daily P&L arrays** (not individual trade scalars)
- Calls `TopStepRiskManager` on each resampled sequence
- Does NOT bootstrap a pre-computed scalar pass_rate value

If it bootstraps a scalar, rewrite it to resample day sequences.

### Task 4 — Verify profit_factor cap

In `bootstrap.py`, find `_profit_factor()`. Confirm it caps the result at `10.0` to handle all-win sequences that would otherwise produce `inf`.

If no cap exists, add:
```python
return min(gross_wins / gross_losses, 10.0)
```

### Task 5 — Update tests

In `ml/tests/test_agent3e.py`, review existing tests. Add or update tests to cover:
- `test_max_contracts_is_50` — `TopStepRiskManager().max_contracts == 50`
- `test_position_size_caps_at_50` — `position_size(stop_pts=1.0, confidence=0.99) == 50`
- `test_position_size_confidence_scaling` — lower confidence gives fewer contracts
- `test_simulate_trade_accepts_contracts_param` — `simulate_trade(..., contracts=10)` works
- `test_profit_factor_capped_at_10` — all-win series returns 10.0 not inf
- `test_block_bootstrap_wider_than_iid_on_autocorrelated` — block bootstrap std > IID bootstrap std on sine wave series
- `test_bootstrap_reproducible` — same random_state gives identical results
- `test_deployment_gate_rejects_low_p05_sharpe` — p05 Sharpe=-0.1 → not approved
- `test_deployment_gate_rejects_low_profit_factor` — p50 PF=1.1 → not approved
- `test_deployment_gate_rejects_low_pass_rate` — p05 pass_rate=0.25 → not approved

---

## Status File

Write `ml/AGENT3E_STATUS.md`:
```markdown
# Agent 3E Status

## Status: CODE COMPLETE — NOT EXECUTED

## Changes Made
- [list each file changed with line numbers]

## What Was Already Correct
- [list what you verified but did not change]

## Tests Added/Updated
- [list test names]

## Known Issues
- [any concerns found during review]
```

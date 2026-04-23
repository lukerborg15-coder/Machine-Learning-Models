# Agent 3E-Audit — Statistical Review

**Role:** Independent auditor for Agent 3E's work
**Final gate:** This is the last audit before deployment decisions are made. Be strict.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Agents\Agent 3E - Bootstrap Confidence Intervals.md`
2. `C:\Users\Luker\strategyLabbrain\ml\bootstrap.py` — read in full
3. `C:\Users\Luker\strategyLabbrain\ml\funded_sim.py` — read the deployment gate
4. `C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py` — read in full
5. Do NOT read `AGENT3E_STATUS.md` until after you have run all checks and written your verdict

---

## Absolute Paths

```
Project root: C:\Users\Luker\strategyLabbrain
bootstrap.py: C:\Users\Luker\strategyLabbrain\ml\bootstrap.py
funded_sim.py: C:\Users\Luker\strategyLabbrain\ml\funded_sim.py
topstep_risk.py: C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py
```

---

## Check 1 — max_contracts Is 50, Not 5

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.topstep_risk import TopStepRiskManager

rm = TopStepRiskManager()
print(f"max_contracts: {rm.max_contracts}")
print(f"point_value:   {rm.point_value}")

assert rm.max_contracts == 50, (
    f"FAIL: max_contracts={rm.max_contracts}, expected 50. "
    "TopStep 50K allows 5 NQ mini = 50 MNQ micro."
)
assert rm.point_value == 2.0, f"FAIL: point_value={rm.point_value}, expected 2.0"
print("PASS: max_contracts=50, point_value=2.0")
```

---

## Check 2 — position_size Caps at 50 and Scales with Confidence

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.topstep_risk import TopStepRiskManager

rm = TopStepRiskManager()

# Hard ceiling
size_max = rm.position_size(stop_pts=1.0, confidence=0.99)
assert size_max == 50, f"FAIL: Max confidence should give 50 contracts, got {size_max}"

# Confidence scaling: lower confidence = fewer contracts
size_low  = rm.position_size(stop_pts=10, confidence=0.60)
size_high = rm.position_size(stop_pts=10, confidence=0.80)
assert size_low < size_high, (
    f"FAIL: Lower confidence should give fewer contracts. low={size_low}, high={size_high}"
)

# Stop scaling: tighter stop = more contracts at same risk
size_tight = rm.position_size(stop_pts=5,  confidence=0.75)
size_wide  = rm.position_size(stop_pts=20, confidence=0.75)
assert size_tight > size_wide, (
    f"FAIL: Tighter stop should give more contracts. tight={size_tight}, wide={size_wide}"
)

# Zero or negative stop must raise
try:
    rm.position_size(stop_pts=0, confidence=0.75)
    assert False, "FAIL: stop_pts=0 should raise an error"
except (ValueError, ZeroDivisionError):
    pass

print(f"PASS: position_size correct")
print(f"  stop=1pt, conf=0.99: {size_max} contracts (ceiling)")
print(f"  stop=10pt, conf=0.60: {size_low} contracts")
print(f"  stop=10pt, conf=0.80: {size_high} contracts")
print(f"  stop=5pt,  conf=0.75: {size_tight} contracts")
print(f"  stop=20pt, conf=0.75: {size_wide} contracts")
```

---

## Check 3 — simulate_trade Accepts contracts Parameter

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.topstep_risk import TopStepRiskManager

rm = TopStepRiskManager()

# Should not raise
pnl_1  = rm.simulate_trade(entry=100, stop=97, target=106, exit_price=106, contracts=1)
pnl_10 = rm.simulate_trade(entry=100, stop=97, target=106, exit_price=106, contracts=10)
pnl_50 = rm.simulate_trade(entry=100, stop=97, target=106, exit_price=106, contracts=50)

# 10 contracts should give 10x the PnL of 1 contract (ignoring per-contract commission rounding)
assert pnl_10 > pnl_1, f"FAIL: 10 contracts should earn more than 1 contract"
assert pnl_50 > pnl_10, f"FAIL: 50 contracts should earn more than 10 contracts"

print(f"PASS: simulate_trade accepts contracts parameter")
print(f"  1 contract: ${pnl_1:.2f}")
print(f"  10 contracts: ${pnl_10:.2f}")
print(f"  50 contracts: ${pnl_50:.2f}")
```

---

## Check 4 — Block Bootstrap Wider Than IID on Autocorrelated Series

```python
import numpy as np, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.bootstrap import stationary_block_bootstrap

np.random.seed(0)
n = 200
# Strong autocorrelation via sine wave
autocorrelated = np.sin(np.linspace(0, 6*np.pi, n)) * 100 + np.random.randn(n) * 10

def sharpe(x):
    return float(np.sqrt(252) * x.mean() / x.std()) if x.std() > 0 else 0.0

block_result = stationary_block_bootstrap(
    autocorrelated, sharpe, n_resamples=1000, expected_block_len=10.0, random_state=42
)
iid_result = stationary_block_bootstrap(
    autocorrelated, sharpe, n_resamples=1000, expected_block_len=1.0, random_state=42
)

print(f"Block bootstrap std: {block_result['std']:.4f}")
print(f"IID bootstrap std:   {iid_result['std']:.4f}")

assert block_result["std"] > iid_result["std"], (
    f"FAIL: Block bootstrap std ({block_result['std']:.4f}) should be > "
    f"IID std ({iid_result['std']:.4f}) on autocorrelated series."
)
print("PASS: Block bootstrap correctly wider than IID on autocorrelated series")
```

---

## Check 5 — Bootstrap Reproducible with Same random_state

```python
import numpy as np, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.bootstrap import stationary_block_bootstrap

np.random.seed(99)
trades = np.random.randn(100) * 50

def sharpe(x):
    return float(np.sqrt(252) * x.mean() / x.std()) if x.std() > 0 else 0.0

r1 = stationary_block_bootstrap(trades, sharpe, n_resamples=500, random_state=42)
r2 = stationary_block_bootstrap(trades, sharpe, n_resamples=500, random_state=42)

for key in ["point", "mean", "p05", "p50", "p95", "std"]:
    assert abs(r1[key] - r2[key]) < 1e-10, (
        f"FAIL: {key} not reproducible: r1={r1[key]}, r2={r2[key]}"
    )
print("PASS: Bootstrap fully reproducible with same random_state")
```

---

## Check 6 — Percentile Order p05 <= p50 <= p95

```python
import numpy as np, sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.bootstrap import bootstrap_trade_metrics

np.random.seed(0)
trades = np.random.randn(150) * 100 + 50

results = bootstrap_trade_metrics(trades, random_state=42)

for metric, data in results.items():
    assert data["p05"] <= data["p50"], (
        f"FAIL: {metric} p05 > p50: p05={data['p05']:.4f}, p50={data['p50']:.4f}"
    )
    assert data["p50"] <= data["p95"], (
        f"FAIL: {metric} p50 > p95: p50={data['p50']:.4f}, p95={data['p95']:.4f}"
    )
    print(f"  PASS {metric}: p05={data['p05']:.3f} <= p50={data['p50']:.3f} <= p95={data['p95']:.3f}")

print("PASS: All percentiles in correct order")
```

---

## Check 7 — Profit Factor Capped at 10.0 (No Inf)

```python
import sys, math
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.bootstrap import bootstrap_trade_metrics

# All-win trades — profit_factor would be infinite without a cap
all_wins = [100.0, 200.0, 150.0, 80.0, 120.0] * 20

import numpy as np
results = bootstrap_trade_metrics(np.array(all_wins), random_state=42)

for metric, data in results.items():
    for key, val in data.items():
        if isinstance(val, float):
            assert not math.isinf(val), f"FAIL: {metric}.{key} is infinite"
            assert not math.isnan(val), f"FAIL: {metric}.{key} is NaN"

pf = results["profit_factor"]["point"]
assert pf <= 10.0, f"FAIL: profit_factor={pf} exceeds cap of 10.0"
print(f"PASS: All-win series profit_factor capped at {pf:.1f}, no inf/NaN anywhere")
```

---

## Check 8 — Deployment Gate Rejects All Failure Cases

```python
import sys
sys.path.insert(0, r"C:\Users\Luker\strategyLabbrain")
from ml.funded_sim import evaluate_bootstrap_deployment_gate

def make_results(p05_sharpe=0.3, p50_sharpe=0.7, min_fold_sharpe=0.1,
                 p50_pf=1.5, p05_pass_rate=0.4, n_folds=5):
    return {
        "per_fold": [{"sharpe": {"point": min_fold_sharpe}}] * n_folds,
        "aggregated": {
            "sharpe":        {"p05": p05_sharpe, "p50": p50_sharpe},
            "profit_factor": {"p50": p50_pf},
            "pass_rate":     {"p05": p05_pass_rate},
        }
    }

# Case A: p05 Sharpe <= 0
r = evaluate_bootstrap_deployment_gate(make_results(p05_sharpe=-0.1))
assert not r.approved, f"FAIL case A: should reject p05 Sharpe=-0.1"
print(f"PASS case A: rejected ({r.reason})")

# Case B: one fold Sharpe <= -0.3
r = evaluate_bootstrap_deployment_gate(make_results(min_fold_sharpe=-0.4))
assert not r.approved, f"FAIL case B: should reject fold Sharpe=-0.4"
print(f"PASS case B: rejected ({r.reason})")

# Case C: p50 profit factor < 1.2
r = evaluate_bootstrap_deployment_gate(make_results(p50_pf=1.1))
assert not r.approved, f"FAIL case C: should reject PF=1.1"
print(f"PASS case C: rejected ({r.reason})")

# Case D: p05 pass rate < 0.30
r = evaluate_bootstrap_deployment_gate(make_results(p05_pass_rate=0.25))
assert not r.approved, f"FAIL case D: should reject pass_rate=0.25"
print(f"PASS case D: rejected ({r.reason})")

# Case E: all conditions met
r = evaluate_bootstrap_deployment_gate(make_results(
    p05_sharpe=0.1, p50_sharpe=0.6, min_fold_sharpe=0.0,
    p50_pf=1.3, p05_pass_rate=0.35
))
assert r.approved, f"FAIL case E: should approve when all conditions met. reason={r.reason}"
print(f"PASS case E: approved")

print("PASS: All 5 gate cases behave correctly")
```

---

## Check 9 — Full Test Suite

```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```

Record exact pass/fail counts.

---

## Audit Decision

Write `ml/AGENT3E_AUDIT.md`:

```markdown
# Agent 3E Audit Report

## Check Results
| Check | Description | Result | Notes |
|---|---|---|---|
| 1 | max_contracts=50, point_value=2.0 | PASS/FAIL | |
| 2 | position_size caps at 50, scales correctly | PASS/FAIL | [values] |
| 3 | simulate_trade accepts contracts param | PASS/FAIL | |
| 4 | Block bootstrap wider than IID | PASS/FAIL | [std values] |
| 5 | Bootstrap reproducible | PASS/FAIL | |
| 6 | Percentile order p05<=p50<=p95 | PASS/FAIL | |
| 7 | Profit factor capped at 10.0 | PASS/FAIL | |
| 8 | Gate rejects all 5 cases correctly | PASS/FAIL | |
| 9 | Full test suite | PASS/FAIL | [N passed] |

## Verdict
**3E APPROVED — [N] of 4 models cleared for TopStep 50K**
OR
**3E BLOCKED — [exact reasons]**
```

# Agent 3 — Hyperparameter Search, ONNX Export & Final Validation

**Phase:** F + G + H
**Deliverable:** One `model_{strategy}.onnx` per strategy + `feature_order_{strategy}.json` + final eval report
**Estimated session length:** 1 full context session
**Prerequisite:** Agent 2 must be complete - all 8 checkpoints and eval CSVs present

---

## Starting Point (2026-04-09)

Agent 2 baseline is complete.

- Verification state: `27 passed, 1 warning`
- Baseline artifacts present in `ml\artifacts\`
- Final Agent 2 handoff: `ml\AGENT2_STATUS.md`

Inherited conventions from the completed Agent 2 run:

- Use the shared 35 feature columns as inputs
- Keep `future_return` and `label` out of exported feature manifests
- Train/evaluate each strategy only on rows where its own raw signal column is non-zero
- Baseline evaluation uses fixed 5 MNQ contracts, 1.5 ATR stop, 1.0R target, and $1.40 round-trip transaction cost
- Walk-forward baseline uses two anchored folds:
  - 2021-03-19 to 2023-12-31 train, 2024 validate, 2025 test
  - 2021-03-19 to 2024-12-31 train, 2025 validate, 2026-03-18 test

Before changing architecture or export logic, review `ml\AGENT2_STATUS.md` and `Implementation\ML Cleanup Backlog.md`.

---

## Absolute Paths (use these for all file references)

```
Project root:         C:\Users\Luker\strategyLabbrain
ML folder:            C:\Users\Luker\strategyLabbrain\ml
Model artifacts:      C:\Users\Luker\strategyLabbrain\ml\artifacts
Agent 2 status:       C:\Users\Luker\strategyLabbrain\ml\AGENT2_STATUS.md
Eval results:         C:\Users\Luker\strategyLabbrain\ml\artifacts\eval_*.csv
ONNX outputs:         C:\Users\Luker\strategyLabbrain\ml\artifacts\model_{strategy}.onnx
```

Do not reference StrategyLab. All work is inside `strategyLabbrain\ml\`.

---

## Context to Read First (in this order)

1. [[Home]]
2. [[Architecture Overview]]
3. [[NinjaScript Integration]] — ONNX deployment spec
4. [[CNN Research Notes]] — overfitting warnings
5. [[Prop Firm Rules]] — final eval uses TopStep rules
6. `ml\AGENT2_STATUS.md` — **read this before anything else**
7. `ml\artifacts\eval_*.csv` — review all baseline results before tuning

---

## Architecture: Separate ONNX Per Strategy

Each strategy gets its own ONNX file. Do not combine models. NinjaScript loads each ONNX independently and runs them in parallel — each strategy fires its own signal with its own position.

**8 ONNX files to produce:**
```
model_ifvg.onnx
model_ifvg_open.onnx
model_orb_ib.onnx
model_orb_vol.onnx
model_orb_wick.onnx
model_orb_va.onnx
model_ttm.onnx
model_connors.onnx
```

Each paired with:
```
feature_order_ifvg.json
feature_order_ifvg_open.json
... etc
```

And:
```
model_config_ifvg.json       ← seq_len, n_features, n_filters, confidence_threshold
... etc
```

---

## Task List

### Task 1 — Verify Agent 2 deliverables
```bash
python -m pytest ml\tests\ -v --tb=short   # all tests must pass
ls ml\artifacts\                # verify 8 checkpoints + 8 eval CSVs
Get-Content ml\AGENT2_STATUS.md
```
Review eval results — record baseline Val Sharpe and Test Sharpe for each strategy before tuning.

### Task 2 — Hyperparameter search (per strategy)
In `ml\hyperparam_search.py`, run random search for each strategy independently:

```python
search_space = {
    "n_filters":    [32, 64, 128],
    "kernel_size":  [3, 5, 7],
    "n_layers":     [2, 3, 4],
    "dropout":      [0.2, 0.3, 0.5],
    "learning_rate":[1e-3, 3e-4, 1e-4],
    "seq_len":      [20, 30, 60]
}
n_trials = 30 per strategy (reduce to 15 if compute is limited)
```

**Rules:**
- Tune on **validation set only** — test set untouched during search
- Select best config per strategy based on **validation Sharpe**, not F1
- Save top-3 configs per strategy to `ml\artifacts\hyperparam_{strategy}.csv`

### Task 3 — Optional: retrain best config on train+val

If best val Sharpe > baseline by 20%+:
- Retrain on combined train+val data (2021–2025)
- Evaluate on test set (2026) only
- If test Sharpe drops > 30% vs val Sharpe → likely overfit → revert to baseline checkpoint, document it
- If within 30% → proceed with retrained model

### Task 4 — Final test set evaluation (per strategy)
Run full evaluation on 2026 test set for each strategy using the selected best config:
- Apply `TopStepRiskManager` from `ml\topstep_risk.py`
- Report: F1, Sharpe, Calmar, profit factor, win rate, max DD
- Report: TopStep Combine pass rate, avg days to pass, consistency violations
- Flag any strategy where test Sharpe < 0 — do not export that strategy's ONNX

### Task 5 — ONNX export (per strategy)
In `ml\export_onnx.py`, loop over all 8 strategies:

```python
import torch, json

def export_strategy(strategy_name, model, seq_len, n_features, scaler, feature_cols):
    model.eval()
    dummy = torch.zeros(1, seq_len, n_features)

    torch.onnx.export(
        model, dummy,
        f"ml/artifacts/model_{strategy_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=12   # .NET 4.8 / NT8 compatible
    )

    # Save feature order — this is the contract with NinjaScript
    with open(f"ml/artifacts/feature_order_{strategy_name}.json", "w") as f:
        json.dump(list(feature_cols), f, indent=2)

    # Save scaler params for NinjaScript (hardcode these in C# or load from CSV)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "feature_order": list(feature_cols)
    }
    with open(f"ml/artifacts/scaler_params_{strategy_name}.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    # Save model config
    config = {
        "strategy": strategy_name,
        "seq_len": seq_len,
        "n_features": n_features,
        "n_classes": 3,
        "class_mapping": {"0": "Long", "1": "Short", "2": "NoTrade"},
        "confidence_threshold": 0.60
    }
    with open(f"ml/artifacts/model_config_{strategy_name}.json", "w") as f:
        json.dump(config, f, indent=2)
```

**opset_version=12 is mandatory** — NT8 runs .NET 4.8 and requires older ONNX opset for the wrapper DLL.

### Task 6 — Verify every ONNX matches its PyTorch model
For each of the 8 strategies, run:

```python
def test_onnx_matches_pytorch(strategy_name):
    import onnxruntime as ort
    model = load_checkpoint(f"ml/artifacts/best_model_{strategy_name}.pt")
    model.eval()
    seq_len = model_configs[strategy_name]["seq_len"]
    n_features = model_configs[strategy_name]["n_features"]

    dummy = torch.zeros(1, seq_len, n_features)
    pt_out = model(dummy).detach().numpy()

    sess = ort.InferenceSession(f"ml/artifacts/model_{strategy_name}.onnx")
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - onnx_out).max()
    assert max_diff < 1e-5, f"{strategy_name}: ONNX mismatch, max diff={max_diff}"
    print(f"{strategy_name}: ONNX verified ✓ (max diff={max_diff:.2e})")
```

Install if needed: `python -m pip install onnxruntime`

**All 8 must pass before writing AGENT3_STATUS.md.**

### Task 7 — Write FINAL_EVAL_REPORT.md

Save to `ml\artifacts\FINAL_EVAL_REPORT.md`:

```markdown
# Final Evaluation Report

## Per-Strategy Results

| Strategy | Val Sharpe | Test Sharpe | Overfitting? | Combine Pass% | Consistency OK | Export? |
|---|---|---|---|---|---|---|
| ifvg | | | | | | ✅/❌ |
| ifvg_open | | | | | | |
| orb_ib | | | | | | |
| orb_vol | | | | | | |
| orb_wick | | | | | | |
| orb_va | | | | | | |
| ttm | | | | | | |
| connors | | | | | | |

## Deployed Models
List only models with test Sharpe > 0:
- model_ifvg.onnx — n_features=[N], seq_len=[L], confidence_threshold=0.60
- [etc]

## NinjaScript Deployment Notes
- opset_version: 12
- Each model loads independently in NT8 via wrapper DLL
- Timezone: NT8 Central → Eastern = +1 hour offset
- Feature orders are in feature_order_{strategy}.json — hardcode in C#
- Scaler params are in scaler_params_{strategy}.json — apply before inference
```

---

## Deliverables

```
strategyLabbrain\ml\artifacts\
├── model_{strategy}.onnx           ← 8 files (only strategies with test Sharpe > 0)
├── feature_order_{strategy}.json   ← 8 files
├── scaler_params_{strategy}.json   ← 8 files (mean + std for each feature)
├── model_config_{strategy}.json    ← 8 files
├── hyperparam_{strategy}.csv       ← 8 files
├── eval_{strategy}.csv             ← 8 files (updated with best config results)
├── FINAL_EVAL_REPORT.md
└── AGENT3_STATUS.md
```

---

## AGENT3_STATUS.md Format

```markdown
## Completed
- Hyperparameter search: [N trials per strategy]
- ONNX exported: [list of strategies]
- ONNX output matches PyTorch: PASS for all exported models (max diff < 1e-5)
- All tests pass: python -m pytest ml\tests\ -v --tb=short

## Deployed Models (test Sharpe > 0)
| Strategy | ONNX file | n_features | seq_len | Confidence threshold |

## Excluded Models (test Sharpe ≤ 0)
| Strategy | Reason |

## NinjaScript Next Steps
1. Build wrapper DLL in Visual Studio targeting .NET 4.8
2. Reference Microsoft.ML.OnnxRuntime (use v1.x for .NET 4.8 compatibility)
3. Copy DLLs + ONNX files to Documents\NinjaTrader 8\bin\Custom
4. Load each model independently, apply scaler_params, feed features in exact feature_order
5. Apply +1 hour timezone offset for session filters
```

---

## Logic Gaps to Guard Against

1. **opset_version=12**: newer opsets may not be compatible with the NT8 wrapper DLL using .NET 4.8
2. **Scaler params must be exported**: NinjaScript must apply the same standardization as training — export mean and std as JSON, hardcode in C#
3. **Test set contamination**: test set is only used in Task 4, never during hyperparam search
4. **Overfitting threshold**: if test Sharpe < 50% of val Sharpe, flag as overfit — do not deploy
5. **Strategies with test Sharpe ≤ 0**: do not export ONNX for these — document them in the report, they need architectural changes before deployment
6. **dynamic_axes batch dimension**: required for NT8 to run inference on batch size 1
7. **Feature order is the contract**: export the exact 35 feature columns used by the selected Agent 2 training run; do not include `future_return` or `label`

# ML Operators Guide

**Purpose:** Step-by-step reference for running, extending, and debugging the ML training pipeline
**Related:** [[Architecture Overview]], [[Testing Requirements]]

---

## Project Structure

> **Note:** The ML pipeline is built from scratch inside `strategyLabbrain\ml\`. All agent specs reference this location. Do not reference or use the old `StrategyLab/` folder.

```
strategyLabbrain\
├── data\                          ← raw OHLCV CSVs (Databento)
├── Strategies\                    ← strategy spec files (read-only reference)
├── Implementation\
│   └── ifvg_generator.py          ← pre-built IFVG signal generator
├── ml\                             ← ML training pipeline (primary build target)
│   ├── signal_generators.py       ← all strategy signal generators
│   ├── dataset_builder.py         ← feature matrix construction
│   ├── model.py                   ← Conv1D CNN architecture
│   ├── train.py                   ← training loop + walk-forward
│   ├── evaluate.py                ← dual metrics (classification + trading)
│   ├── topstep_risk.py            ← TopStep risk simulator
│   ├── hyperparam_search.py       ← grid/random search
│   ├── export_onnx.py             ← ONNX export + deployment manifests
│   ├── data\                      ← feature matrix parquet files
│   ├── artifacts\                 ← model checkpoints, scalers, ONNX files
│   └── tests\
│       └── test_pipeline.py       ← full test suite (see Testing Requirements)
```

---

## Current Baseline

As of 2026-04-09:

- Agent 1A and Agent 1B are verified.
- Agent 2 baseline is complete.
- `python -m pytest ml/tests/ -v --tb=short` passes with `27 passed, 1 warning`.
- `ml/artifacts/` contains 8 baseline scalers, 8 checkpoints, and 8 evaluation CSVs.

### Baseline Conventions Locked By Agent 2 (Superseded — see new architecture)

> ⚠️ The Agent 4A/4B/3C/3D/3E build track supersedes these conventions. The items below are historical baseline. The new architecture is defined in `MASTER_CHANGE_PLAN.md`.

**New architecture (effective 2026-04-16):**
- **All-bars training:** each of the 4 models trains on the full ~98,000 session bars. Non-signal bars are labeled NoTrade (class 2). Signal-bar gating is removed.
- **4 grouped models:** model_1 (IFVG+ConnorsRSI2), model_2 (IFVGOpen+TTMSqueeze), model_3 (ORBIIB+SessionPivot), model_4 (ORBVol+ORBWick+ORBVA)
- **5 rolling walk-forward folds** with purge + embargo
- **Triple-barrier labels** instead of shift(-N) return
- **50 MNQ max** with dynamic confidence-based sizing (not fixed 5 MNQ)
- **9 strategies** including Session Level Pivots (session_pivot_signal)

---

## Phase A - Data Pipeline

**Goal:** Produce a clean, verified, leakage-free feature matrix from raw CSVs

1. Load data via `load_data(instrument, timeframe, session_only=True)`
   - `session_only=True` returns Eastern-time data filtered to 09:30–15:00 ET
   - `session_only=False` returns full tz-aware data with no RTH filter, used for forward-label computation before session alignment
2. Verify with `test_data_loader.py` tests before proceeding
3. Compute features: OHLCV normalized, synthetic delta, ATR, returns
4. Add strategy signals from signal_generators.py — each function returns a Series of 1/-1/0
5. Add session level pivot features
6. Add time features (time_of_day, day_of_week, is_news_day)
7. Run `test_feature_engineering.py` — all tests must pass
8. Export feature matrix to `ml\data\features_{instrument}_{tf}.parquet`

**Agent 1B handoff requirement:** All tests pass. Feature matrix parquet files exist for the required `MNQ` timeframes. `AGENT1B_STATUS.md` written with: files created, feature column list, shape of each parquet, any known issues.

---

## Phase B — Label Generation

**Goal:** Produce triple-barrier labels that match actual trade execution logic

Strategy signals remain input features. Labels are a separate 3-class target:
- 0 = Long win (target hit before stop)
- 1 = Short win (target hit before stop)
- 2 = NoTrade (non-signal bar, OR signal bar that timed out or was a loss direction)

**Label logic (triple-barrier):** For each signal bar, simulate a forward price path:
- Stop = 1.5 × ATR below (long) or above (short) entry
- Target = 2.5 × ATR above (long) or below (short) entry
- Time limit = session end (15:00 ET) or max_bars (whichever comes first)
- Whichever barrier is hit first determines the label
- If stop or time limit hit first → NoTrade(2) [conservative — model should learn to filter these]
- Non-signal bars → always NoTrade(2)

**This replaces `shift(-N)` return labels.** The shift(-N) approach does not account for ATR-based stops and cannot model session boundary exits. Triple-barrier labels match the deployment objective exactly.

See Agent 3D spec for full implementation details.

---

## Phase C — Model Architecture

**Goal:** Build the Conv1D CNN, test forward pass

See [[Architecture Overview]] for architecture spec.

Key implementation notes:
- Use manual left-only causal padding in PyTorch (for example `ConstantPad1d((kernel_size - 1, 0), 0)` before `Conv1d(..., padding=0)`) — mandatory to prevent temporal leakage
- Input normalization: apply scaler before feeding to model, not inside the model
- Output: 3 logits (Long, Short, No Trade) — use CrossEntropyLoss for training

---

## Phase D - Walk-Forward Training

**Goal:** Train on 5 rolling walk-forward folds with all-bars data, validate on held-out windows

**New architecture — 5 rolling folds:**

```python
WALK_FORWARD_FOLDS = [
    FoldSpec("fold_1", train_start="2021-03-19", train_end="2023-06-30",
             val_start="2023-07-01", val_end="2023-12-31",
             test_start="2024-01-01", test_end="2024-06-30"),
    FoldSpec("fold_2", train_start="2021-03-19", train_end="2023-12-31",
             val_start="2024-01-01", val_end="2024-06-30",
             test_start="2024-07-01", test_end="2024-12-31"),
    FoldSpec("fold_3", train_start="2021-03-19", train_end="2024-06-30",
             val_start="2024-07-01", val_end="2024-12-31",
             test_start="2025-01-01", test_end="2025-06-30"),
    FoldSpec("fold_4", train_start="2021-03-19", train_end="2024-12-31",
             val_start="2025-01-01", val_end="2025-06-30",
             test_start="2025-07-01", test_end="2025-12-31"),
    FoldSpec("fold_5", train_start="2021-03-19", train_end="2025-06-30",
             val_start="2025-07-01", val_end="2025-12-31",
             test_start="2026-01-01", test_end="2026-03-18"),
]
```

Within each fold, per model group:
- Apply purge: remove the last `forward_horizon_bars` rows from the train split (prevents label leakage)
- Apply embargo: skip the first `embargo_bars` rows of the val split (5min = 78 bars = 1 full session)
- Fit the scaler on the purged train split only
- Transform val and test using that train-fit scaler
- Build sequence windows after scaling
- **Train on ALL rows** — non-signal bars included (NoTrade label). Do NOT filter to signal rows.
- Use class weights to compensate for NoTrade imbalance (~95%+ of bars)

Save scaler to `ml/artifacts/scaler_{model_name}_fold{n}.pkl` per fold.

---

## Phase E — Evaluation

**Goal:** Report both classification metrics AND trading metrics

**Classification metrics:**
- F1 score (macro-averaged across 3 classes)
- ROC-AUC (one-vs-rest)
- Precision/Recall per class

**Trading metrics (simulate the strategy using model predictions):**
- Sharpe ratio (annualized)
- Calmar ratio (annualized return / max drawdown)
- Profit factor (gross wins / gross losses)
- Max drawdown (%)
- Win rate
- Average R per trade

Current evaluation assumptions (updated architecture):

- Signal bars only are evaluated (model must predict Long or Short with confidence ≥ 0.60 to count as a trade)
- Trade direction must match the model's predicted class (0=Long, 1=Short)
- **Position sizing is dynamic:** `position_size(stop_pts, confidence)` — target $500 risk/trade, max 50 MNQ
- Stop distance: 1.5 × ATR(14) at signal bar
- Profit target: 2.5 × ATR(14) at signal bar
- Transaction cost: $1.40 round trip per contract
- Deployment gate: 4-condition bootstrap check (Sharpe p05 > 0, per-fold Sharpe > -0.3, PF p50 > 1.2, pass_rate p05 > 0.30)

Both sets of metrics must be reported. A model with good F1 but negative Sharpe is not usable. A model with poor F1 but good Sharpe/Calmar is interesting but suspicious.

---

## Phase F — Hyperparameter Search

**Goal:** Find best model config without overfitting to validation set

Use random search or Optuna. Test:
- `n_filters`: 32, 64, 128
- `kernel_size`: 3, 5, 7
- `n_layers`: 2, 3, 4
- `dropout`: 0.2, 0.3, 0.5
- `learning_rate`: 1e-3, 3e-4, 1e-4
- `sequence_length`: 20, 30, 60

Report best config on **validation set only**. Test set is not touched until Phase G.

---

## Phase G — ONNX Export

```bash
python ml/export_onnx.py  # exports 4 ONNX model files
```

This script writes 4 sets of files — one per model group:
```
ml/artifacts/model_1.onnx  + feature_order_1.json  + scaler_params_1.json
ml/artifacts/model_2.onnx  + feature_order_2.json  + scaler_params_2.json
ml/artifacts/model_3.onnx  + feature_order_3.json  + scaler_params_3.json
ml/artifacts/model_4.onnx  + feature_order_4.json  + scaler_params_4.json
```

All 12 files are required for NinjaScript integration. Feature order and scaler params in the JSON files must match the C# implementation exactly.

**Verification:** Run `test_onnx_output_matches_pytorch()` for all 4 models — must pass before marking complete.

---

## Running Tests

```bash
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests/ -v --tb=short
```

All tests must pass. On any failure, fix the issue and re-run before proceeding to the next phase.

## Runtime Progress Tracking

Runtime status files are only for explicit pipeline execution commands.

- `ml/artifacts/run_status.json` and `ml/artifacts/run_history.jsonl` are written only while a user-executed pipeline command is running
- Module import, unit tests, and passive helper calls must not create or mutate runtime status files
- Agent 1B batch feature-build commands are the first place runtime tracking is expected to run
- `AGENT*_STATUS.md` files remain handoff documents, not live runtime status files

---

## Agent Session Handoff Protocol

At end of each agent session:
1. Run full test suite — all tests pass
2. Write `ml/AGENT{N}_STATUS.md`:
   ```
   ## Completed
   - [List of phases/files completed]
   
   ## Artifacts Produced
   - [List of files with their paths and descriptions]
   
   ## Known Issues
   - [Any known limitations or deferred items]
   
   ## Next Agent Instructions
   - [Exact first step for the next agent to take]
   - [Any parameters or decisions the next agent must make]
   ```
3. The next agent's first action is always: read this vault + read the status file

---

## Common Mistakes to Avoid

See [[Testing Requirements]] for the full test suite. Most critical:

1. **Never shuffle time-series data** — no `train_test_split(shuffle=True)`, no random resampling
2. **Scaler fit on train only** — fitting on all data before split leaks validation/test statistics into training
3. **Causal convolution padding** — `padding='same'` looks forward; in PyTorch use manual left-only causal padding
4. **Forward return direction** — `shift(-1)` moves future data to current row (correct), `shift(+1)` moves past data to current row (backward-looking, wrong for labels)
5. **High == low guard** — synthetic delta formula must check for zero range
6. **SMA(200) warmup** — 200 bars before any signal; on a 5min chart this is ~3.3 full session days

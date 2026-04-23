# Agent 2 — Model Architecture, Training Loop & Evaluation

**Phase:** C + D + E
**Deliverable:** Trained model checkpoint per strategy + evaluation results CSV per strategy
**Estimated session length:** 1 full context session
**Prerequisite:** Agent 1B must be complete - all parquet files and tests passing

---

## Completion Snapshot (2026-04-09)

This phase is complete.

- Final verification result: `27 passed, 1 warning`
- Artifacts written: 8 scalers, 8 checkpoints, 8 evaluation CSVs
- Final handoff file: `ml\AGENT2_STATUS.md`

Baseline decisions locked during the completed run:

- All strategies use the shared 35 feature columns as inputs
- `future_return` and `label` are target-only columns
- Each strategy trains and evaluates only on rows where its own raw signal column is non-zero
- Walk-forward uses two anchored folds:
  - 2021-03-19 to 2023-12-31 train, 2024 validate, 2025 test
  - 2021-03-19 to 2024-12-31 train, 2025 validate, 2026-03-18 test
- Evaluation baseline uses fixed 5 MNQ contracts, 1.5 ATR stop, 1.0R target, and $1.40 round-trip transaction cost

Use this note as historical spec only. Agent 3 should start from `ml\AGENT2_STATUS.md` plus `ml\artifacts\eval_*.csv`.

---

## Absolute Paths (use these for all file references)

```
Project root:         C:\Users\Luker\strategyLabbrain
Data (CSV files):     C:\Users\Luker\strategyLabbrain\data
ML folder:            C:\Users\Luker\strategyLabbrain\ml
Feature matrices:     C:\Users\Luker\strategyLabbrain\ml\data
Model artifacts:      C:\Users\Luker\strategyLabbrain\ml\artifacts
Agent 1B status:      C:\Users\Luker\strategyLabbrain\ml\AGENT1B_STATUS.md
```

Do not reference StrategyLab. Everything was built from scratch by Agent 1A and 1B inside `strategyLabbrain\ml\`.

---

## Context to Read First (in this order)

1. [[Home]]
2. [[Architecture Overview]]
3. [[CNN Research Notes]]
4. [[Prop Firm Rules]] — TopStep risk rules and $300–$600 risk per trade
5. [[Testing Requirements]]
6. `ml\AGENT1B_STATUS.md` — if present, read it before anything else
7. The MNQ parquet column order documented in `AGENT1B_STATUS.md` — preserve it exactly through training and export

---

## Architecture: One Model Per Strategy

Each strategy gets its own independently trained Conv1D CNN. This means 8 separate training runs producing 8 separate checkpoint files. The completed baseline uses the same shared 35-feature input contract for all strategies, but only rows where that strategy's own raw signal column is non-zero are used for windows and evaluation.

**Why separate models:**
- Each strategy fires in different market conditions — one combined model would average across them
- Easier to retrain individual strategies as rules are refined
- Each ONNX file loads independently in NinjaScript — strategies run in parallel
- Failed strategies can be swapped out without retraining everything

**Strategy → model mapping:**
```
ifvg           → best_model_ifvg.pt       → model_ifvg.onnx
ifvg_open      → best_model_ifvg_open.pt  → model_ifvg_open.onnx
orb_ib         → best_model_orb_ib.pt     → model_orb_ib.onnx
orb_vol        → best_model_orb_vol.pt    → model_orb_vol.onnx
orb_wick       → best_model_orb_wick.pt   → model_orb_wick.onnx
orb_va         → best_model_orb_va.pt     → model_orb_va.onnx
ttm            → best_model_ttm.pt        → model_ttm.onnx
connors        → best_model_connors.pt    → model_connors.onnx
```

---

## Task List

### Task 1 — Verify Agent 1 deliverables
```bash
python -m pytest ml\tests\ -v --tb=short
ls ml\data\          # verify all parquet files exist
Get-Content ml\AGENT1B_STATUS.md   # read before proceeding if present
```
If any test fails: stop, fix Agent 1A/1B's work, rerun tests before continuing.

### Task 2 — Build Conv1D CNN architecture
In `ml\model.py`, implement `TradingCNN`:

```python
class TradingCNN(nn.Module):
    def __init__(self, n_features, seq_len, n_filters=64,
                 kernel_size=3, n_layers=2, dropout=0.3, n_classes=3):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_layers):
            out_ch = n_filters * (2 ** i)
            # Causal padding: pad (kernel_size-1) on left, 0 on right
            layers += [
                nn.ConstantPad1d((kernel_size - 1, 0), 0),
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=0),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch)
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) → transpose to (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

**Critical:** Causal padding is implemented manually using `ConstantPad1d` — this ensures no future data leaks through the convolution.

Verify with:
```python
model = TradingCNN(n_features=35, seq_len=30)
dummy = torch.zeros(2, 30, 35)
out = model(dummy)
assert out.shape == (2, 3)
```

### Task 3 — Implement sliding window dataset
In `ml\train.py`, implement `TradingDataset`:

```python
class TradingDataset(Dataset):
    def __init__(self, features_array, labels_array, seq_len=30):
        # features_array: np.array shape (n_bars, n_features)
        # labels_array: np.array shape (n_bars,) — 0=Long, 1=Short, 2=NoTrade
        # window: bars [i-seq_len+1 : i+1], label = labels[i]
        self.X = []
        self.y = []
        for i in range(seq_len - 1, len(features_array)):
            self.X.append(features_array[i - seq_len + 1 : i + 1])
            self.y.append(labels_array[i])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
```

**Never** include the label bar's future price in the window. Window ends at bar i, label is for bar i.

### Task 4 — Implement train/val/test split (per strategy)

```python
TRAIN_END = pd.Timestamp("2024-12-31", tz="America/New_York")
VAL_END   = pd.Timestamp("2025-12-31", tz="America/New_York")
# Test = everything after VAL_END

def split_and_scale(df, feature_cols, label_col):
    train = df[df.index <= TRAIN_END]
    val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test  = df[df.index > VAL_END]

    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    joblib.dump(scaler, f'ml/artifacts/scaler_{strategy_name}.pkl')

    return (
        scaler.transform(train[feature_cols]), train[label_col].values,
        scaler.transform(val[feature_cols]),   val[label_col].values,
        scaler.transform(test[feature_cols]),  test[label_col].values
    )
```

### Task 5 — Implement training loop

In `ml\train.py`, implement `train_model(strategy_name, config)`:
- Adam optimizer, `lr=3e-4`
- CrossEntropyLoss with class weights = inverse of class frequency (NoTrade dominates without this)
- Early stopping: patience=10 epochs on val loss
- Save best checkpoint to `ml\artifacts\best_model_{strategy_name}.pt`
- Log per epoch: train_loss, val_loss, val_f1

### Task 6 — Implement TopStep risk simulator
In `ml\topstep_risk.py`, **copy the canonical `TopStepRiskManager` class exactly from [[Prop Firm Rules]]**.

Do NOT rewrite or create a new version. The canonical class in Prop Firm Rules.md includes:
- Position sizing fields (`contracts`, `point_value`, `pnl_per_point`)
- `simulate_trade()` method with commission handling
- `update_eod()` using `self.max_trailing_dd` (not hardcoded)
- `check_intraday()` using `self.daily_loss_limit` (not hardcoded)
- `is_passed()` using `self.initial_account` and `self.profit_target` (not hardcoded)
- `check_consistency()` for the 40% single-day rule

Copy it exactly, then add any additional helper methods needed for the evaluation phase.

### Task 7 — Implement evaluation (per strategy)
In `ml\evaluate.py`, implement `evaluate_strategy(strategy_name, model, test_dataset)`:

**Classification metrics:** F1 macro, ROC-AUC, confusion matrix
**Trading simulation:**
- For each Long/Short signal: simulate trade with ATR-based stop, 1.0–1.5R target
- Transaction cost: $1.40 round trip per MNQ contract (commission + slippage)
- Apply `TopStepRiskManager` — enforce DLL, trailing DD, consistency rule
- Report: Sharpe, Calmar, profit factor, win rate, avg R, max drawdown
- Report: TopStep Combine pass rate across walk-forward windows, avg days to pass

Save to `ml\artifacts\eval_{strategy_name}.csv`

### Task 8 — Train all 8 strategy models
Run training sequentially for all 8 strategies. Prioritize if compute is limited:
1. `mnq_5min` features for ORB strategies (IB, Vol, Wick, VolAdapt)
2. `mnq_1min` or `mnq_3min` for IFVG and IFVG Open
3. `mnq_5min` for TTMSqueeze and ConnorsRSI2

### Task 9 — Add model tests to test suite
Add to `ml\tests\test_pipeline.py`:
- `test_model_output_shape()` — output is (batch, 3)
- `test_causal_padding_no_leakage()` — truncating input should not change earlier predictions
- `test_no_shuffle_in_time_split()` — train < val < test chronologically
- `test_scaler_fit_only_on_train()` — verify train scaler stats differ from full-data stats
- `test_class_weights_not_uniform()` — NoTrade is dominant, weights must compensate
- `test_topstep_trailing_dd_from_eod_peak()` — DD trails from max EOD, not account start
- `test_topstep_consistency_rule()` — single day > 40% of profits fails check

All tests must pass before writing AGENT2_STATUS.md.

---

## Deliverables

```
strategyLabbrain\ml\
├── model.py                              ← Conv1D CNN with causal padding
├── train.py                              ← training loop + split + dataset
├── topstep_risk.py                       ← TopStep risk simulator
├── evaluate.py                           ← dual metrics evaluation
├── artifacts\
│   ├── scaler_{strategy}.pkl             ← one per strategy, fit on train only
│   ├── best_model_{strategy}.pt          ← one per strategy
│   └── eval_{strategy}.csv              ← one per strategy
└── AGENT2_STATUS.md
```

---

## AGENT2_STATUS.md Format

```markdown
## Completed
- [tasks completed]

## Model Config Used
- n_features per strategy: [list each strategy + feature count]
- seq_len: [value]
- n_filters / n_layers / dropout: [values]

## Results per Strategy
| Strategy | Val F1 | Val Sharpe | Test F1 | Test Sharpe | Combine Pass Rate |
|---|---|---|---|---|---|

## Known Issues
- [convergence issues, class imbalance, data gaps]

## Next Agent Instructions
Agent 3 runs hyperparameter search then exports each model to ONNX.
- model.py and train.py are ready
- Baseline results in ml\artifacts\eval_*.csv
- 8 checkpoints in ml\artifacts\best_model_*.pt
- All tests pass: python -m pytest ml\tests\ -v --tb=short
```

---

## Logic Gaps to Guard Against

1. **Causal padding**: use `ConstantPad1d((kernel_size-1, 0), 0)` before each Conv1d — never use `padding='same'`
2. **Class imbalance**: NoTrade will be 70–90% of bars — always apply inverse frequency weights to CrossEntropyLoss
3. **Scaler on train only**: never call `scaler.fit()` on val or test data
4. **Window boundary across sessions**: a 30-bar window may span multiple trading days — this is valid for features, but ensure the label comes from the same day as the last bar in the window. **Critical order of operations:** compute labels (`shift(-n_forward)`) BEFORE session filtering. After filtering to 09:30–15:00, rows whose label references a bar outside the session (e.g., 14:55 + 5 bars = next day) will have NaN labels — drop these NaN-label rows. Do NOT compute labels after session filtering (that would make shift(-5) at 14:55 reference the next day's 9:30 bar, crossing the overnight gap).
5. **TopStep trailing DD**: trails from highest EOD balance seen — NOT from account start
6. **TopStep consistency**: must be explicitly checked — a profitable backtest can still fail this rule
7. **Transaction costs**: $1.40 round trip per MNQ contract — include in every simulated trade
8. **Risk per trade $300–$600**: position_size() must cap contracts to TopStep's max — verify the limit on topstep.com

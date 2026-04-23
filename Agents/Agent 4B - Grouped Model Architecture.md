# Agent 4B — Grouped Model Architecture + All-Bars Training

**Phase:** Architecture overhaul
**Deliverable:** 4 grouped models each trained on ALL session bars, 50 MNQ contract limit, confidence-based position sizing, Camarilla continuation signal added
**Estimated session length:** 1–2 full context sessions
**Prerequisite:** Agent 4A APPROVED by 4A-Audit. Rebuilt parquets must be present with pivot columns and session_pivot_signal.

---

## Execution Constraint — CODE COMPLETE, DO NOT EXECUTE

> **Write all code and tests completely. Do NOT run pytest. Do NOT run any training. Do NOT execute any Python.** The human will review the code first and run it on their own schedule. Your status file must be marked "CODE COMPLETE — NOT EXECUTED". If you find yourself about to run a command, stop.

---

## The Single Most Important Rule

> **Every model receives ALL session bars as training input. The training pool is NEVER filtered by signal activity.**

Current code gates training rows to only bars where a strategy's signal fires. This is wrong and is the primary cause of zero deployment candidates. The fix is to remove all signal-based row filtering from the training pipeline.

Every session bar within a fold's training date range is a training row. Signal bars get triple-barrier labels (implemented in Agent 3D — for now use the existing shift(-N) labels but applied to all bars). Non-signal bars get label = 2 (NoTrade). No exceptions.

**Verification:** After training, the row count per model per fold must be ~90% of the expected full-session bar count for that timeframe. For 5min data with ~252 trading days/year:
- 1-year fold train period: ~19,656 bars (252 × 78)
- 3-year fold train period: ~58,968 bars
- Any count below 80% of expected → bug in row filtering → BLOCK

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\Home.md`
2. `C:\Users\Luker\strategyLabbrain\MASTER_CHANGE_PLAN.md`
3. `C:\Users\Luker\strategyLabbrain\ml\AGENT4A_AUDIT.md` — must say APPROVED
4. `C:\Users\Luker\strategyLabbrain\Reference\Prop Firm Rules.md` — new position sizing
5. `C:\Users\Luker\strategyLabbrain\ml\train.py` — full file
6. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — full file
7. `C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py` — full file
8. `C:\Users\Luker\strategyLabbrain\ml\funded_sim.py` — full file

---

## Absolute Paths

```
Project root:       C:\Users\Luker\strategyLabbrain
train.py:           C:\Users\Luker\strategyLabbrain\ml\train.py
dataset_builder.py: C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py
topstep_risk.py:    C:\Users\Luker\strategyLabbrain\ml\topstep_risk.py
funded_sim.py:      C:\Users\Luker\strategyLabbrain\ml\funded_sim.py
evaluate.py:        C:\Users\Luker\strategyLabbrain\ml\evaluate.py
Parquets:           C:\Users\Luker\strategyLabbrain\ml\data\
Artifacts:          C:\Users\Luker\strategyLabbrain\ml\artifacts\
```

---

## New Model Architecture — 4 Grouped Models

### Strategy List (9 total — ORB VA removed, Camarilla Break added)

| Signal Column | Strategy | Type |
|---|---|---|
| `ifvg_signal` | IFVG | Trend sweep |
| `connors_signal` | ConnorsRSI2 | Mean reversion |
| `ifvg_open_signal` | IFVG Open | 9:30 gap sweep |
| `ttm_signal` | TTMSqueeze | Vol compression breakout |
| `orb_vol_signal` | ORB Vol | ORB + volume confirmation |
| `session_pivot_signal` | Session Pivot Rejection | Camarilla H3/H4/S3/S4 fade |
| `session_pivot_break_signal` | Session Pivot Break | Camarilla H4/S4 continuation |
| `orb_ib_signal` | ORB IB | Initial balance breakout |
| `orb_wick_signal` | ORBWick | ORB + clean candle wick filter |

**ORB VA is removed.** It was three nearly-identical ORB variants grouped together — low diversity. Replaced by `session_pivot_break_signal` which pairs naturally with `orb_vol_signal` (both confirm breakouts through different lenses — volume vs. level).

Replace the 8 separate single-strategy `TRAINING_JOBS` with 4 grouped model configs:

```python
# ml/train.py

MODEL_GROUPS = [
    {
        "model_name": "model_1",
        "strategies": ["ifvg", "connors"],
        "signal_cols": ["ifvg_signal", "connors_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "IFVG (trend sweep) + ConnorsRSI2 (mean reversion) — opposite logic",
    },
    {
        "model_name": "model_2",
        "strategies": ["ifvg_open", "ttm"],
        "signal_cols": ["ifvg_open_signal", "ttm_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "IFVG Open (9:30 sweep) + TTMSqueeze (vol compression) — different character",
    },
    {
        "model_name": "model_3",
        "strategies": ["orb_vol", "session_pivot", "session_pivot_break"],
        "signal_cols": ["orb_vol_signal", "session_pivot_signal", "session_pivot_break_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "ORB Vol (volume breakout) + Camarilla Rejection + Camarilla Continuation — all confirm breakouts from different angles",
    },
    {
        "model_name": "model_4",
        "strategies": ["orb_ib", "orb_wick"],
        "signal_cols": ["orb_ib_signal", "orb_wick_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "ORB IB (initial balance break) + ORBWick (clean candle filter) — range breakout pair",
    },
]
```

---

## Task 0 — Add session_pivot_break_signal (Camarilla Continuation)

Agent 4A added `session_pivot_signal` (rejection). This task adds `session_pivot_break_signal` (continuation/breakout) and rebuilds the parquets before any training changes.

**Logic — Long break signal (+1):**
All of the following on the same bar:
1. Price **closes above** H4 (not just touches it) — `bar.close > camarilla_h4`
2. The prior bar's close was at or below H4 — this is the first bar to close through
3. `atr_14` is not NaN and > 0
4. Daily count < max_per_day (2, shared with `session_pivot_signal`)

**Logic — Short break signal (-1):**
All of the following on the same bar:
1. Price **closes below** S4 — `bar.close < camarilla_s4`
2. The prior bar's close was at or above S4
3. `atr_14` is not NaN and > 0
4. Daily count < max_per_day (2, shared with `session_pivot_signal`)

**Critical distinction from rejection signal:**
- Rejection: bar touches H4, close stays BELOW H4 → fade the level
- Break: bar closes ABOVE H4 → go with the breakout

The daily cap of 2 is shared across both `session_pivot_signal` and `session_pivot_break_signal` combined — not 2 each.

**Implementation:**

In `ml/signal_generators.py`, add `session_pivot_break_signal()` following the same pattern as `session_pivot_signal()`. The function takes the same DataFrame input and returns a Series of -1/0/+1.

In `ml/dataset_builder.py`, wire it in alongside `session_pivot_signal` so both columns appear in the rebuilt parquet.

Column name must be exactly: `session_pivot_break_signal`

**After adding, rebuild parquets:**
```
python ml/dataset_builder.py --rebuild
```

**Verify the new signal:**
```python
import pandas as pd
from ml.signal_generators import session_pivot_break_signal

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
sig = session_pivot_break_signal(df)
vc = sig.value_counts()
print("Break signal distribution:", vc.to_dict())
assert 1 in vc.index, "No long break signals — check H4 close condition"
assert -1 in vc.index, "No short break signals — check S4 close condition"
n_signals = vc.get(1, 0) + vc.get(-1, 0)
rate = n_signals / len(df)
assert 0.001 < rate < 0.05, f"Signal rate {rate:.4f} outside expected range 0.1%-5%"
print(f"Break signal rate: {rate:.4f} ({n_signals} signals in {len(df)} bars)")

# Verify rejection and break never fire on same bar (they are mutually exclusive by definition)
rej = df["session_pivot_signal"]
brk = sig
conflict = ((rej != 0) & (brk != 0))
assert not conflict.any(), "FAIL: rejection and break signals fire on same bar — they are mutually exclusive"
print("PASS: rejection and break signals are mutually exclusive")
```

---

## Task 1 — Verify Starting State

```
cd C:\Users\Luker\strategyLabbrain
python -m pytest ml/tests -v --tb=short
```

Confirm Agent 4A-Audit says APPROVED. Check parquets have session_pivot_signal and camarilla columns. Record starting test count.

---

## Task 2 — Remove Signal-Bar Gating from dataset_builder.py

Find every location in `dataset_builder.py` and `train.py` where rows are filtered based on signal values. This includes:

- Any line like `df = df[df[signal_col] != 0]`
- Any filter like `df[df["label"].notna()]` that effectively removes non-signal rows before label assignment
- Any windowing step that skips non-signal bars

**Remove all such filters.**

After removing filters, confirm the training pool size:

```python
import pandas as pd

df = pd.read_parquet(r"C:\Users\Luker\strategyLabbrain\ml\data\features_mnq_5min.parquet")
# Count bars in the fold_1 train window (2021-03-19 to 2023-06-30 for the new 5-fold config)
train = df["2021-03-19":"2023-06-30"]
print(f"Training rows for fold_1 (5min): {len(train)}")
# Expected: ~252 days/year × 2.3 years × 78 bars/day ≈ 45,000 rows
# If you get 750-3000, the filter is still active — find and remove it
```

---

## Task 3 — Implement All-Bars Label Assignment

For each model group, assign labels to ALL bars in the training pool:

```python
def assign_labels(
    df: pd.DataFrame,
    signal_cols: list[str],
    forward_bars: int = 5,
) -> pd.DataFrame:
    """Assign 3-class labels to all bars.

    Label rules:
    - Bar where ANY signal in signal_cols is +1 AND future_return > 0: label = 0 (Long win)
    - Bar where ANY signal in signal_cols is +1 AND future_return <= 0: label = 2 (NoTrade / loss)
    - Bar where ANY signal in signal_cols is -1 AND future_return < 0: label = 1 (Short win)
    - Bar where ANY signal in signal_cols is -1 AND future_return >= 0: label = 2 (NoTrade / loss)
    - Bar where ALL signal_cols == 0: label = 2 (NoTrade)

    Note: Agent 3D will replace this with triple-barrier labels.
    This interim label uses shift(-N) return direction as a placeholder.
    The all-bars training structure must be in place BEFORE 3D runs.

    Args:
        df: Full session-filtered DataFrame for this fold's train/val/test split
        signal_cols: List of signal column names this model is responsible for
        forward_bars: Bars to look forward for label (shift(-forward_bars))

    Returns:
        df with "label" column added (0, 1, or 2). No rows dropped.
    """
    df = df.copy()

    # Compute forward return — shift(-N) gives the return N bars in the future
    # CRITICAL: shift(-N) moves future data to current row — this is forward-looking.
    # Labels are computed BEFORE session filtering so shift works across sessions.
    # After session filtering, the last N rows will have NaN future_return.
    # Those NaN rows get label=2 (NoTrade) — do NOT drop them.
    df["future_return"] = df["close"].shift(-forward_bars) / df["close"] - 1

    # Default: NoTrade
    df["label"] = 2

    # Active signal bars
    has_signal = pd.Series(False, index=df.index)
    for col in signal_cols:
        has_signal |= (df[col] != 0)

    long_signal  = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)
    for col in signal_cols:
        long_signal  |= (df[col] == 1)
        short_signal |= (df[col] == -1)

    # Long signal bars that won
    df.loc[long_signal  & (df["future_return"] > 0), "label"] = 0
    # Short signal bars that won
    df.loc[short_signal & (df["future_return"] < 0), "label"] = 1
    # All other signal bars (lost or timed out): already labeled 2 (NoTrade)
    # All non-signal bars: already labeled 2 (NoTrade)

    # IMPORTANT: if both a long and short signal are active on the same bar
    # (two strategies in the model disagree on direction), force label = 2
    conflict = long_signal & short_signal
    df.loc[conflict, "label"] = 2

    # NaN future_return rows (last N bars) stay as label=2 (NoTrade)
    # Do NOT drop these rows — they are valid training rows
    df["label"] = df["label"].fillna(2).astype(int)

    return df
```

**Verify label distribution after assignment:**

```python
label_counts = df["label"].value_counts(normalize=True)
print("Label distribution:")
print(f"  Long (0):    {label_counts.get(0, 0):.2%}")
print(f"  Short (1):   {label_counts.get(1, 0):.2%}")
print(f"  NoTrade (2): {label_counts.get(2, 0):.2%}")
# Expected: NoTrade ~95-98%, Long/Short split remainder
# If NoTrade < 90%: signal filter is still active somewhere
# If Long/Short == 0: label assignment is broken
assert label_counts.get(2, 0) > 0.90, "NoTrade < 90% — signal-bar gating may still be active"
assert label_counts.get(0, 0) > 0.001, "No Long labels assigned"
assert label_counts.get(1, 0) > 0.001, "No Short labels assigned"
```

---

## Task 4 — Update TRAINING_JOBS / Model Loop in train.py

Replace the 8-strategy loop with the 4-group MODEL_GROUPS config. Each model trains once, producing one checkpoint and one scaler.

Key changes:
- Model input: all feature columns from the parquet (including all 9 signal columns as features). The signal columns for OTHER models are still inputs — they provide market context.
- Each model's scaler is fit on that model's fold's train split using ALL feature columns
- Each model's checkpoint is saved as `best_model_{model_name}.pt` (e.g., `best_model_model_1.pt`)
- Each model's scaler is saved as `scaler_{model_name}.pkl`
- Each model's eval CSV is saved as `eval_{model_name}.csv`

**Fold config:** Use the 5-fold rolling walk-forward from Agent 3C spec. Add it here so 4B and 3C are not dependent on each other's run order:

```python
WALK_FORWARD_FOLDS = (
    FoldSpec(name="fold_1",
             train_start="2021-03-19", train_end="2023-06-30",
             val_start="2023-07-01",   val_end="2023-12-31",
             test_start="2024-01-01",  test_end="2024-06-30"),
    FoldSpec(name="fold_2",
             train_start="2021-03-19", train_end="2023-12-31",
             val_start="2024-01-01",   val_end="2024-06-30",
             test_start="2024-07-01",  test_end="2024-12-31"),
    FoldSpec(name="fold_3",
             train_start="2021-03-19", train_end="2024-06-30",
             val_start="2024-07-01",   val_end="2024-12-31",
             test_start="2025-01-01",  test_end="2025-06-30"),
    FoldSpec(name="fold_4",
             train_start="2021-03-19", train_end="2024-12-31",
             val_start="2025-01-01",   val_end="2025-06-30",
             test_start="2025-07-01",  test_end="2025-12-31"),
    FoldSpec(name="fold_5",
             train_start="2021-03-19", train_end="2025-06-30",
             val_start="2025-07-01",   val_end="2025-12-31",
             test_start="2026-01-01",  test_end="2026-03-18"),
)
```

---

## Task 5 — Update topstep_risk.py

Replace the canonical `TopStepRiskManager` class with the updated version from `Reference/Prop Firm Rules.md`. Key changes:
- `max_contracts = 50` (was 5)
- Remove `self.contracts = 5` and `self.pnl_per_point` fixed fields
- Add `position_size(stop_pts, confidence)` method
- Update `simulate_trade()` signature to accept `contracts` parameter
- Keep all DD/DLL/consistency logic unchanged

**Do not change the DD/DLL/consistency logic. Only the contract limit and position sizing change.**

---

## Task 6 — Update funded_sim.py Position Sizing

Update `funded_sim.py` to call `risk_manager.position_size(stop_pts, confidence)` instead of using a fixed 5-contract value. Every trade simulation must:
1. Get stop distance in points from the bar's ATR and stop multiplier
2. Call `position_size(stop_pts, confidence)` to get contract count
3. Pass that contract count to `simulate_trade()`

---

## Task 7 — Class Weights Per Model Per Fold

Class weights must be computed from the actual label distribution in that fold's training split, not hardcoded or computed globally.

```python
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse frequency weights for CrossEntropyLoss.
    Computed on train split labels only. Never on val or test.

    With all-bars training, NoTrade (class 2) will be ~95% of labels.
    Without weighting, the model will predict NoTrade for everything.
    """
    counts = np.bincount(labels, minlength=3)
    counts = counts.astype(float)
    counts[counts == 0] = 1  # prevent division by zero for missing classes
    weights = 1.0 / counts
    weights = weights / weights.sum()  # normalize
    return torch.tensor(weights, dtype=torch.float32)
```

Verify weights:
```python
# With 95% NoTrade, 3% Long, 2% Short:
# NoTrade weight ≈ 0.35, Long weight ≈ 11, Short weight ≈ 16
# The model gets ~30x more gradient signal per Long/Short bar vs NoTrade bar
print("Class weights:", weights)
assert weights[2] < weights[0], "NoTrade weight should be lowest — verify inverse frequency"
```

---

## Task 8 — Dry Run Before Full Training

```
python ml/train.py --dry-run --max-epochs 2
```

A dry run should:
- Load each model group's parquet
- Assign labels
- Print training row count per model per fold
- Print label distribution per model
- Train for 2 epochs only
- Not save checkpoints

Verify training row counts are ~90% of expected full-session bar count. If any model shows <10,000 training rows on a 2-year fold, signal-bar gating is still active.

---

## Task 9 — Full Training Run (Overnight)

After dry run passes:
```
python ml/hyperparam_search.py --trials 10
python ml/train.py --retrain-from-hpo
python ml/evaluate.py
python ml/funded_sim.py
```

---

## Task 10 — Tests

Add `ml/tests/test_agent4b.py`:

- `test_four_model_groups_defined` — `len(MODEL_GROUPS) == 4`
- `test_each_model_has_signal_cols` — every MODEL_GROUP entry has non-empty `signal_cols`
- `test_no_signal_bar_filter_in_dataset_builder` — grep dataset_builder.py for signal filtering code and assert none found. Use `subprocess.run(["grep", "-n", "signal.*!= 0", "ml/dataset_builder.py"])` and assert empty output.
- `test_training_row_count_all_bars` — for fold_1 5min parquet, training rows > 40,000 (well above the 750 per-strategy count from before)
- `test_notrade_label_dominates` — label distribution has NoTrade > 90%
- `test_long_short_labels_nonzero` — both Long and Short labels appear in training data
- `test_conflict_bars_labeled_notrade` — bars where both a long and short signal are active get label=2
- `test_class_weights_inverse_frequency` — NoTrade weight is strictly less than Long and Short weights
- `test_max_contracts_50` — TopStepRiskManager.max_contracts == 50
- `test_position_size_respects_ceiling` — position_size(1.0, 0.99) == 50 (caps at max)
- `test_position_size_scales_with_confidence` — position_size(10, 0.60) < position_size(10, 0.80)
- `test_position_size_scales_with_stop` — position_size(5, 0.80) > position_size(20, 0.80) (tighter stop = more contracts)

---

## Task 11 — Write AGENT4B_STATUS.md

```markdown
## Completed
- [tasks]

## Model Groups
| Model | Strategies | Signal Cols | Parquet |
|---|---|---|---|

## Training Row Counts (per model, per fold)
| Model | Fold | Rows | Expected | % of Expected |
|---|---|---|---|---|
(every model, every fold — must be >80% of expected)

## Label Distribution (per model)
| Model | Long% | Short% | NoTrade% |
(NoTrade must be >90%)

## Position Sizing
- max_contracts: 50
- confidence tiers: [paste table]

## Fold Configuration
- 5 folds confirmed

## Test Suite
- Before: N passed
- After: N passed
- New tests: N

## Known Issues
- [any]

## Next Agent
- Agent 4B-Audit reviews this. After approval: 3C → 3C-Audit → 3D → 3D-Audit → 3E → 3E-Audit
```

---

## Deliverables

```
ml/
├── train.py              ← MODEL_GROUPS config, 5-fold walk-forward, all-bars training
├── dataset_builder.py    ← no signal-bar gating, all-bars label assignment
├── topstep_risk.py       ← max_contracts=50, position_size() method
├── funded_sim.py         ← confidence-based position sizing
├── tests/test_agent4b.py ← 12 new tests
├── artifacts/
│   ├── best_model_model_1.pt through best_model_model_4.pt
│   ├── scaler_model_1.pkl through scaler_model_4.pkl
│   └── eval_model_1.csv through eval_model_4.csv
└── AGENT4B_STATUS.md
```

---

## Logic Gaps to Guard Against

1. **Signal columns as features vs as label targets.** ALL 9 signal columns remain INPUT FEATURES to every model. The `signal_cols` in MODEL_GROUPS only determines which signals generate labels for that model. A model may still see `orb_vol_signal` as a feature even if it is not responsible for that signal's labels.

2. **Conflict bars.** When two strategies in the same model fire in opposite directions on the same bar, label must be 2 (NoTrade). Do not arbitrarily pick one direction's label.

3. **Scaler fit on train split only.** With all-bars training, the scaler must still be fit exclusively on the train split rows. The larger row count does not change this requirement.

4. **Class weights computed per model per fold.** With different parquets and different signal densities, each model group will have a different label distribution. Do not share weights across models or folds.

5. **Checkpoint naming.** Old code saved `best_model_ifvg.pt`, `best_model_connors.pt` etc. New names are `best_model_model_1.pt` through `best_model_model_4.pt`. The old 8 checkpoints remain in `artifacts/` as historical references — do not delete them, but do not load them for any new training run.

6. **Old eval CSVs.** Old `eval_ifvg.csv` etc. remain as historical references. New evaluation writes `eval_model_1.csv` through `eval_model_4.csv`.

7. **position_size() with zero stop.** If stop_pts is 0 or negative (data error), the function must raise ValueError, not divide by zero. The caller must handle this and skip the trade.

8. **Dry run must not save checkpoints.** If `--dry-run` flag is set, no files are written to `artifacts/`. Verify this before running overnight.

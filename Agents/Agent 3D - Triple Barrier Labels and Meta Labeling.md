# Agent 3D — Triple-Barrier Labels

**Phase:** Label wiring
**Prerequisites:**
- Agent 3C complete and 3C-Audit APPROVED (`ml/AGENT3C_AUDIT.md` shows "3C APPROVED")
- 5 rolling folds in place in `train.py`
- `MODEL_GROUPS` in `train.py`
- Non-signal bars must remain in the dataset — never filter them out

---

## Execution Constraint — CODE COMPLETE, DO NOT EXECUTE

> **Write all code changes completely. Do NOT run pytest. Do NOT run any training. Do NOT execute any Python.** Mark your status file as "CODE COMPLETE — NOT EXECUTED". If you find yourself about to run a command, stop.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\ml\labels.py` — read in full
2. `C:\Users\Luker\strategyLabbrain\ml\train.py` — focus on `_load_strategy_frame()` and label handling
3. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — focus on how `label` column is computed
4. `C:\Users\Luker\strategyLabbrain\ml\tests\test_agent3d.py` — read existing tests
5. `C:\Users\Luker\strategyLabbrain\ml\AGENT3C_STATUS.md`

---

## What Already Exists — Do Not Rebuild

**`ml/labels.py`** — fully implemented. Contains:
- `triple_barrier_label()` — computes stop/target/time barrier labels per signal bar
- Session boundary logic (exits at 15:00 ET, not beyond session date)
- Output columns: `label`, `exit_bar`, `exit_time`, `exit_price`, `r_multiple`, `barrier_hit`

**Feature parquets** — already contain per-strategy triple-barrier columns:
- `label_ifvg`, `label_orb_ib`, `label_session_pivot`, etc.
- `barrier_hit_*`, `r_multiple_*`, `exit_price_*` per strategy

---

## Fix Tasks

### Task 1 — Wire triple-barrier labels into training

In `train.py`, find `_load_strategy_frame()`. Currently it uses the generic `label` column (which is the old shift(-N) label).

Change it so training uses the per-strategy triple-barrier label column instead:
- For strategy `ifvg`, use `label_ifvg` column
- For strategy `session_pivot`, use `label_session_pivot` column
- Pattern: `label_{strategy_name}`

The per-strategy label column contains:
- `1` = win (target hit first)
- `0` = loss (stop hit first)
- `NaN` = non-signal bar

Non-signal bars must be labeled `NoTrade (class 2)`, not dropped. After loading the per-strategy label column, fill NaN rows with `2`.

The final `label` column used for training must be:
```python
frame["label"] = frame[f"label_{strategy_name}"].fillna(2).astype(int)
```

### Task 2 — Verify transaction cost in labels

In `labels.py`, find `triple_barrier_label()`. Confirm `transaction_cost_pts=0.07` is applied to the vertical barrier exit P&L. This ensures a flat price path (no movement) produces a loss, not a break-even.

If the cost is not applied to the vertical barrier exit, add it.

### Task 3 — Verify NoTrade handling in build_window_batch

In `train.py`, find `build_window_batch()`. Confirm that ALL bars (including NoTrade bars) are included in the window batch — not just signal bars. The label for NoTrade bars must be `2`.

If signal-bar filtering exists inside `build_window_batch`, remove it.

### Task 4 — Verify class weights account for NoTrade imbalance

In `train.py`, find where class weights are computed. Since ~95% of bars are NoTrade, class weights must compensate. Confirm the weight for class 2 (NoTrade) is lower than for class 0 (Long) and class 1 (Short).

If class weights are not computed or are equal, add inverse-frequency weighting:
```python
counts = np.bincount(train_labels, minlength=3).astype(float)
counts = np.where(counts == 0, 1.0, counts)
weights = 1.0 / counts
weights = weights / weights.sum()
```

### Task 5 — Update tests

In `ml/tests/test_agent3d.py`, review existing tests. Add or update tests to cover:
- `test_stop_hit_first_labels_loss` — synthetic path where price hits stop → label=0
- `test_target_hit_first_labels_win` — synthetic path where price hits target → label=1
- `test_nonsignal_bar_gets_notrade_label` — bar with signal=0 → label=2 after fillna
- `test_session_boundary_exits_at_1500` — signal at 14:45, max_bars=60 → exit at 15:00
- `test_transaction_cost_flat_path_is_loss` — flat price path → negative PnL → label=0

---

## Status File

Write `ml/AGENT3D_STATUS.md`:
```markdown
# Agent 3D Status

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

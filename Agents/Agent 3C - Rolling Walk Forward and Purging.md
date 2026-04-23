# Agent 3C — Rolling Walk-Forward + Purging

**Phase:** Walk-forward review and wiring
**Prerequisites:**
- Agent 4A complete and 4A-Audit APPROVED (`ml/AGENT4A_AUDIT.md` shows "4A APPROVED")
- Agent 4B complete and 4B-Audit APPROVED (`ml/AGENT4B_AUDIT.md` shows "4B APPROVED")

Do not start until `AGENT4B_AUDIT.md` shows "4B APPROVED".

---

## Execution Constraint — CODE COMPLETE, DO NOT EXECUTE

> **Write all code changes completely. Do NOT run pytest. Do NOT run any training. Do NOT execute any Python.** Mark your status file as "CODE COMPLETE — NOT EXECUTED". If you find yourself about to run a command, stop.

---

## Context to Read First (in order)

1. `C:\Users\Luker\strategyLabbrain\MASTER_CHANGE_PLAN.md`
2. `C:\Users\Luker\strategyLabbrain\ml\train.py` — read in full
3. `C:\Users\Luker\strategyLabbrain\ml\dataset_builder.py` — read the purge/embargo section
4. `C:\Users\Luker\strategyLabbrain\ml\tests\test_agent3c.py` — read existing tests
5. `C:\Users\Luker\strategyLabbrain\ml\AGENT4B_STATUS.md`

---

## What Already Exists — Do Not Rebuild

The following are already implemented and correct. Read them to understand the code, then move to the Fix Tasks.

**Already correct in `train.py`:**
- `WALK_FORWARD_FOLDS` — 5 rolling folds with correct dates (fold_1 through fold_5)
- `apply_purge_embargo` imported and called in the fold-split logic
- `FoldSpec` dataclass

**Already correct in `dataset_builder.py`:**
- `apply_purge_embargo()` function
- `embargo_bars_for_timeframe()` function

---

## Fix Tasks

### Task 1 — Add session_pivot to strategy maps

In `train.py`, find `STRATEGY_TIMEFRAME_MAP` and `STRATEGY_SIGNAL_COLUMN_MAP`. Both are missing `session_pivot`.

Add to `STRATEGY_TIMEFRAME_MAP`:
```python
"session_pivot": "5min",
```

Add to `STRATEGY_SIGNAL_COLUMN_MAP`:
```python
"session_pivot": "session_pivot_signal",
```

### Task 2 — Verify MODEL_GROUPS wiring

Agent 4B will have added `MODEL_GROUPS` to `train.py`. Verify it contains all 4 groups and all 9 signal columns including `session_pivot_signal`. If `MODEL_GROUPS` is missing or incomplete, add it:

```python
MODEL_GROUPS = [
    {
        "model_name": "model_1",
        "signal_cols": ["ifvg_signal", "connors_signal"],
        "timeframe": "5min",
    },
    {
        "model_name": "model_2",
        "signal_cols": ["ifvg_open_signal", "ttm_signal"],
        "timeframe": "5min",
    },
    {
        "model_name": "model_3",
        "signal_cols": ["orb_vol_signal", "session_pivot_signal", "session_pivot_break_signal"],
        "timeframe": "5min",
    },
    {
        "model_name": "model_4",
        "signal_cols": ["orb_ib_signal", "orb_wick_signal"],
        "timeframe": "5min",
    },
]
```

### Task 3 — Verify purge gap logic

In `dataset_builder.py`, read `apply_purge_embargo()` and confirm:
- It removes the last `forward_horizon_bars` rows from the train split
- It skips the first `embargo_bars` rows of the val split
- It does NOT filter rows by signal activity — all bars remain

If any of these are wrong, fix them.

### Task 4 — Verify all-bars guarantee

In `train.py`, find where the training frame is built per fold. Confirm there is no filter like `df[df[signal_col] != 0]` or similar that reduces the training set to signal-only rows.

If any such filter exists, remove it. Every session bar in the fold's train date range must be a training row.

### Task 5 — Update tests

In `ml/tests/test_agent3c.py`, review existing tests. Add or update tests to cover:
- `test_session_pivot_in_strategy_maps` — verifies `session_pivot` in both maps
- `test_model_groups_covers_all_9_signals` — verifies all 9 signal columns covered, no duplicates
- `test_purge_gap_removes_leaking_rows` — verifies gap >= forward_horizon_bars
- `test_no_signal_bar_filter_in_training` — grep-based: confirms no `!= 0` signal filter active in dataset_builder or train

---

## Status File

Write `ml/AGENT3C_STATUS.md`:
```markdown
# Agent 3C Status

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

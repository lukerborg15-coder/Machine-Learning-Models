# Agent 3C Status

## Status: CODE COMPLETE — NOT EXECUTED

## Changes Made
- `ml/dataset_builder.py:527` - added `_drop_embargo_head()` for validation embargo removal from the start of the validation split.
- `ml/dataset_builder.py:535` - updated `apply_purge_embargo()` so train drops only the final `forward_horizon_bars`, validation skips the first `embargo_bars`, and test remains the direct fold slice.
- `ml/tests/test_agent3c.py:5` - added source-reading imports for the grep-style signal-filter test.
- `ml/tests/test_agent3c.py:14` - imported `MODEL_GROUPS`, `STRATEGY_SIGNAL_COLUMN_MAP`, and `STRATEGY_TIMEFRAME_MAP` for Agent 3C coverage tests.
- `ml/tests/test_agent3c.py:23` - added the expected nine grouped signal columns.
- `ml/tests/test_agent3c.py:63` - added `test_session_pivot_in_strategy_maps`.
- `ml/tests/test_agent3c.py:68` - added `test_model_groups_covers_all_9_signals`.
- `ml/tests/test_agent3c.py:80` - updated `test_purge_gap_removes_leaking_rows` for train-tail purge plus validation-head embargo.
- `ml/tests/test_agent3c.py:148` - added `test_no_signal_bar_filter_in_training`.
- `ml/AGENT3C_STATUS.md:1` - replaced the previous executed-run status with this code-complete, not-executed status.

## What Was Already Correct
- `ml/AGENT4B_AUDIT.md` showed `4B APPROVED - proceed to 3C`.
- `ml/train.py:46` already included `session_pivot` in `STRATEGY_TIMEFRAME_MAP`.
- `ml/train.py:58` already included `session_pivot` in `STRATEGY_SIGNAL_COLUMN_MAP`.
- `ml/train.py:70` already defined four `MODEL_GROUPS` covering all nine signal columns, including `session_pivot_signal`.
- `ml/train.py:126` already defined five rolling walk-forward folds.
- `ml/train.py:529` already routed fold construction through `apply_purge_embargo()`.
- `ml/train.py:750` and `ml/train.py:917` already trained from the full fold split frames without active signal-only row filtering.
- `ml/dataset_builder.py:535` already kept purge/embargo logic independent of signal activity; no signal activity filter was added.

## Tests Added/Updated
- `test_session_pivot_in_strategy_maps`
- `test_model_groups_covers_all_9_signals`
- `test_purge_gap_removes_leaking_rows`
- `test_no_signal_bar_filter_in_training`

## Known Issues
- None found in this pass.
- Tests were not run by instruction.
- No training was run by instruction.
- No Python was executed by instruction.

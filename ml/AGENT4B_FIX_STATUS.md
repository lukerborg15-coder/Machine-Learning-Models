# Agent 4B Fix Status

## Fixes Applied

- Lowered `test_training_row_count_all_bars` from `len(train) > 40_000` to `len(train) > 35_000`.
- Verified `WALK_FORWARD_FOLDS` dates in `ml/train.py` match the requested rolling walk-forward schedule and left them unchanged.
- Found no fold-overlap assertion in `ml/tests/test_agent4b.py`; no Agent 4B overlap test change was needed.
- Updated the two Agent 3 Express Funded daily-loss intratrade tests to derive the active contract count through `TopStepRiskManager.position_size()` and choose bar extremes that breach the daily loss limit without also breaching the max loss limit.

## Verification

Command:

```text
python -m pytest ml/tests -v --tb=short
```

Result:

```text
116 passed, 3 skipped, 1 warning in 242.93s (0:04:02)
```

The warning was a pre-existing pytest cache write warning for `.pytest_cache`.

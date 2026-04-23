"""Agent 3C rolling walk-forward, purge/embargo, and gate tests."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from ml.dataset_builder import apply_purge_embargo, embargo_bars_for_timeframe
from ml.evaluate import aggregate_across_folds
from ml.funded_sim import evaluate_deployment_gate
from ml.train import (
    MODEL_GROUPS,
    STRATEGY_SIGNAL_COLUMN_MAP,
    STRATEGY_TIMEFRAME_MAP,
    FoldSpec,
    WALK_FORWARD_FOLDS,
)


EXPECTED_AGENT3C_SIGNAL_COLUMNS = {
    "ifvg_signal",
    "ifvg_open_signal",
    "orb_ib_signal",
    "orb_vol_signal",
    "orb_wick_signal",
    "ttm_signal",
    "connors_signal",
    "session_pivot_signal",
    "session_pivot_break_signal",
}


def test_five_folds_defined() -> None:
    assert len(WALK_FORWARD_FOLDS) == 5
    assert [fold.name for fold in WALK_FORWARD_FOLDS] == ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]


def test_no_fold_test_overlap() -> None:
    windows = []
    for fold in WALK_FORWARD_FOLDS:
        start = pd.Timestamp(fold.test_start)
        end = pd.Timestamp(fold.test_end)
        windows.append((fold.name, start, end))

    for left_idx, (left_name, left_start, left_end) in enumerate(windows):
        for right_name, right_start, right_end in windows[left_idx + 1 :]:
            overlaps = left_start <= right_end and right_start <= left_end
            assert not overlaps, f"{left_name} overlaps {right_name}"


def test_fold_chronology() -> None:
    for fold in WALK_FORWARD_FOLDS:
        train_end = pd.Timestamp(fold.train_end)
        val_start = pd.Timestamp(fold.val_start)
        val_end = pd.Timestamp(fold.val_end)
        test_start = pd.Timestamp(fold.test_start)
        assert train_end < val_start < val_end < test_start


def test_session_pivot_in_strategy_maps() -> None:
    assert STRATEGY_TIMEFRAME_MAP["session_pivot"] == "5min"
    assert STRATEGY_SIGNAL_COLUMN_MAP["session_pivot"] == "session_pivot_signal"


def test_model_groups_covers_all_9_signals() -> None:
    grouped_signals = [
        signal_column
        for group in MODEL_GROUPS
        for signal_column in group["signal_cols"]
    ]

    assert len(MODEL_GROUPS) == 4
    assert set(grouped_signals) == EXPECTED_AGENT3C_SIGNAL_COLUMNS
    assert len(grouped_signals) == len(set(grouped_signals))


def test_purge_gap_removes_leaking_rows() -> None:
    index = pd.date_range("2024-01-01 09:30", periods=50, freq="min", tz="America/New_York")
    frame = pd.DataFrame({"close": range(len(index))}, index=index)
    fold = FoldSpec(
        name="synthetic",
        train_start="2024-01-01 09:30:00",
        train_end="2024-01-01 09:49:00",
        val_start="2024-01-01 09:50:00",
        val_end="2024-01-01 10:09:00",
        test_start="2024-01-01 10:10:00",
        test_end="2024-01-01 10:20:00",
    )

    split = apply_purge_embargo(frame, fold, forward_horizon_bars=5, embargo_bars=2)

    assert split["train"].index.max() == index[14]
    assert split["val"].index.min() == index[22]

    row_positions = pd.Series(range(len(index)), index=index)
    gap_bars = (
        row_positions.loc[split["val"].index.min()]
        - row_positions.loc[split["train"].index.max()]
        - 1
    )
    assert gap_bars >= 5

    label_timestamps = pd.Series(frame.index, index=frame.index).shift(-5).reindex(split["train"].index)
    val_start = pd.Timestamp(fold.val_start, tz="America/New_York")
    val_end = pd.Timestamp(fold.val_end, tz="America/New_York")
    assert not label_timestamps.between(val_start, val_end, inclusive="both").any()


def test_embargo_size_matches_timeframe() -> None:
    # TopStep session: 09:30-15:00 ET = 330 minutes (not 390 — flattens at 15:00, not 16:00)
    expected = {"1min": 330, "3min": 110, "5min": 66}
    for tf, expected_bars in expected.items():
        actual = embargo_bars_for_timeframe(tf)
        assert actual == expected_bars, f"{tf} embargo={actual}, expected {expected_bars}"


def test_no_signal_bar_filter_in_code() -> None:
    root = Path(__file__).parent.parent
    files_to_check = [root / "dataset_builder.py", root / "train.py"]
    filter_pattern = re.compile(r"\[df\[.*signal.*\]\s*!=\s*0\]|\.loc\[.*signal.*!=\s*0")
    for filepath in files_to_check:
        lines = filepath.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            assert not filter_pattern.search(line), (
                f"Active signal-bar filter found at {filepath.name}:{i}: {line.rstrip()}"
            )

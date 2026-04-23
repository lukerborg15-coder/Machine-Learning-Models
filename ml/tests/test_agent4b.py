"""Agent 4B grouped-model and all-bars training tests."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from ml.topstep_risk import TopStepRiskManager
from ml.train import MODEL_GROUPS, WALK_FORWARD_FOLDS, assign_labels, compute_class_weights


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "ml" / "data" / "features_mnq_5min.parquet"


def _model_group(name: str) -> dict[str, object]:
    return next(group for group in MODEL_GROUPS if group["model_name"] == name)


def _timestamp(value: str, index: pd.DatetimeIndex, end_of_day: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if end_of_day and timestamp.time() == pd.Timestamp("00:00").time():
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if index.tz is not None:
        return timestamp.tz_localize(index.tz) if timestamp.tzinfo is None else timestamp.tz_convert(index.tz)
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp


def _fold1_train_frame() -> pd.DataFrame:
    frame = pd.read_parquet(DATA_PATH).sort_index()
    fold = WALK_FORWARD_FOLDS[0]
    start = _timestamp(fold.train_start, frame.index)
    end = _timestamp(fold.train_end, frame.index, end_of_day=True)
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def test_four_model_groups_defined() -> None:
    assert len(MODEL_GROUPS) == 4


def test_each_model_has_signal_cols() -> None:
    for group in MODEL_GROUPS:
        assert group["signal_cols"]


def test_no_signal_bar_filter_in_dataset_builder() -> None:
    path = ROOT_DIR / "ml" / "dataset_builder.py"
    try:
        completed = subprocess.run(
            ["grep", "-n", "signal.*!= 0", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == ""
    except FileNotFoundError:
        assert re.search(r"signal.*!= 0", path.read_text(encoding="utf-8")) is None


def test_training_row_count_all_bars() -> None:
    train = _fold1_train_frame()
    assert len(train) > 35_000


def test_notrade_label_dominates() -> None:
    group = _model_group("model_3")
    labeled = assign_labels(_fold1_train_frame(), group["signal_cols"])
    distribution = labeled["label"].value_counts(normalize=True)
    assert distribution.get(2, 0.0) > 0.90


def test_long_short_labels_nonzero() -> None:
    group = _model_group("model_3")
    labeled = assign_labels(_fold1_train_frame(), group["signal_cols"])
    counts = labeled["label"].value_counts()
    assert counts.get(0, 0) > 0
    assert counts.get(1, 0) > 0


def test_conflict_bars_labeled_notrade() -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "long_signal": [1, 0, 0, 0, 0, 0],
            "short_signal": [-1, 0, 0, 0, 0, 0],
        }
    )
    labeled = assign_labels(frame, ["long_signal", "short_signal"], forward_bars=1)
    assert labeled.loc[0, "label"] == 2


def test_class_weights_inverse_frequency() -> None:
    labels = np.array([0] * 30 + [1] * 20 + [2] * 950, dtype=int)
    weights = compute_class_weights(labels)
    assert float(weights[2]) < float(weights[0])
    assert float(weights[2]) < float(weights[1])


def test_max_contracts_50() -> None:
    assert TopStepRiskManager().max_contracts == 50


def test_position_size_respects_ceiling() -> None:
    assert TopStepRiskManager().position_size(1.0, 0.99) == 50


def test_position_size_scales_with_confidence() -> None:
    risk = TopStepRiskManager()
    assert risk.position_size(10.0, 0.60) < risk.position_size(10.0, 0.80)


def test_position_size_scales_with_stop() -> None:
    risk = TopStepRiskManager()
    assert risk.position_size(5.0, 0.80) > risk.position_size(20.0, 0.80)

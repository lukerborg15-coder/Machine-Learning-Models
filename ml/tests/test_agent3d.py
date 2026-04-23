"""Agent 3D triple-barrier and meta-label tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import simulate_trading
from ml.labels import triple_barrier_label
from ml.train import TrainingJobSpec, _load_strategy_frame, compute_binary_pos_weight


def _price_frame(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    start: str = "2025-01-02 09:30",
) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close), freq="5min", tz="America/New_York", name="datetime")
    close_series = pd.Series(close, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close_series,
            "high": high if high is not None else (close_series + 0.1).tolist(),
            "low": low if low is not None else (close_series - 0.1).tolist(),
            "close": close_series,
            "volume": np.full(len(close), 1_000.0),
        },
        index=index,
    )


def _load_frame_job_spec(parquet_path: Path) -> TrainingJobSpec:
    return TrainingJobSpec(
        strategy_name="model_1",
        timeframe="5min",
        parquet_path=str(parquet_path),
        artifact_stem="model_1",
        strategies=("ifvg",),
        signal_cols=("ifvg_signal",),
    )


def test_load_strategy_frame_uses_triple_barrier(tmp_path: Path) -> None:
    parquet_path = tmp_path / "features.parquet"
    frame = _price_frame([100.0, 101.0, 102.0])
    frame["ifvg_signal"] = [1, -1, 0]
    frame["label_ifvg"] = [0.0, 1.0, np.nan]
    frame.to_parquet(parquet_path)

    loaded = _load_strategy_frame(_load_frame_job_spec(parquet_path))

    assert loaded["label"].tolist() == [0, 1, 2]


def test_load_strategy_frame_raises_if_label_col_missing(tmp_path: Path) -> None:
    parquet_path = tmp_path / "features.parquet"
    frame = _price_frame([100.0, 101.0, 102.0])
    frame["ifvg_signal"] = [1, -1, 0]
    frame.to_parquet(parquet_path)

    with pytest.raises(KeyError, match="label_ifvg"):
        _load_strategy_frame(_load_frame_job_spec(parquet_path))


def test_triple_barrier_stop_hit_first() -> None:
    frame = _price_frame([100.0, 99.0, 98.0, 97.0, 96.0])
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(2.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr)

    assert labels["label"].iloc[0] == 0
    assert labels["barrier_hit"].iloc[0] == "stop"


def test_triple_barrier_target_hit_first() -> None:
    frame = _price_frame([100.0, 101.0, 102.0, 103.0, 104.0])
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(2.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr)

    assert labels["label"].iloc[0] == 1
    assert labels["barrier_hit"].iloc[0] == "target"


def test_triple_barrier_vertical_timeout() -> None:
    frame = _price_frame([100.0, 100.0, 100.0, 100.0, 100.0])
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(10.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=4, transaction_cost_pts=0.07)

    assert labels["label"].iloc[0] == 0
    assert labels["barrier_hit"].iloc[0] == "vertical"


def test_triple_barrier_short_symmetry() -> None:
    frame = _price_frame([100.0, 99.0, 98.0], low=[99.9, 98.9, 96.9])
    signal = pd.Series([-1, 0, 0], index=frame.index)
    atr = pd.Series(2.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr)

    assert labels["label"].iloc[0] == 1
    assert labels["barrier_hit"].iloc[0] == "target"


def test_triple_barrier_respects_atr_scaling() -> None:
    frame = _price_frame([100.0, 90.0], low=[99.9, 90.0])
    signal = pd.Series([1, 0], index=frame.index)
    labels_atr_1 = triple_barrier_label(frame, signal, pd.Series(1.0, index=frame.index))
    labels_atr_2 = triple_barrier_label(frame, signal, pd.Series(2.0, index=frame.index))

    assert 100.0 - labels_atr_2["exit_price"].iloc[0] == pytest.approx(
        2.0 * (100.0 - labels_atr_1["exit_price"].iloc[0])
    )


def test_meta_label_only_on_signal_bars() -> None:
    frame = _price_frame([100.0, 101.0, 102.0, 103.0])
    signal = pd.Series([0, 1, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr)

    assert labels["label"].notna().sum() == 1
    assert pd.isna(labels["label"].iloc[0])
    assert pd.notna(labels["label"].iloc[1])


def test_two_class_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from ml.model import TradingCNN

    model = TradingCNN(n_features=35, seq_len=30, n_classes=2)
    out = model(torch.zeros(4, 30, 35))

    assert out.shape == (4, 2)


def test_bce_loss_class_imbalance_weighted() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    pos_weight = compute_binary_pos_weight(labels)

    assert pos_weight == pytest.approx(2.0)
    assert pos_weight > 1.5


def test_confidence_threshold_filters_trades() -> None:
    rows: list[dict[str, float | int]] = []
    index: list[pd.Timestamp] = []
    for day_idx, confidence in enumerate([0.51, 0.65, 0.75]):
        day = pd.Timestamp("2025-01-02", tz="America/New_York") + pd.Timedelta(days=day_idx)
        index.extend([day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=9, minutes=35)])
        rows.extend(
            [
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "atr": 2.0,
                    "orb_wick_signal": 1,
                    "prediction": 1,
                    "confidence": confidence,
                    "label": 1,
                },
                {
                    "open": 103.0,
                    "high": 103.0,
                    "low": 103.0,
                    "close": 103.0,
                    "atr": 2.0,
                    "orb_wick_signal": 0,
                    "prediction": 0,
                    "confidence": 0.0,
                    "label": 0,
                },
            ]
        )
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="datetime"))

    low_threshold = simulate_trading(
        frame,
        "orb_wick",
        signal_column="orb_wick_signal",
        objective="meta_label",
        confidence_threshold=0.50,
    )
    high_threshold = simulate_trading(
        frame,
        "orb_wick",
        signal_column="orb_wick_signal",
        objective="meta_label",
        confidence_threshold=0.70,
    )

    assert high_threshold["trade_count"] <= low_threshold["trade_count"]
    assert low_threshold["trade_count"] == 3
    assert high_threshold["trade_count"] == 1


def test_barrier_label_matches_trade_sim() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:30",
            "2025-01-02 09:35",
            "2025-01-03 09:30",
            "2025-01-03 09:35",
        ],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 103.0, 100.0, 97.0],
            "high": [100.5, 103.5, 100.5, 97.5],
            "low": [99.5, 102.5, 99.5, 96.5],
            "close": [100.0, 103.0, 100.0, 97.0],
            "atr": [2.0, 2.0, 2.0, 2.0],
            "orb_wick_signal": [1, 0, 1, 0],
            "prediction": [1, 0, 1, 0],
            "confidence": [1.0, 0.0, 1.0, 0.0],
        },
        index=index,
    )
    labels = triple_barrier_label(frame, frame["orb_wick_signal"], frame["atr"])
    result = simulate_trading(
        frame,
        "orb_wick",
        signal_column="orb_wick_signal",
        objective="meta_label",
        confidence_threshold=0.50,
    )

    expected = labels.loc[frame["orb_wick_signal"] != 0, "label"].astype(int).tolist()
    actual = (result["trades"]["pnl"] > 0).astype(int).tolist()

    assert actual == expected

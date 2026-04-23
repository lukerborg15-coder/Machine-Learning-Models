"""Agent 1A and Agent 1B pipeline tests."""

from __future__ import annotations

from datetime import time
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from Implementation.camarilla_pivot_generator import compute_camarilla
from ml.dataset_builder import (
    build_feature_matrix,
    compute_labels,
    compute_ohlcv_features,
    compute_pivot_features,
    load_data,
)
from ml.signal_generators import (
    ifvg_combined,
    orb_initial_balance,
    orb_volume_adaptive,
    orb_wick_rejection,
)
from ml.topstep_risk import TopStepRiskManager
from ml.train import (
    SimpleStandardScaler,
    build_temporal_splits,
    build_window_batch,
    compute_class_weights,
)


@lru_cache(maxsize=None)
def _session_data(timeframe: str) -> pd.DataFrame:
    return load_data("mnq", timeframe, session_only=True)


@lru_cache(maxsize=None)
def _raw_data(timeframe: str) -> pd.DataFrame:
    return load_data("mnq", timeframe, session_only=False)


@lru_cache(maxsize=None)
def _feature_matrix(timeframe: str) -> pd.DataFrame:
    return build_feature_matrix("mnq", timeframe)


def _daily_signal_counts(signals: pd.Series) -> pd.Series:
    active = (signals != 0).astype(int)
    return active.groupby(active.index.date).sum()


def test_timezone_is_eastern() -> None:
    df = _session_data("5min")
    assert str(df.index.tz) == "America/New_York"


def test_session_hours_only() -> None:
    df = _session_data("5min")
    times = df.index.time
    assert min(times) == time(9, 30)
    assert max(times) == time(15, 0)
    assert not df.between_time("15:05", "23:59").any(axis=None)


def test_no_duplicate_timestamps() -> None:
    df = _session_data("5min")
    assert df.index.is_unique


def test_strategy_signal_no_lookahead() -> None:
    df = _session_data("5min").iloc[:800].copy()
    baseline = orb_wick_rejection(df)

    mutate_loc = df.index[500]
    mutated = df.copy()
    mutated.at[mutate_loc, "open"] = mutated.at[mutate_loc, "open"] * 1.05
    mutated.at[mutate_loc, "high"] = mutated.at[mutate_loc, "high"] * 1.08
    mutated.at[mutate_loc, "low"] = mutated.at[mutate_loc, "low"] * 0.92
    mutated.at[mutate_loc, "close"] = mutated.at[mutate_loc, "close"] * 1.04
    mutated.at[mutate_loc, "volume"] = mutated.at[mutate_loc, "volume"] * 2

    updated = orb_wick_rejection(mutated)
    earlier = baseline.index < mutate_loc
    pd.testing.assert_series_equal(baseline.loc[earlier], updated.loc[earlier], check_names=False)


def test_max_signals_per_day() -> None:
    df = _session_data("5min")

    wick_counts = _daily_signal_counts(orb_wick_rejection(df))
    volume_counts = _daily_signal_counts(orb_volume_adaptive(df))
    ib_counts = _daily_signal_counts(orb_initial_balance(df))

    assert (wick_counts <= 1).all()
    assert (volume_counts <= 1).all()
    assert (ib_counts <= 1).all()


def test_ifvg_shared_daily_limit() -> None:
    df = _session_data("5min")
    base, open_variant = ifvg_combined(df, timeframe_minutes=5)

    combined = (base != 0).astype(int) + (open_variant != 0).astype(int)
    daily_totals = combined.groupby(combined.index.date).sum()

    assert (daily_totals <= 2).all()


def test_synthetic_delta_no_division_by_zero() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="5min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": np.full(20, 100.0),
            "low": np.full(20, 100.0),
            "close": np.full(20, 100.0),
            "volume": np.linspace(1_000, 2_000, 20),
        },
        index=idx,
    )

    features = compute_ohlcv_features(df)
    assert (features["synthetic_delta"] == 0.0).all()
    assert np.isfinite(features["synthetic_delta"]).all()


def test_session_feature_fallback_excludes_unavailable_extended_hours_columns() -> None:
    pivot_features = compute_pivot_features(_session_data("5min"))

    for missing_column in (
        "asia_high_dist",
        "asia_low_dist",
        "london_high_dist",
        "london_low_dist",
        "premarket_high_dist",
        "premarket_low_dist",
    ):
        assert missing_column not in pivot_features.columns

    assert "ny_am_high_dist" in pivot_features.columns
    assert "ny_am_low_dist" in pivot_features.columns


def test_no_nan_in_features_after_warmup() -> None:
    matrix = _feature_matrix("5min")
    feature_columns = matrix.attrs["feature_columns"]
    assert not matrix[feature_columns].isna().any().any()
    assert not matrix["label"].isna().any()


def test_forward_return_uses_future_bar() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=20, freq="1min", tz="America/New_York")
    close = pd.Series(np.arange(100.0, 120.0), index=idx)
    df = pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(20, 1_000.0),
        }
    )

    labels = compute_labels(df, n_forward=1)
    expected = df["close"].iloc[1] / df["close"].iloc[0] - 1.0
    assert abs(labels["future_return"].iloc[0] - expected) < 1e-12


def test_camarilla_uses_prior_day() -> None:
    df = _session_data("5min")
    levels = compute_camarilla(df)

    for _, group in levels.groupby(levels.index.date):
        h3_values = group["H3"].dropna()
        if not h3_values.empty:
            assert h3_values.nunique() == 1


def test_camarilla_correct_prior_day() -> None:
    df = _session_data("5min")
    levels = compute_camarilla(df)
    dates = sorted(levels.index.normalize().unique())

    for i in range(1, min(len(dates), 10)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_day = df[df.index.normalize() == prev_date]
        curr_day = levels[levels.index.normalize() == curr_date]
        if len(prev_day) == 0 or len(curr_day) == 0:
            continue
        expected_h3 = prev_day["close"].iloc[-1] + (prev_day["high"].max() - prev_day["low"].min()) * 0.275
        actual_h3 = curr_day["H3"].iloc[0]
        if pd.notna(actual_h3):
            assert abs(actual_h3 - expected_h3) < 1e-6


def test_temporal_split_boundaries_do_not_overlap() -> None:
    matrix = _feature_matrix("5min")

    train = matrix[matrix.index <= pd.Timestamp("2024-12-31 23:59:59", tz="America/New_York")]
    val = matrix[
        (matrix.index >= pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York"))
        & (matrix.index <= pd.Timestamp("2025-12-31 23:59:59", tz="America/New_York"))
    ]
    test = matrix[matrix.index >= pd.Timestamp("2026-01-01 00:00:00", tz="America/New_York")]

    assert not train.empty
    assert not val.empty
    assert not test.empty
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_scaler_style_guard_uses_train_statistics_only() -> None:
    matrix = _feature_matrix("5min")
    feature_columns = matrix.attrs["feature_columns"]

    train = matrix.loc[matrix.index < pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York"), feature_columns]
    val = matrix.loc[
        (matrix.index >= pd.Timestamp("2025-01-01 00:00:00", tz="America/New_York"))
        & (matrix.index < pd.Timestamp("2026-01-01 00:00:00", tz="America/New_York")),
        feature_columns
    ]

    train_std = train.std(ddof=0).replace(0.0, 1.0)
    val_std = val.std(ddof=0).replace(0.0, 1.0)

    val_scaled_by_train = (val - train.mean()) / train_std
    val_scaled_by_val = (val - val.mean()) / val_std

    assert not np.allclose(val_scaled_by_train.to_numpy(), val_scaled_by_val.to_numpy(), equal_nan=True)


def test_model_output_shape() -> None:
    torch = pytest.importorskip("torch")
    from ml.model import TradingCNN

    model = TradingCNN(n_features=35, seq_len=30)
    dummy = torch.zeros(2, 30, 35)
    out = model(dummy)

    assert out.shape == (2, 3)


def test_causal_padding_no_leakage() -> None:
    torch = pytest.importorskip("torch")
    from ml.model import TradingCNN

    model = TradingCNN(n_features=4, seq_len=10, n_filters=4, n_layers=2, dropout=0.0)
    model.eval()

    base = torch.zeros(1, 10, 4)
    mutated = base.clone()
    mutated[:, 7:, :] = 5.0

    with torch.no_grad():
        base_conv = model.conv(base.transpose(1, 2))
        mutated_conv = model.conv(mutated.transpose(1, 2))

    assert torch.allclose(base_conv[:, :, :7], mutated_conv[:, :, :7])


def test_no_shuffle_in_time_split() -> None:
    splits = build_temporal_splits(_feature_matrix("5min"))

    for split in splits.values():
        train = split["train"]
        val = split["val"]
        test = split["test"]

        assert not train.empty
        assert not val.empty
        assert not test.empty
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()


def test_scaler_fit_only_on_train() -> None:
    matrix = _feature_matrix("5min")
    splits = build_temporal_splits(matrix)
    feature_columns = matrix.attrs["feature_columns"]
    first_fold = splits["fold_1"]

    train = first_fold["train"]
    val = first_fold["val"]

    train_scaler = SimpleStandardScaler.fit(train, feature_columns)
    full_scaler = SimpleStandardScaler.fit(pd.concat([train, val]), feature_columns)

    train_scaled_val = train_scaler.transform(val)
    full_scaled_val = full_scaler.transform(val)

    assert not np.allclose(train_scaled_val, full_scaled_val, equal_nan=True)


def test_window_batch_includes_all_eligible_bars() -> None:
    matrix = _feature_matrix("5min").iloc[:500].copy()
    feature_columns = matrix.attrs["feature_columns"]
    raw_signal = matrix["orb_wick_signal"].copy()

    scaled = matrix.copy()
    scaled["orb_wick_signal"] = 99.0

    batch = build_window_batch(
        scaled,
        feature_columns=feature_columns,
        signal_column="orb_wick_signal",
        seq_len=30,
        signal_values=raw_signal,
    )

    assert len(batch) == len(matrix) - 29
    np.testing.assert_array_equal(batch.raw_signals, raw_signal.iloc[29:].to_numpy(dtype=int))


def test_class_weights_not_uniform() -> None:
    labels = np.array([0, 2, 2, 2, 2, 2, 1, 2, 2, 2], dtype=int)
    weights = compute_class_weights(labels)

    assert not np.allclose(weights, weights[0])
    assert weights[0] > weights[2]
    assert weights[1] > weights[2]


def test_topstep_trailing_dd_from_eod_peak() -> None:
    risk = TopStepRiskManager()
    risk.update_eod(52000, 2000)

    assert risk.active

    risk.update_eod(49950, -2050)
    assert not risk.active


def test_topstep_consistency_rule() -> None:
    risk = TopStepRiskManager()
    risk.daily_pnls = [1400.0, 1000.0, 1000.0]

    assert not risk.check_consistency()

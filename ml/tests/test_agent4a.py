"""Agent 4A pivot feature and Session Level Pivot signal tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Implementation.camarilla_pivot_generator import compute_camarilla
from ml.dataset_builder import compute_pivot_features
from ml.signal_generators import session_pivot_signal


def _two_day_ohlcv() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:30",
            "2025-01-02 15:55",
            "2025-01-02 16:00",
            "2025-01-03 09:30",
            "2025-01-03 09:35",
            "2025-01-03 16:00",
        ],
        tz="America/New_York",
        name="datetime",
    )
    return pd.DataFrame(
        {
            "open": [100.0, 103.0, 101.0, 120.0, 121.0, 122.0],
            "high": [106.0, 108.0, 104.0, 150.0, 160.0, 170.0],
            "low": [96.0, 95.0, 97.0, 119.0, 118.0, 117.0],
            "close": [102.0, 101.0, 100.0, 121.0, 122.0, 123.0],
            "volume": [1000.0] * 6,
        },
        index=index,
    )


def _signal_frame(periods: int = 18) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 09:30", periods=periods, freq="5min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": np.full(periods, 100.0),
            "high": np.full(periods, 101.0),
            "low": np.full(periods, 99.0),
            "close": np.full(periods, 100.0),
            "volume": np.full(periods, 1000.0),
            "atr_14": np.full(periods, 2.0),
            "camarilla_h3": np.full(periods, 104.0),
            "camarilla_h4": np.full(periods, 105.0),
            "camarilla_s3": np.full(periods, 96.0),
            "camarilla_s4": np.full(periods, 95.0),
            "prev_day_high": np.full(periods, 110.0),
            "prev_day_low": np.full(periods, 90.0),
            "prev_day_close": np.full(periods, 100.0),
        },
        index=index,
    )
    return frame


def test_camarilla_h4_above_h3() -> None:
    levels = compute_camarilla(_two_day_ohlcv()).dropna()

    assert (levels["H4"] > levels["H3"]).all()


def test_camarilla_s3_above_s4() -> None:
    levels = compute_camarilla(_two_day_ohlcv()).dropna()

    assert (levels["S3"] > levels["S4"]).all()


def test_camarilla_uses_prior_day() -> None:
    base = _two_day_ohlcv()
    base_levels = compute_camarilla(base)
    current_day = pd.Timestamp("2025-01-03", tz="America/New_York")
    base_h4 = base_levels.loc[base_levels.index.normalize() == current_day, "H4"].iloc[0]

    prior_changed = base.copy()
    prior_changed.loc[pd.Timestamp("2025-01-02 15:55", tz="America/New_York"), "high"] = 140.0
    prior_h4 = compute_camarilla(prior_changed).loc[
        base_levels.index.normalize() == current_day,
        "H4",
    ].iloc[0]

    current_changed = base.copy()
    current_changed.loc[pd.Timestamp("2025-01-03 09:35", tz="America/New_York"), "high"] = 300.0
    current_h4 = compute_camarilla(current_changed).loc[
        base_levels.index.normalize() == current_day,
        "H4",
    ].iloc[0]

    assert prior_h4 != base_h4
    assert current_h4 == base_h4


def test_session_pivot_signal_rejection_only() -> None:
    frame = _signal_frame()
    frame.iloc[14, frame.columns.get_loc("low")] = 94.5
    frame.iloc[14, frame.columns.get_loc("close")] = 94.75

    signals = session_pivot_signal(frame)

    assert signals.iloc[14] == 0


def test_session_pivot_signal_daily_cap() -> None:
    frame = _signal_frame()
    for row in (14, 15, 16):
        frame.iloc[row, frame.columns.get_loc("low")] = 95.0
        frame.iloc[row, frame.columns.get_loc("close")] = 96.0

    signals = session_pivot_signal(frame, max_per_day=2)

    assert signals.iloc[14] == 1
    assert signals.iloc[15] == 1
    assert signals.iloc[16] == 0
    assert int((signals != 0).sum()) == 2


def test_session_pivot_signal_skips_invalid_atr_values() -> None:
    frame = _signal_frame()
    frame.iloc[:14, frame.columns.get_loc("low")] = 95.0
    frame.iloc[:14, frame.columns.get_loc("close")] = 96.0
    frame.iloc[:7, frame.columns.get_loc("atr_14")] = np.nan
    frame.iloc[7:14, frame.columns.get_loc("atr_14")] = 0.0
    frame.iloc[14, frame.columns.get_loc("low")] = 95.0
    frame.iloc[14, frame.columns.get_loc("close")] = 96.0

    signals = session_pivot_signal(frame)

    assert (signals.iloc[:14] == 0).all()
    assert signals.iloc[14] == 1


def test_ny_am_high_excludes_current_bar() -> None:
    index = pd.date_range("2025-01-02 09:30", periods=20, freq="5min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": np.full(20, 100.0),
            "high": [100.0, 110.0, 105.0, *([102.0] * 17)],
            "low": np.full(20, 99.0),
            "close": np.full(20, 100.0),
            "volume": np.full(20, 1000.0),
        },
        index=index,
    )
    atr = pd.Series(1.0, index=index)

    features = compute_pivot_features(frame, level_source_df=frame, atr_series=atr)
    derived_nyam_high = frame["close"] - features["ny_am_high_dist"] * atr

    assert derived_nyam_high.iloc[1] == 100.0
    assert derived_nyam_high.iloc[2] == 110.0


def test_completed_session_levels_use_full_source_before_rth_filter() -> None:
    full_index = pd.DatetimeIndex(
        [
            "2025-01-01 20:00",
            "2025-01-02 01:55",
            "2025-01-02 02:00",
            "2025-01-02 06:55",
            "2025-01-02 07:00",
            "2025-01-02 09:25",
            "2025-01-02 09:30",
            "2025-01-02 09:35",
        ],
        tz="America/New_York",
    )
    full = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 112.0, 120.0, 125.0, 130.0, 135.0, 999.0, 150.0],
            "low": [90.0, 88.0, 80.0, 85.0, 70.0, 75.0, 1.0, 95.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0] * 8,
        },
        index=full_index,
    )
    rth = full.between_time("09:30", "15:00")
    atr = pd.Series(1.0, index=rth.index)

    features = compute_pivot_features(rth, level_source_df=full, atr_series=atr)

    assert "asia_high_dist" in features.columns
    assert "london_high_dist" in features.columns
    assert "premarket_high_dist" in features.columns
    assert rth["close"].iloc[0] - features["asia_high_dist"].iloc[0] == 112.0
    assert rth["close"].iloc[0] - features["london_low_dist"].iloc[0] == 80.0
    assert rth["close"].iloc[0] - features["premarket_high_dist"].iloc[0] == 135.0
    assert rth["close"].iloc[0] - features["premarket_low_dist"].iloc[0] == 70.0

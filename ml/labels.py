"""Triple-barrier + meta-labels for signal-bar classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SESSION_END_HOUR = 15
SESSION_END_MINUTE = 0


def _session_close_position(index: pd.DatetimeIndex, entry_pos: int, max_bars: int) -> int:
    """Return the vertical-barrier position without crossing 15:00 ET or the session date."""
    entry_time = index[entry_pos]
    entry_day = entry_time.normalize()
    max_pos = min(entry_pos + int(max_bars), len(index) - 1)
    session_end = entry_day + pd.Timedelta(hours=SESSION_END_HOUR, minutes=SESSION_END_MINUTE)
    if entry_time.tzinfo is not None and session_end.tzinfo is None:
        session_end = session_end.tz_localize(entry_time.tzinfo)

    exit_pos = entry_pos
    for cursor in range(entry_pos + 1, max_pos + 1):
        timestamp = index[cursor]
        if timestamp.normalize() != entry_day or timestamp > session_end:
            break
        exit_pos = cursor
    return exit_pos


def _nan_row() -> dict[str, Any]:
    return {
        "label": np.nan,
        "exit_bar": np.nan,
        "exit_time": pd.NaT,
        "exit_price": np.nan,
        "r_multiple": np.nan,
        "barrier_hit": np.nan,
    }


def triple_barrier_label(
    df: pd.DataFrame,
    signal_series: pd.Series,
    atr_series: pd.Series,
    stop_atr_mult: float = 1.5,
    target_r_mult: float = 1.0,
    max_bars: int = 60,
    transaction_cost_pts: float = 0.07,
) -> pd.DataFrame:
    """Return binary win/loss labels for active signal bars.

    Output columns are ``label`` (0=loss, 1=win), ``exit_bar``, ``exit_time``,
    ``exit_price``, ``r_multiple``, and ``barrier_hit``. Rows with no signal or
    invalid inputs remain NaN-labeled.
    """
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"triple_barrier_label missing price columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("triple_barrier_label requires a DatetimeIndex")
    if max_bars < 1:
        raise ValueError("max_bars must be at least 1")
    if stop_atr_mult <= 0:
        raise ValueError("stop_atr_mult must be positive")
    if target_r_mult <= 0:
        raise ValueError("target_r_mult must be positive")

    frame = df.loc[:, ["high", "low", "close"]].copy().sort_index()
    signals = pd.Series(signal_series, index=signal_series.index).reindex(frame.index)
    atr = pd.Series(atr_series, index=atr_series.index).reindex(frame.index)

    rows = [_nan_row() for _ in range(len(frame))]
    index = frame.index
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    signal_values = pd.to_numeric(signals, errors="coerce").to_numpy(dtype=float)
    atr_values = pd.to_numeric(atr, errors="coerce").to_numpy(dtype=float)

    for entry_pos in range(len(frame)):
        raw_signal = signal_values[entry_pos]
        if not np.isfinite(raw_signal) or raw_signal == 0:
            continue

        direction = 1 if raw_signal > 0 else -1
        entry = closes[entry_pos]
        atr_value = atr_values[entry_pos]
        if not np.isfinite(entry) or not np.isfinite(atr_value) or atr_value <= 0:
            continue

        stop_pts = float(stop_atr_mult) * atr_value
        target_pts = float(target_r_mult) * stop_pts
        if stop_pts <= 0 or target_pts <= 0:
            continue

        if direction > 0:
            stop_price = entry - stop_pts
            target_price = entry + target_pts
        else:
            stop_price = entry + stop_pts
            target_price = entry - target_pts

        exit_pos = _session_close_position(index, entry_pos, max_bars=max_bars)
        label: int | None = None
        exit_price: float | None = None
        barrier_hit = "vertical"
        barrier_pos = exit_pos

        for cursor in range(entry_pos + 1, exit_pos + 1):
            high = highs[cursor]
            low = lows[cursor]
            if direction > 0:
                stop_hit = low <= stop_price
                target_hit = high >= target_price
            else:
                stop_hit = high >= stop_price
                target_hit = low <= target_price

            if stop_hit:
                label = 0
                exit_price = stop_price
                barrier_hit = "stop"
                barrier_pos = cursor
                break
            if target_hit:
                label = 1
                exit_price = target_price
                barrier_hit = "target"
                barrier_pos = cursor
                break

        if label is None:
            exit_price = closes[exit_pos]
            pnl_pts = (exit_price - entry) * direction - float(transaction_cost_pts)
            label = 1 if pnl_pts > 0 else 0
            barrier_pos = exit_pos

        assert exit_price is not None
        pnl_points = (exit_price - entry) * direction - float(transaction_cost_pts)
        rows[entry_pos] = {
            "label": float(label),
            "exit_bar": float(barrier_pos),
            "exit_time": index[barrier_pos],
            "exit_price": float(exit_price),
            "r_multiple": float(pnl_points / stop_pts) if stop_pts > 0 else np.nan,
            "barrier_hit": barrier_hit,
        }

    result = pd.DataFrame(rows, index=frame.index)
    result["label"] = result["label"].astype("float64")
    return result


__all__ = ["triple_barrier_label"]

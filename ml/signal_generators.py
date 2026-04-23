"""Signal generators for Agent 1A."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from Implementation.ifvg_generator import ifvg_combined, ifvg_open_signals, ifvg_signals
from Implementation.ttm_squeeze_generator import ttm_squeeze


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100).where(avg_gain.notna() | avg_loss.notna())


def _infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 0
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 0
    median_delta = deltas.median()
    return max(int(round(median_delta.total_seconds() / 60.0)), 1)


def _window_bar_count(day_df: pd.DataFrame, minutes: int) -> int:
    bar_minutes = _infer_bar_minutes(day_df.index)
    if bar_minutes <= 0 or minutes % bar_minutes != 0:
        raise ValueError("Signal generators require evenly spaced intraday bars.")
    return minutes // bar_minutes


def _parse_intraday_time(value: str | time | None) -> time | None:
    if value is None or isinstance(value, time):
        return value
    return pd.Timestamp(value).time()


def _session_mask(
    index: pd.Index,
    session_start: str | time | None = "09:30",
    session_end: str | time | None = "15:00",
    session_tz: str = "America/New_York",
) -> pd.Series:
    start = _parse_intraday_time(session_start)
    end = _parse_intraday_time(session_end)
    if start is None or end is None or not isinstance(index, pd.DatetimeIndex):
        return pd.Series(True, index=index, dtype=bool)

    local_index = index.tz_convert(session_tz) if index.tz is not None else index
    local_times = pd.Series(local_index.time, index=index)
    if start <= end:
        return (local_times >= start) & (local_times <= end)
    return (local_times >= start) | (local_times <= end)


def orb_volatility_filtered(
    df: pd.DataFrame,
    or_minutes: int = 10,
    atr_period: int = 14,
    atr_lookback: int = 100,
    min_atr_pct: int = 25,
    max_atr_pct: int = 85,
    max_signals_per_day: int = 1,
) -> pd.Series:
    """Return ORB signals gated by ATR percentile regime."""
    signals = pd.Series(0, index=df.index, dtype=int)
    atr = _atr(df, atr_period)
    atr_low = atr.rolling(atr_lookback, min_periods=atr_lookback).quantile(min_atr_pct / 100.0)
    atr_high = atr.rolling(atr_lookback, min_periods=atr_lookback).quantile(max_atr_pct / 100.0)

    for _, day_df in df.groupby(df.index.date):
        try:
            window_bars = _window_bar_count(day_df, or_minutes)
        except ValueError:
            continue

        if len(day_df) < window_bars + 1:
            continue

        opening_window = day_df.iloc[:window_bars]
        if len(opening_window) != window_bars or opening_window.index[0].strftime("%H:%M") != "09:30":
            continue

        or_high = float(opening_window["high"].max())
        or_low = float(opening_window["low"].min())
        candidate_df = day_df.iloc[window_bars:]
        candidate_df = candidate_df.between_time("09:40", "11:00", inclusive="both")
        if candidate_df.empty:
            continue

        daily_count = 0
        for ts, row in candidate_df.iterrows():
            if daily_count >= max_signals_per_day:
                break

            row_atr = atr.loc[ts]
            low_gate = atr_low.loc[ts]
            high_gate = atr_high.loc[ts]
            if pd.isna(row_atr) or pd.isna(low_gate) or pd.isna(high_gate):
                continue
            if not (low_gate <= row_atr <= high_gate):
                continue

            if row["close"] > or_high:
                signals.at[ts] = 1
                daily_count += 1
            elif row["close"] < or_low:
                signals.at[ts] = -1
                daily_count += 1

    return signals


def orb_wick_rejection(
    df: pd.DataFrame,
    or_minutes: int = 10,
    min_body_pct: float = 0.55,
    atr_period: int = 14,
    max_signals_per_day: int = 1,
) -> pd.Series:
    """Return ORB signals only when the breakout candle body dominates the range."""
    signals = pd.Series(0, index=df.index, dtype=int)
    atr = _atr(df, atr_period)

    for _, day_df in df.groupby(df.index.date):
        try:
            window_bars = _window_bar_count(day_df, or_minutes)
        except ValueError:
            continue

        if len(day_df) < window_bars + 1:
            continue

        opening_window = day_df.iloc[:window_bars]
        if len(opening_window) != window_bars or opening_window.index[0].strftime("%H:%M") != "09:30":
            continue

        or_high = float(opening_window["high"].max())
        or_low = float(opening_window["low"].min())
        candidate_df = day_df.iloc[window_bars:]
        candidate_df = candidate_df.between_time("09:40", "11:00", inclusive="both")
        if candidate_df.empty:
            continue

        daily_count = 0
        for ts, row in candidate_df.iterrows():
            if daily_count >= max_signals_per_day:
                break
            if pd.isna(atr.loc[ts]):
                continue

            bar_range = row["high"] - row["low"]
            if bar_range == 0:
                continue

            body_pct = abs(row["close"] - row["open"]) / bar_range
            if body_pct < min_body_pct:
                continue

            if row["close"] > or_high:
                signals.at[ts] = 1
                daily_count += 1
            elif row["close"] < or_low:
                signals.at[ts] = -1
                daily_count += 1

    return signals


def orb_initial_balance(
    df: pd.DataFrame,
    atr_period: int = 14,
    max_signals_per_day: int = 1,
) -> pd.Series:
    """Return initial-balance breakout signals."""
    signals = pd.Series(0, index=df.index, dtype=int)
    atr = _atr(df, atr_period)

    for _, day_df in df.groupby(df.index.date):
        try:
            window_bars = _window_bar_count(day_df, 60)
        except ValueError:
            continue

        if len(day_df) < window_bars + 1:
            continue

        ib_window = day_df.iloc[:window_bars]
        if len(ib_window) != window_bars or ib_window.index[0].strftime("%H:%M") != "09:30":
            continue

        ib_high = float(ib_window["high"].max())
        ib_low = float(ib_window["low"].min())
        ib_range = ib_high - ib_low

        candidate_df = day_df.iloc[window_bars:]
        candidate_df = candidate_df.between_time("10:30", "11:00", inclusive="both")
        if candidate_df.empty:
            continue

        first_candidate_ts = candidate_df.index[0]
        first_atr = atr.loc[first_candidate_ts]
        if pd.isna(first_atr) or ib_range < first_atr:
            continue

        daily_count = 0
        for ts, row in candidate_df.iterrows():
            if daily_count >= max_signals_per_day:
                break
            if pd.isna(atr.loc[ts]):
                continue

            if row["close"] > ib_high:
                signals.at[ts] = 1
                daily_count += 1
            elif row["close"] < ib_low:
                signals.at[ts] = -1
                daily_count += 1

    return signals


def orb_volume_adaptive(
    df: pd.DataFrame,
    or_minutes: int = 10,
    vol_multiplier: float = 1.5,
    atr_period: int = 14,
    max_signals_per_day: int = 1,
) -> pd.Series:
    """Return ORB signals only when breakout volume exceeds the OR average."""
    signals = pd.Series(0, index=df.index, dtype=int)
    atr = _atr(df, atr_period)

    for _, day_df in df.groupby(df.index.date):
        try:
            window_bars = _window_bar_count(day_df, or_minutes)
        except ValueError:
            continue

        if len(day_df) < window_bars + 1:
            continue

        opening_window = day_df.iloc[:window_bars]
        if len(opening_window) != window_bars or opening_window.index[0].strftime("%H:%M") != "09:30":
            continue

        or_high = float(opening_window["high"].max())
        or_low = float(opening_window["low"].min())
        or_avg_volume = float(opening_window["volume"].mean())
        if or_avg_volume == 0:
            continue

        candidate_df = day_df.iloc[window_bars:]
        candidate_df = candidate_df.between_time("09:40", "11:00", inclusive="both")
        if candidate_df.empty:
            continue

        daily_count = 0
        for ts, row in candidate_df.iterrows():
            if daily_count >= max_signals_per_day:
                break
            if pd.isna(atr.loc[ts]):
                continue
            if row["volume"] < or_avg_volume * vol_multiplier:
                continue

            if row["close"] > or_high:
                signals.at[ts] = 1
                daily_count += 1
            elif row["close"] < or_low:
                signals.at[ts] = -1
                daily_count += 1

    return signals


def connors_rsi2(
    df: pd.DataFrame,
    rsi_period: int = 2,
    rsi_entry: int = 10,
    rsi_exit: int = 90,
    exit_ma: int = 5,
    trend_ma: int = 200,
    stop_mult: float = 1.5,
    target_atr_mult: float = 1.0,
    atr_period: int = 14,
    session_start: str | time | None = "09:30",
    session_end: str | time | None = "15:00",
    session_tz: str = "America/New_York",
    legacy_output: bool = True,
) -> pd.Series | pd.DataFrame:
    """Return ConnorsRSI2 signals, optionally with stop/target metadata."""
    signals = pd.Series(0, index=df.index, dtype=int)
    stop_px = pd.Series(np.nan, index=df.index, dtype=float)
    target_px = pd.Series(np.nan, index=df.index, dtype=float)

    active_session = _session_mask(df.index, session_start, session_end, session_tz)
    session_df = df.loc[active_session]
    if session_df.empty:
        if legacy_output:
            return signals
        return pd.DataFrame(
            {"direction": signals, "stop_px": stop_px, "target_px": target_px},
            index=df.index,
        )

    close = session_df["close"]
    trend = close.rolling(trend_ma).mean()
    exit_avg = close.rolling(exit_ma).mean()
    atr = _atr(session_df, atr_period)
    rsi = _rsi(close, rsi_period)

    warmup = max(trend_ma, exit_ma, rsi_period, atr_period)
    position = 0

    for i in range(warmup, len(session_df)):
        ts = session_df.index[i]
        price = close.iloc[i]
        trend_value = trend.iloc[i]
        exit_value = exit_avg.iloc[i]
        rsi_value = rsi.iloc[i]
        atr_value = atr.iloc[i]

        if pd.isna(trend_value) or pd.isna(exit_value) or pd.isna(rsi_value) or pd.isna(atr_value):
            continue

        if position == 1:
            if rsi_value > rsi_exit or price > exit_value:
                position = 0
            continue

        if position == -1:
            if rsi_value < rsi_entry or price < exit_value:
                position = 0
            continue

        prev_rsi = rsi.iloc[i - 1]
        if pd.isna(prev_rsi):
            continue

        if price > trend_value and prev_rsi >= rsi_entry and rsi_value < rsi_entry:
            signals.at[ts] = 1
            stop_px.at[ts] = price - stop_mult * atr_value
            target_px.at[ts] = price + target_atr_mult * atr_value
            position = 1
        elif price < trend_value and prev_rsi <= rsi_exit and rsi_value > rsi_exit:
            signals.at[ts] = -1
            stop_px.at[ts] = price + stop_mult * atr_value
            target_px.at[ts] = price - target_atr_mult * atr_value
            position = -1

    if legacy_output:
        return signals
    return pd.DataFrame(
        {"direction": signals, "stop_px": stop_px, "target_px": target_px},
        index=df.index,
    )


def _level_series(df: pd.DataFrame, level_col: str) -> pd.Series:
    """Return a pivot level series, deriving it from ATR-normalized distance if needed."""
    if level_col in df.columns:
        return pd.to_numeric(df[level_col], errors="coerce")

    dist_candidates = [f"{level_col}_dist"]
    if level_col.startswith("camarilla_"):
        short_name = level_col.removeprefix("camarilla_")
        dist_candidates.append(f"{short_name}_dist")

    if "atr_14" not in df.columns or "close" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    for dist_col in dist_candidates:
        if dist_col not in df.columns:
            continue
        dist = pd.to_numeric(df[dist_col], errors="coerce")
        atr = pd.to_numeric(df["atr_14"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        return close.sub(dist.mul(atr))

    return pd.Series(np.nan, index=df.index, dtype=float)


def _prev_day_close_series(df: pd.DataFrame) -> pd.Series:
    if "prev_day_close" in df.columns:
        return pd.to_numeric(df["prev_day_close"], errors="coerce")

    h3 = _level_series(df, "camarilla_h3")
    s3 = _level_series(df, "camarilla_s3")
    prev_close = h3.add(s3).div(2.0)

    h4 = _level_series(df, "camarilla_h4")
    s4 = _level_series(df, "camarilla_s4")
    fallback = h4.add(s4).div(2.0)
    return prev_close.where(prev_close.notna(), fallback)


def _date_keys(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        return pd.Index(index.date)
    return index


def session_pivot_signal(
    df: pd.DataFrame,
    proximity_atr: float = 0.5,
    atr_period: int = 14,
    max_per_day: int = 2,
    legacy_output: bool = True,
) -> pd.Series | pd.DataFrame:
    """Generate session level pivot rejection signals.

    CRITICAL REQUIREMENTS:
    - df must have columns: open, high, low, close, volume
    - df must have columns: camarilla_h3, camarilla_h4, camarilla_s3, camarilla_s4
      (computed from PRIOR day OHLC — see camarilla_pivot_generator.py)
    - df must have columns: session_*_high, session_*_low (Asia, London, Pre-Market, NY AM)
    - df must have column: prev_day_high, prev_day_low, prev_day_close
    - df must have column: atr_14 (ATR computed from lookback-only bars, never forward)
    - df index must be tz-aware DatetimeIndex in America/New_York
    - df must be pre-filtered to 09:30–15:00 ET session bars only

    Logic:
    1. For each bar, check if bar.low penetrates any support level within proximity
    2. If yes, check rejection: bar.close must be ABOVE the touched level (wick-tag only)
    3. Check context: bar.close < prev_day_close (oversold context for longs)
    4. If all pass and daily count < max_per_day: signal = +1
    5. Symmetric for shorts: bar.high near resistance, close BELOW level, close > prev_day_close
    6. Max 2 signals per calendar day (resets at session open)

    Parameters
    ----------
    df : DataFrame with required price and level columns (see above)
    proximity_atr : ATR multiple for level touch range (default 0.5)
    atr_period : ATR period for volatility normalization (default 14)
    max_per_day : Maximum signals per calendar day (default 2)
    legacy_output : If True (default), return pd.Series of direction only (+1/-1/0) for backward compat.
                    If False, return DataFrame with columns: direction, stop_px, target_px, level_hit

    Returns
    -------
    If legacy_output=True: pd.Series of +1 (long), -1 (short), 0 (no signal), same index as df
    If legacy_output=False: DataFrame with columns [direction, stop_px, target_px, level_hit]
    """
    required_price_columns = {"high", "low", "close", "atr_14"}
    if not required_price_columns.issubset(df.columns):
        if "session_pivot_signal" in df.columns:
            result = pd.to_numeric(df["session_pivot_signal"], errors="coerce").fillna(0).astype(int)
            if legacy_output:
                return result
            else:
                return pd.DataFrame(index=df.index, columns=["direction", "stop_px", "target_px", "level_hit"])
        missing = sorted(required_price_columns.difference(df.columns))
        raise KeyError(f"session_pivot_signal requires columns: {missing}")

    candidate = pd.Series(0, index=df.index, dtype=int)
    atr_14 = pd.to_numeric(df["atr_14"], errors="coerce") if "atr_14" in df.columns else pd.Series(np.nan, index=df.index)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    prev_day_close = _prev_day_close_series(df)
    valid_base = atr_14.notna() & (atr_14 > 0)
    proximity = atr_14.mul(proximity_atr)

    # Metadata columns (only needed if not legacy_output)
    level_hit = pd.Series("", index=df.index, dtype=str)
    stop_px = pd.Series(np.nan, index=df.index, dtype=float)
    target_px = pd.Series(np.nan, index=df.index, dtype=float)

    # Map of level column name to display name for level_hit
    level_names = {
        "camarilla_s4": "s4",
        "camarilla_h4": "h4",
        "camarilla_s3": "s3",
        "camarilla_h3": "h3",
        "prev_day_low": "prev_day_low",
        "prev_day_high": "prev_day_high",
        "asia_low": "asia_low",
        "asia_high": "asia_high",
        "london_low": "london_low",
        "london_high": "london_high",
        "premarket_low": "premarket_low",
        "premarket_high": "premarket_high",
    }

    priority_checks = [
        (1, "camarilla_s4"),
        (-1, "camarilla_h4"),
        (1, "camarilla_s3"),
        (-1, "camarilla_h3"),
        (1, "prev_day_low"),
        (-1, "prev_day_high"),
        (1, "asia_low"),
        (-1, "asia_high"),
        (1, "london_low"),
        (-1, "london_high"),
        (1, "premarket_low"),
        (-1, "premarket_high"),
    ]

    for direction, level_col in priority_checks:
        level = _level_series(df, level_col)
        if level.isna().all():
            continue

        if direction == 1:
            # Long: low touches support, close rejects above
            condition = valid_base & low.le(level.add(proximity)) & close.gt(level) & close.lt(prev_day_close)
            if not legacy_output:
                # For longs: stop is the level touched (or slightly below for risk), target is 2R (example)
                touch_bars = df[condition & candidate.eq(0)]
                for idx in touch_bars.index:
                    stop_px.at[idx] = level.loc[idx] - atr_14.loc[idx] * 0.5  # Stop half-ATR below level
                    target_px.at[idx] = close.loc[idx] + atr_14.loc[idx] * 2.0  # 2R target
                    level_hit.at[idx] = level_names.get(level_col, level_col)
        else:
            # Short: high touches resistance, close rejects below
            condition = valid_base & high.ge(level.sub(proximity)) & close.lt(level) & close.gt(prev_day_close)
            if not legacy_output:
                touch_bars = df[condition & candidate.eq(0)]
                for idx in touch_bars.index:
                    stop_px.at[idx] = level.loc[idx] + atr_14.loc[idx] * 0.5  # Stop half-ATR above level
                    target_px.at[idx] = close.loc[idx] - atr_14.loc[idx] * 2.0  # 2R target
                    level_hit.at[idx] = level_names.get(level_col, level_col)

        candidate.loc[condition & candidate.eq(0)] = direction

    active = candidate.ne(0)
    daily_signal_number = active.astype(int).groupby(_date_keys(candidate.index)).cumsum()
    result_signal = candidate.where(active & daily_signal_number.le(max_per_day), 0).astype(int)

    if legacy_output:
        return result_signal
    else:
        # Return DataFrame with direction and metadata
        output = pd.DataFrame(index=df.index)
        output["direction"] = result_signal
        output["stop_px"] = stop_px
        output["target_px"] = target_px
        output["level_hit"] = level_hit
        return output


def session_pivot_break_signal(
    df: pd.DataFrame,
    atr_period: int = 14,
    max_per_day: int = 2,
) -> pd.Series:
    """Camarilla H4/S4 continuation break signals.

    STATUS: Companion signal documented in Strategies/Session Level Pivots Break.md.

    PURPOSE:
    Complements session_pivot_signal by capturing a different market structure. While
    session_pivot_signal fires on mean-reversion rejections AT support/resistance levels,
    this signal fires when price BREAKS THROUGH the outermost Camarilla levels (H4/S4),
    suggesting a directional continuation move away from pivot structure.

    LOGIC:
    Long (+1): bar closes ABOVE H4 and prior bar closed at or below H4.
              Signals institutional breakout buy (price exceeded upper limit).
    Short (-1): bar closes BELOW S4 and prior bar closed at or above S4.
               Signals institutional breakdown sell (price fell below lower limit).

    RELATIONSHIP TO session_pivot_signal:
    Mutually exclusive by construction:
      - session_pivot_signal: rejection (low touches level + close > level, or high touches level + close < level)
      - session_pivot_break_signal: break (close transitions from ≤ level to > level, or ≥ level to < level)

    Both share the daily cap of 2 signals per calendar day.

    PARAMETERS:
    atr_period : Used for ATR validation only; not used in the break calculation itself.
    max_per_day : Maximum signals per calendar day (shared with session_pivot_signal if both run).

    NOTE: This function reconstructs H4/S4 from Camarilla distance columns. If these are not present
          in the input df, the signal returns all zeros. For full feature parity with session_pivot_signal,
          ensure compute_pivot_features(..., expose_raw=True) is called upstream.
    """
    required_price_columns = {"close", "atr_14"}
    if not required_price_columns.issubset(df.columns):
        if "session_pivot_break_signal" in df.columns:
            return pd.to_numeric(df["session_pivot_break_signal"], errors="coerce").fillna(0).astype(int)
        missing = sorted(required_price_columns.difference(df.columns))
        raise KeyError(f"session_pivot_break_signal requires columns: {missing}")

    atr_14 = pd.to_numeric(df["atr_14"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    valid_base = atr_14.notna() & (atr_14 > 0)

    h4 = _level_series(df, "camarilla_h4")
    s4 = _level_series(df, "camarilla_s4")

    prev_close = close.shift(1)
    prev_h4 = h4.shift(1)
    prev_s4 = s4.shift(1)

    long_break = valid_base & h4.notna() & close.gt(h4) & prev_close.le(prev_h4.where(prev_h4.notna(), h4))
    short_break = valid_base & s4.notna() & close.lt(s4) & prev_close.ge(prev_s4.where(prev_s4.notna(), s4))

    candidate = pd.Series(0, index=df.index, dtype=int)
    candidate.loc[long_break] = 1
    candidate.loc[short_break & candidate.eq(0)] = -1

    active = candidate.ne(0)
    daily_signal_number = active.astype(int).groupby(_date_keys(candidate.index)).cumsum()
    return candidate.where(active & daily_signal_number.le(max_per_day), 0).astype(int)

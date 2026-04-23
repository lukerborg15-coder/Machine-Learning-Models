"""Dataset loading and feature assembly utilities for the ML pipeline."""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Implementation.camarilla_pivot_generator import (
    compute_camarilla,
    compute_prev_day_week,
    compute_session_levels,
)
from ml.labels import triple_barrier_label
from ml.signal_generators import (
    connors_rsi2,
    ifvg_combined,
    orb_initial_balance,
    orb_volatility_filtered,
    orb_wick_rejection,
    session_pivot_break_signal,
    session_pivot_signal,
    ttm_squeeze,
)


DATA_DIR = ROOT_DIR / "data"
ML_DATA_DIR = Path(__file__).resolve().parent / "data"
ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DATES_PATH = ML_DATA_DIR / "news_dates.csv"
EASTERN_TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "15:00"
SESSION_START_MINUTES = 9 * 60 + 30
SESSION_END_MINUTES = 15 * 60
PRICE_NORM_WINDOW = 20
ATR_PERIOD = 14
ATR_NORM_WINDOW = 20
WARMUP_BARS = 200
DEFAULT_FORWARD_HORIZON_BARS = 5
DEFAULT_META_LABEL_MAX_BARS = 60
EMBARGO_BARS_BY_TIMEFRAME: dict[str, int] = {
    # One full TopStep session: 09:30-15:00 ET = 330 minutes (TopStep flattens at 15:00, not 16:00)
    "1min": 330,
    "3min": 110,
    "5min": 66,
}

SIGNAL_COLUMNS = [
    "orb_vol_signal",
    "orb_wick_signal",
    "orb_ib_signal",
    "ifvg_signal",
    "ifvg_open_signal",
    "ttm_signal",
    "connors_signal",
    "session_pivot_signal",
    "session_pivot_break_signal",
]
SESSION_PIVOT_SIGNAL_COLUMN = "session_pivot_signal"
SESSION_PIVOT_BREAK_SIGNAL_COLUMN = "session_pivot_break_signal"
STRATEGY_SIGNAL_COLUMN_MAP: dict[str, str] = {
    "ifvg": "ifvg_signal",
    "ifvg_open": "ifvg_open_signal",
    "orb_ib": "orb_ib_signal",
    "orb_vol": "orb_vol_signal",
    "orb_wick": "orb_wick_signal",
    "ttm": "ttm_signal",
    "connors": "connors_signal",
    "session_pivot": SESSION_PIVOT_SIGNAL_COLUMN,
    "session_pivot_break": SESSION_PIVOT_BREAK_SIGNAL_COLUMN,
}
TRIPLE_BARRIER_OUTPUT_COLUMNS = ("label", "exit_bar", "exit_time", "exit_price", "r_multiple", "barrier_hit")
PIVOT_FEATURE_COLUMNS = [
    "h3_dist",
    "h4_dist",
    "s3_dist",
    "s4_dist",
    "h3_above",
    "h4_above",
    "s3_above",
    "s4_above",
    "ny_am_high_dist",
    "ny_am_low_dist",
    "prev_day_high_dist",
    "prev_day_low_dist",
    "prev_week_high_dist",
    "prev_week_low_dist",
]
EXTENDED_SESSION_PIVOT_FEATURE_COLUMNS = [
    "asia_high_dist",
    "asia_low_dist",
    "london_high_dist",
    "london_low_dist",
    "premarket_high_dist",
    "premarket_low_dist",
]
ADDITIVE_PIVOT_FEATURE_COLUMNS = [
    "camarilla_h3_dist",
    "camarilla_h4_dist",
    "camarilla_s3_dist",
    "camarilla_s4_dist",
]
RAW_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
BASE_FEATURE_COLUMNS = [
    "open_norm",
    "high_norm",
    "low_norm",
    "close_norm",
    "volume_log",
    "synthetic_delta",
    "return_1",
    "return_5",
    "atr_norm",
]
TIME_FEATURE_COLUMNS = ["time_of_day", "dow_sin", "dow_cos", "is_news_day"]


def _feature_matrix_output_path(instrument: str, timeframe: str) -> Path:
    instrument_key = instrument.strip().lower()
    timeframe_key = timeframe.strip().lower()
    return ML_DATA_DIR / f"features_{instrument_key}_{timeframe_key}.parquet"


def load_data(instrument: str, timeframe: str, session_only: bool = True) -> pd.DataFrame:
    """Load an instrument/timeframe CSV as a tz-aware DataFrame indexed in Eastern time."""
    instrument_key = instrument.strip().lower()
    timeframe_key = timeframe.strip().lower()
    csv_path = DATA_DIR / f"{instrument_key}_{timeframe_key}_databento.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError(f"Expected 'datetime' column in {csv_path}")

    datetime_index = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(EASTERN_TZ)

    loaded = df.drop(columns=["datetime"]).copy()
    loaded.index = pd.DatetimeIndex(datetime_index, name="datetime")
    loaded = loaded.sort_index()

    if session_only:
        loaded = loaded.between_time(SESSION_START, SESSION_END, inclusive="both")

    return loaded


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return _true_range(df).rolling(period).mean()


def _rolling_zscore(series: pd.Series, window: int = PRICE_NORM_WINDOW) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def _timeframe_to_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        return int(normalized[:-3])
    if normalized.endswith("h"):
        return int(normalized[:-1]) * 60
    match = re.fullmatch(r"(\d+)", normalized)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unsupported timeframe format: {timeframe}")


def _resolve_ifvg_htf(instrument: str, timeframe: str) -> pd.DataFrame | None:
    """Return the best available HTF DataFrame for IFVG confluence.

    Per strategy spec (IFVG.md, Condition 2), 1/2/3min entries require confluence
    with 5min/15min/1h/4h FVGs. 5min entries require 15min/1h/4h.

    We try timeframes in order of preference (higher = stronger, more reliable HTF signal).
    For 1min entries, try: [5min, 15min, 1h, 4h] â€” pick the first available.

    CRITICAL: Load the full day (session_only=False), not just 09:30-15:00.
    Overnight and pre-market HTF FVGs are legitimate confluence sources for early-session
    IFVG entries. The session filter (09:30-15:00) applies to the *entry* timeframe,
    NOT to the HTF confluence check.
    """
    timeframe_key = timeframe.strip().lower()
    # Preferred HTF order per entry timeframe (highest first; prefer stronger HTF)
    htf_candidates = {
        "1min": ["5min", "15min", "1h", "4h"],
        "2min": ["15min", "1h", "4h", "5min"],
        "3min": ["15min", "1h", "4h", "5min"],
        "5min": ["1h", "4h", "15min"],
    }
    candidates = htf_candidates.get(timeframe_key, [])
    for htf_timeframe in candidates:
        try:
            # Load full day (including overnight/premarket) â€” entry TF filter handles the gate
            return load_data(instrument, htf_timeframe, session_only=False)
        except FileNotFoundError:
            continue
    return None


def _load_news_dates() -> set[pd.Timestamp]:
    if not NEWS_DATES_PATH.exists():
        return set()

    news_df = pd.read_csv(NEWS_DATES_PATH)
    if news_df.empty or "date" not in news_df.columns:
        return set()

    dates = pd.to_datetime(news_df["date"], errors="coerce").dropna()
    normalized_dates: set[pd.Timestamp] = set()
    for ts in dates:
        if ts.tzinfo is None:
            normalized_dates.add(ts.tz_localize(EASTERN_TZ).normalize())
        else:
            normalized_dates.add(ts.tz_convert(EASTERN_TZ).normalize())
    return normalized_dates


def _supports_extended_sessions(df: pd.DataFrame) -> bool:
    minutes = df.index.hour * 60 + df.index.minute
    return bool((minutes < SESSION_START_MINUTES).any())


def _timestamp_for_date(day: pd.Timestamp, hour: int, minute: int = 0) -> pd.Timestamp:
    return day + pd.Timedelta(hours=hour, minutes=minute)


def _compute_completed_session_levels(target_index: pd.DatetimeIndex, source_df: pd.DataFrame) -> pd.DataFrame:
    """Compute completed overnight levels from the full source frame and align to target bars."""
    result = pd.DataFrame(index=target_index, dtype=float)
    for column in ("asia_high", "asia_low", "london_high", "london_low", "premarket_high", "premarket_low"):
        result[column] = np.nan

    if source_df.empty or target_index.empty:
        return result

    source_index = source_df.index
    source_minute = source_index.hour * 60 + source_index.minute
    has_true_extended_bars = (source_minute < SESSION_START_MINUTES).any() or (source_minute > 16 * 60).any()
    if not has_true_extended_bars:
        return result

    for day in sorted(target_index.normalize().unique()):
        day_mask = target_index.normalize() == day
        session_windows = {
            "asia": (_timestamp_for_date(day, 20, 0) - pd.Timedelta(days=1), _timestamp_for_date(day, 2, 0)),
            "london": (_timestamp_for_date(day, 2, 0), _timestamp_for_date(day, 7, 0)),
            "premarket": (_timestamp_for_date(day, 7, 0), _timestamp_for_date(day, 9, 30)),
        }
        for session_name, (start, end) in session_windows.items():
            session_bars = source_df.loc[(source_index >= start) & (source_index < end)]
            if session_bars.empty:
                continue
            result.loc[day_mask, f"{session_name}_high"] = session_bars["high"].max()
            result.loc[day_mask, f"{session_name}_low"] = session_bars["low"].min()

    return result


def _compute_ny_am_running_levels(target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NY AM running high/low using only prior bars in the current day."""
    result = pd.DataFrame(index=target_df.index, dtype=float)
    result["ny_am_high"] = np.nan
    result["ny_am_low"] = np.nan

    for _, day_df in target_df.groupby(target_df.index.normalize()):
        ny_am_bars = day_df.between_time("09:30", "12:00", inclusive="both")
        if ny_am_bars.empty:
            continue
        result.loc[ny_am_bars.index, "ny_am_high"] = ny_am_bars["high"].expanding().max().shift(1)
        result.loc[ny_am_bars.index, "ny_am_low"] = ny_am_bars["low"].expanding().min().shift(1)

    return result


def _in_training_session(timestamps: pd.Series) -> pd.Series:
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    return timestamps.notna() & minutes.between(SESSION_START_MINUTES, SESSION_END_MINUTES, inclusive="both")


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized OHLCV-derived features on a session-filtered working frame."""
    features = pd.DataFrame(index=df.index)

    features["open_norm"] = _rolling_zscore(df["open"])
    features["high_norm"] = _rolling_zscore(df["high"])
    features["low_norm"] = _rolling_zscore(df["low"])
    features["close_norm"] = _rolling_zscore(df["close"])
    features["volume_log"] = np.log1p(df["volume"])

    bar_range = df["high"] - df["low"]
    directional_move = df["close"] - df["open"]
    features["synthetic_delta"] = np.where(
        bar_range.eq(0.0),
        0.0,
        df["volume"] * directional_move / bar_range,
    )

    features["return_1"] = np.log(df["close"] / df["close"].shift(1))
    features["return_5"] = np.log(df["close"] / df["close"].shift(5))

    atr = _atr(df)
    atr_mean = atr.rolling(ATR_NORM_WINDOW).mean().replace(0.0, np.nan)
    features["atr"] = atr
    features["atr_norm"] = atr / atr_mean

    return features


def compute_pivot_levels(
    df: pd.DataFrame,
    level_source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute raw pivot/session levels from the full source frame and align to df."""
    source_df = level_source_df if level_source_df is not None else df

    camarilla = compute_camarilla(source_df).rename(
        columns={
            "H3": "camarilla_h3",
            "H4": "camarilla_h4",
            "S3": "camarilla_s3",
            "S4": "camarilla_s4",
        }
    )
    completed_sessions = _compute_completed_session_levels(df.index, source_df)
    ny_am_sessions = _compute_ny_am_running_levels(df)
    prev_day_week = compute_prev_day_week(source_df)

    levels = pd.concat(
        [
            camarilla.reindex(df.index),
            completed_sessions,
            ny_am_sessions,
            prev_day_week.reindex(df.index),
        ],
        axis=1,
    )
    return levels


def _safe_distance(close: pd.Series, level: pd.Series, atr: pd.Series) -> pd.Series:
    safe_atr = atr.replace(0.0, np.nan)
    distance = close.sub(level).div(safe_atr)
    return distance.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_pivot_features(
    df: pd.DataFrame,
    level_source_df: pd.DataFrame | None = None,
    atr_series: pd.Series | None = None,
    levels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute ATR-normalized pivot features from prior-day/full-session levels."""
    pivot_levels = levels if levels is not None else compute_pivot_levels(df, level_source_df=level_source_df)
    atr = atr_series.reindex(df.index) if atr_series is not None else _atr(df)
    close = df["close"]
    pivot_features = pd.DataFrame(index=df.index)

    level_map = {
        "h3": "camarilla_h3",
        "h4": "camarilla_h4",
        "s3": "camarilla_s3",
        "s4": "camarilla_s4",
    }
    for short_name, level_column in level_map.items():
        level = pivot_levels[level_column]
        pivot_features[f"{short_name}_dist"] = _safe_distance(close, level, atr)
        pivot_features[f"{short_name}_above"] = close.gt(level).astype(int)
        pivot_features[f"camarilla_{short_name}_dist"] = pivot_features[f"{short_name}_dist"]

    for column in EXTENDED_SESSION_PIVOT_FEATURE_COLUMNS:
        level_column = column.removesuffix("_dist")
        if level_column not in pivot_levels.columns or pivot_levels[level_column].isna().all():
            continue
        pivot_features[column] = _safe_distance(close, pivot_levels[level_column], atr)

    for column in ("ny_am_high_dist", "ny_am_low_dist"):
        level_column = column.removesuffix("_dist")
        if level_column in pivot_levels.columns:
            pivot_features[column] = _safe_distance(close, pivot_levels[level_column], atr)

    for column in ("prev_day_high_dist", "prev_day_low_dist", "prev_week_high_dist", "prev_week_low_dist"):
        level_column = column.removesuffix("_dist")
        if level_column in pivot_levels.columns:
            pivot_features[column] = _safe_distance(close, pivot_levels[level_column], atr)

    ordered_columns = [
        column
        for column in (
            PIVOT_FEATURE_COLUMNS
            + [column for column in EXTENDED_SESSION_PIVOT_FEATURE_COLUMNS if column in pivot_features.columns]
            + ADDITIVE_PIVOT_FEATURE_COLUMNS
        )
        if column in pivot_features.columns
    ]
    return pivot_features.loc[:, ordered_columns]


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic session-time features for the working frame."""
    features = pd.DataFrame(index=df.index)

    minutes_since_open = (df.index.hour * 60 + df.index.minute) - SESSION_START_MINUTES
    features["time_of_day"] = pd.Series(minutes_since_open, index=df.index, dtype=float) / (
        SESSION_END_MINUTES - SESSION_START_MINUTES
    )

    day_of_week = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    features["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    features["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)

    news_dates = _load_news_dates()
    normalized_dates = pd.Series(df.index.normalize(), index=df.index)
    features["is_news_day"] = normalized_dates.isin(news_dates).astype(int)

    return features


def compute_labels(df: pd.DataFrame, n_forward: int = DEFAULT_FORWARD_HORIZON_BARS) -> pd.DataFrame:
    """Compute forward-looking class labels on the raw, unfiltered frame."""
    labels = pd.DataFrame(index=df.index)

    future_return = df["close"].shift(-n_forward).div(df["close"]).sub(1.0)
    future_timestamp = pd.Series(df.index, index=df.index).shift(-n_forward)
    same_day = future_timestamp.dt.normalize().eq(pd.Series(df.index.normalize(), index=df.index))
    in_session = _in_training_session(future_timestamp)
    valid_target = same_day & in_session

    atr = _atr(df)
    threshold = 0.5 * atr.div(df["close"]).replace([np.inf, -np.inf], np.nan)

    label_values = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label_values.loc[valid_target] = 2
    label_values.loc[valid_target & (future_return > threshold)] = 0
    label_values.loc[valid_target & (future_return < -threshold)] = 1

    labels["future_return"] = future_return.where(valid_target)
    labels["label"] = label_values
    return labels


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    signal_features: pd.DataFrame,
    atr_series: pd.Series,
    stop_atr_mult: float = 1.5,
    target_r_mult: float = 1.0,
    max_bars: int = DEFAULT_META_LABEL_MAX_BARS,
    transaction_cost_pts: float = 0.07,
) -> pd.DataFrame:
    """Compute per-strategy binary meta-label columns on the session frame."""
    label_frames: list[pd.DataFrame] = []
    for strategy_name, signal_column in STRATEGY_SIGNAL_COLUMN_MAP.items():
        if signal_column not in signal_features.columns:
            continue
        labels = triple_barrier_label(
            df=df,
            signal_series=signal_features[signal_column],
            atr_series=atr_series,
            stop_atr_mult=stop_atr_mult,
            target_r_mult=target_r_mult,
            max_bars=max_bars,
            transaction_cost_pts=transaction_cost_pts,
        )
        renamed = labels.rename(
            columns={
                column: f"{column}_{strategy_name}"
                for column in TRIPLE_BARRIER_OUTPUT_COLUMNS
            }
        )
        label_frames.append(renamed)

    if not label_frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(label_frames, axis=1)


def embargo_bars_for_timeframe(timeframe: str) -> int:
    """Return a one-session embargo scaled to the bar timeframe."""
    timeframe_key = timeframe.strip().lower()
    if timeframe_key in EMBARGO_BARS_BY_TIMEFRAME:
        return EMBARGO_BARS_BY_TIMEFRAME[timeframe_key]

    timeframe_minutes = _timeframe_to_minutes(timeframe_key)
    if timeframe_minutes <= 0:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
    return int(math.ceil((SESSION_END_MINUTES - SESSION_START_MINUTES) / timeframe_minutes))


def _coerce_fold_timestamp(value: Any, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is None:
        if timestamp.tzinfo is not None:
            return timestamp.tz_convert(EASTERN_TZ).tz_localize(None)
        return timestamp

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(EASTERN_TZ).tz_convert(index.tz)
    return timestamp.tz_convert(index.tz)


def _slice_fold_range(df: pd.DataFrame, fold_spec: Any, start_attr: str, end_attr: str) -> pd.DataFrame:
    start_value = getattr(fold_spec, start_attr)
    end_value = getattr(fold_spec, end_attr)
    start = _coerce_fold_timestamp(start_value, df.index)
    end = _coerce_fold_timestamp(end_value, df.index)
    if isinstance(end_value, str) and ":" not in end_value:
        end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return df.loc[(start <= df.index) & (df.index <= end)].copy()


def _drop_purged_tail(frame: pd.DataFrame, drop_bars: int) -> pd.DataFrame:
    if drop_bars <= 0:
        return frame.copy()
    if len(frame) <= drop_bars:
        return frame.iloc[:0].copy()
    return frame.iloc[:-drop_bars].copy()


def _drop_embargo_head(frame: pd.DataFrame, embargo_bars: int) -> pd.DataFrame:
    if embargo_bars <= 0:
        return frame.copy()
    if len(frame) <= embargo_bars:
        return frame.iloc[:0].copy()
    return frame.iloc[embargo_bars:].copy()


def apply_purge_embargo(
    df: pd.DataFrame,
    fold_spec: "FoldSpec",
    forward_horizon_bars: int,
    embargo_bars: int = EMBARGO_BARS_BY_TIMEFRAME["5min"],
) -> dict[str, pd.DataFrame]:
    """Return purged train/val/test DataFrames for one fold.

    The forward label at row i depends on row i + N. Dropping the final
    N rows from train prevents train labels from referencing validation bars.
    Skipping the first embargo rows of validation leaves a buffer after train.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("apply_purge_embargo requires a DatetimeIndex")
    if forward_horizon_bars < 0:
        raise ValueError("forward_horizon_bars must be non-negative")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")

    train = _slice_fold_range(df, fold_spec, "train_start", "train_end")
    val = _slice_fold_range(df, fold_spec, "val_start", "val_end")
    test = _slice_fold_range(df, fold_spec, "test_start", "test_end")

    return {
        "train": _drop_purged_tail(train, int(forward_horizon_bars)),
        "val": _drop_embargo_head(val, int(embargo_bars)),
        "test": test,
    }


def _compute_signal_features(
    instrument: str,
    timeframe: str,
    df: pd.DataFrame,
    pivot_levels: pd.DataFrame | None = None,
    atr_series: pd.Series | None = None,
) -> pd.DataFrame:
    timeframe_minutes = _timeframe_to_minutes(timeframe)
    htf_df = _resolve_ifvg_htf(instrument, timeframe)

    # FIX #1: Build structural columns for IFVG sweep detection
    # Load raw data (including overnight/premarket) to compute structural levels
    raw_df = load_data(instrument, timeframe, session_only=False)

    # Build df_for_ifvg with all structural-level columns that _detect_sweep expects
    df_for_ifvg = df.copy()

    # Compute prev_day_week levels (high, low from previous calendar day/week)
    prev_dw = compute_prev_day_week(raw_df)
    df_for_ifvg = df_for_ifvg.join(
        prev_dw[['prev_day_high', 'prev_day_low', 'prev_week_high', 'prev_week_low']],
        how='left'
    )

    # Compute session levels (Asia, London, Pre-Market, NY AM)
    session_levels = compute_session_levels(raw_df)
    df_for_ifvg = df_for_ifvg.join(
        session_levels[['asia_high', 'asia_low', 'london_high', 'london_low',
                        'premarket_high', 'premarket_low', 'ny_am_high', 'ny_am_low']],
        how='left'
    )

    # Alias ny_am_high/low as session_high/low for backward compat with _detect_sweep
    df_for_ifvg['session_high'] = df_for_ifvg['ny_am_high']
    df_for_ifvg['session_low'] = df_for_ifvg['ny_am_low']

    # Alias premarket_high/low as overnight_high/low (premarket captures overnight + early session bars)
    df_for_ifvg['overnight_high'] = df_for_ifvg['premarket_high']
    df_for_ifvg['overnight_low'] = df_for_ifvg['premarket_low']

    # HTF structural levels (1h_high/low, 4h_high/low) â€” the high/low of the
    # most recently CLOSED HTF bar as of each entry-tf bar. Uses shift(1) +
    # ffill(reindex) to prevent lookahead: the 10:00 1h bar's high/low only
    # becomes visible at 10:00 and later, never earlier.
    #
    # Data source: prefer on-disk 1h/4h files if present; otherwise resample
    # raw_df (the 5min/1min source) to the HTF. Resampling uses
    # closed='left', label='left' so a bar labeled 09:00 covers [09:00, 10:00).
    def _htf_levels(source_df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
        resampled = source_df.resample(rule, closed='left', label='left').agg(
            {'high': 'max', 'low': 'min'}
        ).dropna()
        # Shift by 1 bar so the value at timestamp T reflects the PREVIOUS closed bar.
        # (Without shift, the bar at 10:00 would expose the 10:00-11:00 range as soon
        #  as 10:00 ticks over, which is not yet closed â†’ lookahead.)
        previous_closed = resampled.shift(1)
        previous_closed.columns = [f'{prefix}_high', f'{prefix}_low']
        return previous_closed.reindex(df_for_ifvg.index, method='ffill')

    # 1h: try on-disk file first (cleaner aggregation by the data builder), else resample.
    try:
        h1_df = load_data(instrument, '1h', session_only=False)
        h1_levels = h1_df[['high', 'low']].shift(1)
        h1_levels.columns = ['1h_high', '1h_low']
        h1_levels = h1_levels.reindex(df_for_ifvg.index, method='ffill')
    except FileNotFoundError:
        h1_levels = _htf_levels(raw_df, '1h', '1h')
    df_for_ifvg[['1h_high', '1h_low']] = h1_levels

    # 4h: no on-disk 4h file â€” resample from 1h data if available, else from raw_df.
    try:
        h4_source = load_data(instrument, '1h', session_only=False)
    except FileNotFoundError:
        h4_source = raw_df
    df_for_ifvg[['4h_high', '4h_low']] = _htf_levels(h4_source, '4h', '4h')

    # Build pre-open context frame for ifvg_open_signals (needs pre-market bars
    # to detect FVGs that form before 09:30 and get inverted at 09:30â€“09:40).
    # Pass the raw (session_only=False) frame that includes pre-market bars,
    # restricted to the time range of df for alignment.
    pre_open_df = raw_df.loc[raw_df.index >= df.index[0] - pd.Timedelta(days=1)]

    # Use legacy_output=True for backward compatibility: return Series of direction only
    ifvg_base, ifvg_open = ifvg_combined(
        df_for_ifvg,
        timeframe_minutes=timeframe_minutes,
        htf_df=htf_df,
        pre_open_df=pre_open_df,
        legacy_output=True,
    )

    signal_features = pd.DataFrame(index=df.index)
    signal_features["orb_vol_signal"] = orb_volatility_filtered(df)
    signal_features["orb_wick_signal"] = orb_wick_rejection(df)
    signal_features["orb_ib_signal"] = orb_initial_balance(df)
    # Reindex IFVG signals to match the original df index (in case df_for_ifvg has a different index after joins)
    signal_features["ifvg_signal"] = ifvg_base.reindex(df.index)
    signal_features["ifvg_open_signal"] = ifvg_open.reindex(df.index)
    signal_features["ttm_signal"] = ttm_squeeze(df)
    signal_features["connors_signal"] = connors_rsi2(df)
    session_pivot_input = df.copy()
    if pivot_levels is not None:
        session_pivot_input = session_pivot_input.join(pivot_levels, how="left")
    session_pivot_input["atr_14"] = atr_series.reindex(df.index) if atr_series is not None else _atr(df)
    pivot_rejection = session_pivot_signal(session_pivot_input)
    signal_features[SESSION_PIVOT_SIGNAL_COLUMN] = pivot_rejection
    session_pivot_input[SESSION_PIVOT_SIGNAL_COLUMN] = pivot_rejection
    signal_features[SESSION_PIVOT_BREAK_SIGNAL_COLUMN] = session_pivot_break_signal(session_pivot_input)
    return signal_features


def build_feature_matrix(instrument: str, timeframe: str) -> pd.DataFrame:
    """Assemble a leakage-safe feature matrix from raw and session-filtered data."""
    import time as _t
    import sys as _sys

    def _stage(name, start):
        elapsed = _t.perf_counter() - start
        print(f"  [{timeframe}] {name}: {elapsed:.1f}s", flush=True)
        _sys.stdout.flush()
        return _t.perf_counter()

    _t0 = _t.perf_counter()
    print(f"  [{timeframe}] === build_feature_matrix START ===", flush=True)

    raw_df = load_data(instrument, timeframe, session_only=False)
    working_df = load_data(instrument, timeframe, session_only=True)
    print(f"  [{timeframe}] loaded raw={len(raw_df):,} working={len(working_df):,}", flush=True)
    _t0 = _stage("load_data", _t0)

    raw_ohlcv = working_df.loc[:, RAW_OHLCV_COLUMNS].copy()
    ohlcv_features = compute_ohlcv_features(working_df)
    _t0 = _stage("compute_ohlcv_features", _t0)

    pivot_levels = compute_pivot_levels(working_df, level_source_df=raw_df)
    _t0 = _stage("compute_pivot_levels", _t0)

    signal_features = _compute_signal_features(
        instrument,
        timeframe,
        working_df,
        pivot_levels=pivot_levels,
        atr_series=ohlcv_features["atr"],
    )
    _t0 = _stage("_compute_signal_features (IFVG, ORB, TTM, Connors, pivots)", _t0)

    pivot_features = compute_pivot_features(
        working_df,
        level_source_df=raw_df,
        atr_series=ohlcv_features["atr"],
        levels=pivot_levels,
    )
    _t0 = _stage("compute_pivot_features", _t0)

    time_features = compute_time_features(working_df)
    _t0 = _stage("compute_time_features", _t0)

    label_frame = compute_labels(raw_df).reindex(working_df.index)
    _t0 = _stage("compute_labels", _t0)

    triple_barrier_frame = compute_triple_barrier_labels(
        df=working_df,
        signal_features=signal_features,
        atr_series=ohlcv_features["atr"],
    )
    _t0 = _stage("compute_triple_barrier_labels", _t0)

    feature_matrix = pd.concat(
        [
            raw_ohlcv,
            ohlcv_features,
            signal_features,
            pivot_features,
            time_features,
            label_frame,
            triple_barrier_frame,
        ],
        axis=1,
    )
    feature_matrix["atr_14"] = ohlcv_features["atr"]

    ordered_feature_columns = (
        BASE_FEATURE_COLUMNS
        + SIGNAL_COLUMNS
        + [column for column in PIVOT_FEATURE_COLUMNS if column in pivot_features.columns]
        + TIME_FEATURE_COLUMNS
        + [column for column in EXTENDED_SESSION_PIVOT_FEATURE_COLUMNS if column in pivot_features.columns]
        + [column for column in ADDITIVE_PIVOT_FEATURE_COLUMNS if column in pivot_features.columns]
    )
    output_columns = RAW_OHLCV_COLUMNS + ordered_feature_columns
    metadata_columns = [
        "future_return",
        "label",
        "atr_14",
        *list(triple_barrier_frame.columns),
    ]

    feature_matrix = feature_matrix.loc[:, output_columns + metadata_columns]
    feature_matrix = feature_matrix.iloc[WARMUP_BARS:].copy()
    feature_matrix = feature_matrix.dropna(subset=ordered_feature_columns)
    feature_matrix["label"] = feature_matrix["label"].fillna(2).astype(int)
    feature_matrix.attrs["feature_columns"] = ordered_feature_columns
    feature_matrix.attrs["target_column"] = "label"
    feature_matrix.attrs["objective"] = "meta_label"
    feature_matrix.attrs["meta_label_columns"] = {
        strategy_name: f"label_{strategy_name}"
        for strategy_name in STRATEGY_SIGNAL_COLUMN_MAP
        if f"label_{strategy_name}" in feature_matrix.columns
    }
    feature_matrix.attrs["instrument"] = instrument.strip().lower()
    feature_matrix.attrs["timeframe"] = timeframe.strip().lower()
    output_path = _feature_matrix_output_path(instrument, timeframe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(output_path)
    feature_matrix.attrs["parquet_path"] = str(output_path)
    return feature_matrix


__all__ = [
    "DEFAULT_FORWARD_HORIZON_BARS",
    "EMBARGO_BARS_BY_TIMEFRAME",
    "apply_purge_embargo",
    "embargo_bars_for_timeframe",
    "load_data",
    "compute_ohlcv_features",
    "compute_pivot_levels",
    "compute_pivot_features",
    "compute_time_features",
    "compute_labels",
    "compute_triple_barrier_labels",
    "build_feature_matrix",
    "rebuild_feature_matrices",
]


def rebuild_feature_matrices(
    instrument: str = "mnq",
    timeframes: Sequence[str] | None = None,
) -> dict[str, str]:
    """Rebuild feature parquets for the requested timeframes."""
    selected_timeframes = list(timeframes or ("1min", "2min", "3min", "5min"))
    rebuilt: dict[str, str] = {}
    for timeframe in selected_timeframes:
        matrix = build_feature_matrix(instrument, timeframe)
        rebuilt[timeframe] = str(matrix.attrs["parquet_path"])
    return rebuilt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build ML feature parquet files.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild feature parquets.")
    parser.add_argument("--instrument", default="mnq")
    parser.add_argument("--timeframe", action="append", dest="timeframes")
    parser.add_argument(
        "--data-dir", default=None, help="Override data directory path."
    )
    args = parser.parse_args(argv)

    if not args.rebuild:
        parser.print_help()
        return 0

    timeframes = args.timeframes or ["1min", "2min", "3min", "5min"]
    for tf in timeframes:
        print(f"Building {args.instrument} {tf} parquet...")
        build_feature_matrix(instrument=args.instrument, timeframe=tf)
        print(f"  Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

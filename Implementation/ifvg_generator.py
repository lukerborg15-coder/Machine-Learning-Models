"""
IFVG Signal Generator
=====================
Pre-built implementation for Agent 1 to copy into ml/signal_generators.py.
Do NOT rewrite this — copy as-is and integrate with the rest of the generators.

Covers:
  - detect_fvgs()       : finds all FVGs on a bar dataframe
  - ifvg_signals()      : base IFVG signal generator
  - ifvg_open_signals() : open variant (9:30 to configurable sweep_window_end, default 9:45)
  - ifvg_combined()     : runs both with shared 2/day limit

Strategy spec: strategyLabbrain/Strategies/IFVG.md
               strategyLabbrain/Strategies/IFVG - Open Variant.md
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# HTF min-gap scaling: multiply the entry-TF min_gap by this factor based on HTF timeframe.
# A 15-min FVG of 5 points is noise; it should be ~10-15 points. A 1H FVG should be ~20-30 points.
HTF_MIN_GAP_MULT = {
    '5min': 1.0,   # 5min is close to entry TF; no scaling
    '15min': 2.0,  # 15min FVG ~2x larger
    '1h': 4.0,     # 1h FVG ~4x larger
    '4h': 8.0,     # 4h FVG ~8x larger
}


# ─────────────────────────────────────────────
# FVG DETECTION
# ─────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame, min_gap_pts: float = 5.0, htf_timeframe: str = '') -> list:
    """
    Scan a bar DataFrame and return all FVGs meeting the minimum size.

    A Fair Value Gap is a 3-candle imbalance:
      Bullish FVG : bar[i-2].high < bar[i].low   (gap up between candle 1 top and candle 3 bottom)
      Bearish FVG : bar[i-2].low  > bar[i].high  (gap down between candle 1 bottom and candle 3 top)

    Returns list of dicts:
        fvg_type       : 'bullish' or 'bearish'
        zone_top       : upper boundary of the FVG zone
        zone_bottom    : lower boundary of the FVG zone
        gap_size       : zone_top - zone_bottom
        formed_at      : timestamp of bar[i] (third candle, when FVG is confirmed)
        formed_at_idx  : index position of bar[i] in the dataframe
        active         : True (all start active; set False after inversion)
        htf_timeframe  : timeframe string if this is an HTF FVG (for confluence tracking)
    """
    highs = df['high'].values
    lows  = df['low'].values
    index = df.index

    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG
        if highs[i - 2] < lows[i]:
            gap = lows[i] - highs[i - 2]
            if gap >= min_gap_pts:
                fvgs.append({
                    'fvg_type':       'bullish',
                    'zone_top':       lows[i],
                    'zone_bottom':    highs[i - 2],
                    'gap_size':       gap,
                    'formed_at':      index[i],
                    'formed_at_idx':  i,
                    'active':         True,
                    'htf_timeframe':  htf_timeframe,
                })

        # Bearish FVG
        if lows[i - 2] > highs[i]:
            gap = lows[i - 2] - highs[i]
            if gap >= min_gap_pts:
                fvgs.append({
                    'fvg_type':       'bearish',
                    'zone_top':       lows[i - 2],
                    'zone_bottom':    highs[i],
                    'gap_size':       gap,
                    'formed_at':      index[i],
                    'formed_at_idx':  i,
                    'active':         True,
                    'htf_timeframe':  htf_timeframe,
                })

    return fvgs


def _is_fvg_invalidated_before(
    fvg: dict,
    current_bar_i: int,
    closes: np.ndarray,
    fvg_type: str,
) -> bool:
    """
    Check if an FVG has been invalidated by a prior candle closing through it.

    Per IFVG spec: an FVG is invalid if ANY bar between formed_at and current_bar_i
    closed through the FVG zone in the opposite direction.

    Parameters
    ----------
    fvg : dict
        FVG dict with 'formed_at' (bar index), 'zone_top', 'zone_bottom', 'fvg_type'
    current_bar_i : int
        Current bar index being evaluated
    closes : np.ndarray
        Array of close prices from the day
    fvg_type : str
        'bullish' or 'bearish'

    Returns
    -------
    bool
        True if the FVG has been closed through in the opposite direction (invalidated)
    """
    # FVG is identified by fvg_type. Inversion direction is opposite.
    # Bullish FVG (zone_bottom to zone_top, support): invalidated by close BELOW zone_bottom
    # Bearish FVG (zone_top to zone_bottom, resistance): invalidated by close ABOVE zone_top

    # Find the bar index where the FVG formed
    # Note: fvg['formed_at'] is a pd.Timestamp; we need its index in the closes array
    # This is tricky because we're working with a daily slice.
    # The caller must track this; for now we assume the FVG dict has a 'formed_at_idx'
    # if not provided, we cannot check—return False to be safe.

    formed_at_idx = fvg.get('formed_at_idx', None)
    if formed_at_idx is None:
        return False  # Cannot invalidate without formation bar index

    # Check PRIOR bars in (formed_at_idx, current_bar_i) — EXCLUSIVE of current bar.
    # The current bar IS the inversion bar (its close is, by definition, on the
    # invalidating side of the zone — that's what triggers the inversion signal).
    # Including it would make every inversion invalidate itself. Invalidation here
    # means: a PRIOR bar already closed through the zone before this inversion.
    if fvg_type == 'bullish':
        # Bullish FVG: zone is [zone_bottom, zone_top]
        # Invalidated if a PRIOR close fell BELOW zone_bottom (gap already filled)
        zone_bottom = fvg['zone_bottom']
        for bar_idx in range(formed_at_idx + 1, current_bar_i):
            if bar_idx < len(closes) and closes[bar_idx] < zone_bottom:
                return True

    elif fvg_type == 'bearish':
        # Bearish FVG: zone is [zone_bottom, zone_top] (zone_bottom < zone_top)
        # Invalidated if a PRIOR close rose ABOVE zone_top (gap already filled)
        zone_top = fvg['zone_top']
        for bar_idx in range(formed_at_idx + 1, current_bar_i):
            if bar_idx < len(closes) and closes[bar_idx] > zone_top:
                return True

    return False


def _check_htf_confluence(
    price: float,
    bar_time: pd.Timestamp,
    htf_fvgs: list,
    proximity_pts: float,
    htf_timeframe_str: str = '',
) -> tuple:
    """
    Check whether price is near an active HTF FVG that formed before the current bar.

    Returns tuple (has_confluence: bool, htf_tf_used: str)
    - has_confluence: True if price is within proximity_pts of an active HTF FVG
    - htf_tf_used: the timeframe string of the matched HTF FVG (e.g., '15min', '1h'),
                   or empty string if no match

    The proximity_pts parameter is scaled by HTF_MIN_GAP_MULT[timeframe] to account
    for HTF bars being much larger than entry-TF bars.
    """
    # Scale proximity by HTF timeframe if provided
    scaled_proximity = proximity_pts
    if htf_timeframe_str and htf_timeframe_str in HTF_MIN_GAP_MULT:
        scaled_proximity = proximity_pts * HTF_MIN_GAP_MULT[htf_timeframe_str]

    for fvg in htf_fvgs:
        if not fvg['active']:
            continue
        if fvg['formed_at'] >= bar_time:
            continue
        near = (
            price >= fvg['zone_bottom'] - scaled_proximity
            and price <= fvg['zone_top'] + scaled_proximity
        )
        if near:
            htf_tf = fvg.get('htf_timeframe', '')
            return True, htf_tf
    return False, ''


def _detect_sweep(
    signal_direction: int,
    bar_i: int,
    df: pd.DataFrame,
) -> tuple:
    """
    Detect whether the given bar swept a structural liquidity level.

    A sweep occurs when the wick extends beyond a structural level (prev-day H/L,
    prev-week H/L, 1H/4H/session H/L) and the close is back on the original side.
    This is a true structural sweep, not a rolling-window proxy.

    Parameters
    ----------
    signal_direction : int
        +1 for long (swept lows), -1 for short (swept highs)
    bar_i : int
        Index of the bar in df to check for sweep
    df : pd.DataFrame
        Full day DataFrame with columns: high, low, close, plus structural level columns
        (prev_day_high, prev_day_low, prev_week_high, prev_week_low, and optionally
        1h_high, 1h_low, 4h_high, 4h_low, session_high, session_low)

    Returns
    -------
    tuple
        (sweep_detected: bool, sweep_bar_ts: pd.Timestamp or NaT)
        If sweep detected, returns the timestamp of the bar that performed the sweep.
        If not detected, returns NaT.

    Notes
    -----
    Per IFVG spec, a valid sweep touches or wicks through one of:
    - prev_day_high / prev_day_low
    - prev_week_high / prev_week_low
    - 1H high/low (if available)
    - 4H high/low (if available)
    - Session high/low (if available)

    The sweep must be on the opposite side of the signal direction (e.g., lows for long).
    Missing columns generate a warning (once per session) but do not fail the check.
    """
    if bar_i >= len(df):
        return False, pd.NaT

    bar = df.iloc[bar_i]
    bar_high = bar['high']
    bar_low = bar['low']
    bar_close = bar['close']
    bar_ts = df.index[bar_i]

    # List of structural level columns to check, based on signal direction
    if signal_direction == 1:  # Long: check for low sweep
        level_columns = [
            'prev_day_low', 'prev_week_low',
            '1h_low', '4h_low', 'session_low',
            'overnight_low', 'premarket_low', 'asia_low',
        ]
        # Sweep = wick below level, close back above level
        sweep_found = False
        for col in level_columns:
            if col not in df.columns:
                # Log warning once per session
                if not hasattr(_detect_sweep, '_warned_cols'):
                    _detect_sweep._warned_cols = set()
                if col not in _detect_sweep._warned_cols:
                    warnings.warn(
                        f"Structural level column '{col}' not found in DataFrame. "
                        f"Continuing without this level for sweep detection.",
                        category=UserWarning,
                        stacklevel=2
                    )
                    _detect_sweep._warned_cols.add(col)
                continue

            level = bar[col]
            if pd.isna(level):
                continue

            # Long sweep: low wicks below level, close above level
            if bar_low <= level and bar_close > level:
                sweep_found = True
                break

        return sweep_found, bar_ts if sweep_found else pd.NaT

    elif signal_direction == -1:  # Short: check for high sweep
        level_columns = [
            'prev_day_high', 'prev_week_high',
            '1h_high', '4h_high', 'session_high',
            'overnight_high', 'premarket_high', 'asia_high',
        ]
        # Sweep = wick above level, close back below level
        sweep_found = False
        for col in level_columns:
            if col not in df.columns:
                # Log warning once per session
                if not hasattr(_detect_sweep, '_warned_cols'):
                    _detect_sweep._warned_cols = set()
                if col not in _detect_sweep._warned_cols:
                    warnings.warn(
                        f"Structural level column '{col}' not found in DataFrame. "
                        f"Continuing without this level for sweep detection.",
                        category=UserWarning,
                        stacklevel=2
                    )
                    _detect_sweep._warned_cols.add(col)
                continue

            level = bar[col]
            if pd.isna(level):
                continue

            # Short sweep: high wicks above level, close below level
            if bar_high >= level and bar_close < level:
                sweep_found = True
                break

        return sweep_found, bar_ts if sweep_found else pd.NaT

    return False, pd.NaT


# ─────────────────────────────────────────────
# BASE IFVG SIGNALS
# ─────────────────────────────────────────────

def ifvg_signals(
    df: pd.DataFrame,
    timeframe_minutes: int = 5,
    max_signals_per_day: int = 2,
    swing_lookback: int = 10,
    htf_df: Optional[pd.DataFrame] = None,
    htf_proximity_pts: float = 10.0,
    _external_daily_counts: Optional[dict] = None,
    legacy_output: bool = False,
) -> pd.Series:
    """
    Generate IFVG signals bar-by-bar.

    Returns pd.Series with columns:
    - If legacy_output=True: direction only (int: -1/0/+1) for backward compatibility
    - If legacy_output=False: DataFrame with columns:
        - direction: -1/0/+1 (signal direction)
        - stop_px: float (stop loss price based on sweep swing)
        - target_px: float (target price based on 1R-1.5R)
        - sweep_bar_ts: pd.Timestamp (timestamp of the liquidity sweep bar)
        - htf_confluence_tf: str (HTF timeframe used for confluence, if any)

    Parameters
    ----------
    df                   : OHLCV DataFrame, DatetimeIndex in America/New_York, 09:30-15:00 only
    timeframe_minutes    : bar size (determines min gap: ≤2min→5pts, ≥3min→7pts)
    max_signals_per_day  : hard cap per day (shared with open variant via _external_daily_counts)
    swing_lookback       : bars back for liquidity sweep detection (NOT USED; preserved for compat)
    htf_df               : optional HTF dataframe for FVG confluence check
    htf_proximity_pts    : price proximity to HTF FVG zone in points
    _external_daily_counts: {date: count} injected by ifvg_combined() for shared daily cap
    legacy_output        : if True, return Series of direction only (backward compat)

    Notes
    -----
    - Entry signal fires on the CLOSE of the inversion candle.
    - Each FVG can only produce one signal (deactivated after inversion).
    - An FVG is invalid if ANY bar between formed_at and current closed through it
      in the opposite direction (FVG invalidation check per spec).
    - Liquidity sweep must detect actual structural level sweep (prev-day H/L, prev-week H/L, etc.)
      not just a rolling-window extreme.
    """
    min_gap = 5.0 if timeframe_minutes <= 2 else 7.0

    if legacy_output:
        signals = pd.Series(0, index=df.index, dtype=int)
    else:
        signals = pd.DataFrame(
            {
                'direction': 0,
                'stop_px': np.nan,
                'target_px': np.nan,
                'sweep_bar_ts': pd.NaT,
                'htf_confluence_tf': '',
            },
            index=df.index,
        )

    # Pre-compute HTF FVGs once (expensive if done per bar)
    # Get HTF timeframe from the HTF df if available
    # FIX #2: Determine htf_timeframe_str BEFORE calling detect_fvgs
    htf_timeframe_str = ''
    if htf_df is not None and len(htf_df) > 1:
        delta_minutes = (htf_df.index[1] - htf_df.index[0]).total_seconds() / 60
        if delta_minutes == 5:
            htf_timeframe_str = '5min'
        elif delta_minutes == 15:
            htf_timeframe_str = '15min'
        elif delta_minutes == 60:
            htf_timeframe_str = '1h'
        elif delta_minutes == 240:
            htf_timeframe_str = '4h'

    if htf_df is not None:
        # FIX: Scale min_gap for HTF detection. A 5-point FVG on 1h HTF is noise;
        # per HTF_MIN_GAP_MULT, it should be 5pts × 4 = 20pts. Without scaling,
        # HTF accepts 1.5-2.3× more FVGs than spec intent, inflating confluence.
        htf_min_gap = min_gap * HTF_MIN_GAP_MULT.get(htf_timeframe_str, 1.0)
        htf_fvgs = detect_fvgs(htf_df, min_gap_pts=htf_min_gap, htf_timeframe=htf_timeframe_str)
    else:
        htf_fvgs = []

    for day, day_df in df.groupby(df.index.date):
        # Apply session filter inside each day
        day_df = day_df.between_time('09:30', '15:00')
        if len(day_df) < 3:
            continue

        closes = day_df['close'].values
        highs  = day_df['high'].values
        lows   = day_df['low'].values
        index  = day_df.index

        # Fresh FVG list per day (only use that day's FVGs)
        day_fvgs = detect_fvgs(day_df, min_gap_pts=min_gap)

        # Daily signal counter — include external count from open variant if provided
        day_count = (_external_daily_counts or {}).get(day, 0)

        for bar_i in range(2, len(day_df)):
            if day_count >= max_signals_per_day:
                break

            close    = closes[bar_i]
            bar_time = index[bar_i]

            for fvg in day_fvgs:
                if not fvg['active']:
                    continue
                # FVG must have already formed (formed_at < current bar_time)
                if fvg['formed_at'] >= bar_time:
                    continue

                signal = 0

                # Bearish FVG violated upward → Bullish IFVG → Long
                if fvg['fvg_type'] == 'bearish' and close > fvg['zone_top']:
                    signal = 1

                # Bullish FVG violated downward → Bearish IFVG → Short
                elif fvg['fvg_type'] == 'bullish' and close < fvg['zone_bottom']:
                    signal = -1

                if signal == 0:
                    continue

                # ── FVG Invalidation Check ────────────────────────────────────
                # Per IFVG spec: if any bar between formed_at and current bar
                # closed through the FVG in the opposite direction, skip this FVG.
                if _is_fvg_invalidated_before(fvg, bar_i, closes, fvg['fvg_type']):
                    # Mark as inactive and skip (don't count against daily limit)
                    fvg['active'] = False
                    continue

                # Deactivate — each FVG fires at most once
                fvg['active'] = False

                # ── Gate 1: Liquidity sweep ──────────────────────────────────
                # Rewrite: detect actual structural-level sweeps from the df
                swept, sweep_bar_ts = _detect_sweep(signal, bar_i, day_df)
                if not swept:
                    continue

                # ── Gate 2: HTF FVG confluence (if HTF provided) ──────────────
                htf_confluence_tf = ''
                if htf_df is not None:
                    has_confluence, htf_confluence_tf = _check_htf_confluence(
                        close, bar_time, htf_fvgs, htf_proximity_pts, htf_timeframe_str
                    )
                    if not has_confluence:
                        continue

                # ── All gates passed ─────────────────────────────────────────
                if legacy_output:
                    signals.at[bar_time] = signal
                else:
                    # Compute stop and target (stub for now — will be refined)
                    # Stop = swing low/high of the sweep leg (before the IFVG formation)
                    # For now, use sweep bar's wick as reference
                    entry_px = close

                    if signal == 1:  # Long
                        # Stop at sweep bar's low (or lower nearby bars)
                        # FIX #3: Use pd.notna() instead of != pd.NaT
                        sweep_bar_idx = day_df.index.get_loc(sweep_bar_ts) if pd.notna(sweep_bar_ts) else bar_i
                        stop_px = lows[max(0, sweep_bar_idx - 2):sweep_bar_idx + 1].min()
                        r_distance = entry_px - stop_px
                        target_px = entry_px + r_distance  # 1R
                    else:  # Short (signal == -1)
                        # Stop at sweep bar's high (or higher nearby bars)
                        # FIX #3: Use pd.notna() instead of != pd.NaT
                        sweep_bar_idx = day_df.index.get_loc(sweep_bar_ts) if pd.notna(sweep_bar_ts) else bar_i
                        stop_px = highs[max(0, sweep_bar_idx - 2):sweep_bar_idx + 1].max()
                        r_distance = stop_px - entry_px
                        target_px = entry_px - r_distance  # 1R

                    signals.at[bar_time, 'direction'] = signal
                    signals.at[bar_time, 'stop_px'] = stop_px
                    signals.at[bar_time, 'target_px'] = target_px
                    signals.at[bar_time, 'sweep_bar_ts'] = sweep_bar_ts
                    signals.at[bar_time, 'htf_confluence_tf'] = htf_confluence_tf

                day_count += 1
                break  # one signal per bar max

    return signals


# ─────────────────────────────────────────────
# OPEN VARIANT SIGNALS (9:30-9:35 sweep)
# ─────────────────────────────────────────────

def ifvg_open_signals(
    df: pd.DataFrame,
    timeframe_minutes: int = 1,
    max_signals_per_day: int = 2,
    swing_lookback: int = 10,
    htf_df: Optional[pd.DataFrame] = None,
    htf_proximity_pts: float = 10.0,
    pre_open_df: Optional[pd.DataFrame] = None,
    sweep_window_end: str = '09:45',
    entry_window_end: str = '10:00',
    _external_daily_counts: Optional[dict] = None,
    legacy_output: bool = False,
) -> pd.Series:
    """
    IFVG Open Variant — identical to ifvg_signals() except with a timing gate around the open.

    Timing gate (tunable):
    - The liquidity sweep MUST occur between 09:30 ET and ``sweep_window_end`` (default 09:45)
    - The inversion/entry candle must close by ``entry_window_end`` (default 10:00)
    - Any 1/2/3/5-minute inversion candle that closes opposite the sweep direction
      within the entry window qualifies, provided FVG + HTF confluence gates pass.

    Default window (09:45) is wider than the original spec's 9:35 cutoff to capture the
    full open-drive manipulation phase. Tighten by passing sweep_window_end='09:35' to
    reproduce the strict spec behavior; widen to '10:00' to test diminishing returns.

    The sweep must be a real structural-level sweep (prev-day H/L, overnight H/L, pre-market H/L)
    not just "price moved" in the open window.

    Returns pd.Series (legacy_output=True) or DataFrame with full metadata (legacy_output=False).
    """
    min_gap = 5.0 if timeframe_minutes <= 2 else 7.0

    if legacy_output:
        signals = pd.Series(0, index=df.index, dtype=int)
    else:
        signals = pd.DataFrame(
            {
                'direction': 0,
                'stop_px': np.nan,
                'target_px': np.nan,
                'sweep_bar_ts': pd.NaT,
                'htf_confluence_tf': '',
            },
            index=df.index,
        )

    # Pre-compute HTF FVGs
    # FIX #2: Determine htf_timeframe_str BEFORE calling detect_fvgs
    htf_timeframe_str = ''
    if htf_df is not None and len(htf_df) > 1:
        delta_minutes = (htf_df.index[1] - htf_df.index[0]).total_seconds() / 60
        if delta_minutes == 5:
            htf_timeframe_str = '5min'
        elif delta_minutes == 15:
            htf_timeframe_str = '15min'
        elif delta_minutes == 60:
            htf_timeframe_str = '1h'
        elif delta_minutes == 240:
            htf_timeframe_str = '4h'

    if htf_df is not None:
        # FIX: Scale min_gap for HTF detection (see base variant for rationale).
        htf_min_gap = min_gap * HTF_MIN_GAP_MULT.get(htf_timeframe_str, 1.0)
        htf_fvgs = detect_fvgs(htf_df, min_gap_pts=htf_min_gap, htf_timeframe=htf_timeframe_str)
    else:
        htf_fvgs = []

    # Build day-keyed map of pre-open bars (if pre_open_df provided).
    # Per IFVG spec (Strategies/IFVG - Open Variant.md line 48): the inversion
    # candle can close "at or slightly after 9:35", and on 5-min the first possible
    # FVG within RTH can't form until 09:40 (3 bars 9:30/9:35/9:40). To capture
    # pre-market FVGs that get inverted at the open, we detect FVGs on a wider
    # pre-open + RTH window, but the inversion bar must still be in 09:30–09:40.
    pre_open_by_date: dict = {}
    if pre_open_df is not None and len(pre_open_df) > 0:
        # Use the window 06:00 ET (covers pre-market) through 15:00 ET for detection context.
        extended = pre_open_df.between_time('06:00', '15:00')
        for day_key, group in extended.groupby(extended.index.date):
            pre_open_by_date[day_key] = group

    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.between_time('09:30', '15:00')
        if len(day_df) < 3:
            continue

        # ── Time gate: Entry/inversion candle must close by entry_window_end ─
        entry_window = day_df.between_time('09:30', entry_window_end)
        if len(entry_window) < 1:
            continue

        # ── Sweep must occur between 09:30 and sweep_window_end ──────────────
        open_window = day_df.between_time('09:30', sweep_window_end)
        if len(open_window) < 1:
            continue

        closes = day_df['close'].values
        highs  = day_df['high'].values
        lows   = day_df['low'].values
        index  = day_df.index

        # FVG detection uses pre-open context when available, otherwise falls back
        # to RTH-only (which on 5-min can't produce an FVG in time for the 09:40 cap).
        detection_frame = pre_open_by_date.get(day, day_df)
        day_fvgs = detect_fvgs(detection_frame, min_gap_pts=min_gap)
        day_count = (_external_daily_counts or {}).get(day, 0)

        for bar_i in range(2, len(entry_window)):
            if day_count >= max_signals_per_day:
                break

            close    = entry_window['close'].iloc[bar_i]
            bar_time = entry_window.index[bar_i]

            for fvg in day_fvgs:
                if not fvg['active']:
                    continue
                if fvg['formed_at'] >= bar_time:
                    continue

                signal = 0
                if fvg['fvg_type'] == 'bearish' and close > fvg['zone_top']:
                    signal = 1
                elif fvg['fvg_type'] == 'bullish' and close < fvg['zone_bottom']:
                    signal = -1

                if signal == 0:
                    continue

                # ── FVG Invalidation Check ────────────────────────────────────
                # Check invalidation in the same frame the FVG was detected in
                # (detection_frame includes pre-open bars if provided, so invalidation
                # must look at the same extended window; otherwise formed_at_idx and
                # current_bar_i are in different reference frames and the check breaks).
                if bar_time in detection_frame.index:
                    bar_i_in_detection = detection_frame.index.get_loc(bar_time)
                    detection_closes = detection_frame['close'].values
                    if _is_fvg_invalidated_before(fvg, bar_i_in_detection, detection_closes, fvg['fvg_type']):
                        fvg['active'] = False
                        continue
                bar_i_in_day = day_df.index.get_loc(bar_time)

                fvg['active'] = False

                # ── Open variant: Sweep must occur in the 09:30–sweep_window_end window ──
                # AND the sweep must be a real structural-level sweep
                open_window_df = day_df.between_time('09:30', sweep_window_end)
                sweep_found = False
                sweep_bar_ts = pd.NaT

                for sweep_bar_i_offset in range(len(open_window_df)):
                    sweep_bar_ts_candidate = open_window_df.index[sweep_bar_i_offset]
                    sweep_bar_i_in_day = day_df.index.get_loc(sweep_bar_ts_candidate)
                    swept, sweep_ts = _detect_sweep(signal, sweep_bar_i_in_day, day_df)
                    if swept:
                        sweep_found = True
                        sweep_bar_ts = sweep_ts
                        break

                if not sweep_found:
                    continue

                # ── HTF confluence ────────────────────────────────────────────
                htf_confluence_tf = ''
                if htf_df is not None:
                    has_confluence, htf_confluence_tf = _check_htf_confluence(
                        close, bar_time, htf_fvgs, htf_proximity_pts, htf_timeframe_str
                    )
                    if not has_confluence:
                        continue

                # ── All gates passed ─────────────────────────────────────────
                if legacy_output:
                    signals.at[bar_time] = signal
                else:
                    entry_px = close
                    if signal == 1:  # Long
                        # FIX #3: Use pd.notna() instead of != pd.NaT
                        sweep_bar_i_in_day = day_df.index.get_loc(sweep_bar_ts) if pd.notna(sweep_bar_ts) else bar_i_in_day
                        stop_px = day_df['low'].iloc[max(0, sweep_bar_i_in_day - 2):sweep_bar_i_in_day + 1].min()
                        r_distance = entry_px - stop_px
                        target_px = entry_px + r_distance
                    else:  # Short
                        # FIX #3: Use pd.notna() instead of != pd.NaT
                        sweep_bar_i_in_day = day_df.index.get_loc(sweep_bar_ts) if pd.notna(sweep_bar_ts) else bar_i_in_day
                        stop_px = day_df['high'].iloc[max(0, sweep_bar_i_in_day - 2):sweep_bar_i_in_day + 1].max()
                        r_distance = stop_px - entry_px
                        target_px = entry_px - r_distance

                    signals.at[bar_time, 'direction'] = signal
                    signals.at[bar_time, 'stop_px'] = stop_px
                    signals.at[bar_time, 'target_px'] = target_px
                    signals.at[bar_time, 'sweep_bar_ts'] = sweep_bar_ts
                    signals.at[bar_time, 'htf_confluence_tf'] = htf_confluence_tf

                day_count += 1
                break

    return signals


# ─────────────────────────────────────────────
# COMBINED: SHARED 2/DAY LIMIT
# ─────────────────────────────────────────────

def ifvg_combined(
    df: pd.DataFrame,
    timeframe_minutes: int = 5,
    htf_df: Optional[pd.DataFrame] = None,
    htf_proximity_pts: float = 10.0,
    swing_lookback: int = 10,
    pre_open_df: Optional[pd.DataFrame] = None,
    sweep_window_end: str = '09:45',
    entry_window_end: str = '10:00',
    legacy_output: bool = True,
) -> tuple:
    """
    Runs both IFVG and IFVG Open Variant with a shared 2-signal/day cap.

    Order: base IFVG fires first and consumes the budget.
           Open Variant gets whatever remains (0, 1, or 2).

    Parameters
    ----------
    df, timeframe_minutes, htf_df, htf_proximity_pts, swing_lookback: passed to signal generators
    legacy_output : bool
        If True (default), return Series of direction only (backward compat with existing callers).
        If False, return DataFrames with full metadata (stop, target, sweep_bar_ts, htf_confluence_tf).

    Returns
    -------
    (base_signals, open_signals) : tuple
        Each element is pd.Series (if legacy_output=True) or pd.DataFrame (if legacy_output=False)
    """
    # Step 1: base IFVG — full 2/day budget
    base = ifvg_signals(
        df,
        timeframe_minutes=timeframe_minutes,
        max_signals_per_day=2,
        swing_lookback=swing_lookback,
        htf_df=htf_df,
        htf_proximity_pts=htf_proximity_pts,
        legacy_output=legacy_output,
    )

    # Step 2: count base signals per calendar day
    base_counts: dict = {}
    if legacy_output:
        for ts, val in base.items():
            if val != 0:
                d = ts.date()
                base_counts[d] = base_counts.get(d, 0) + 1
    else:
        for ts, row in base.iterrows():
            if row['direction'] != 0:
                d = ts.date()
                base_counts[d] = base_counts.get(d, 0) + 1

    # Step 3: open variant gets remaining budget
    open_var = ifvg_open_signals(
        df,
        timeframe_minutes=timeframe_minutes,
        max_signals_per_day=2,
        swing_lookback=swing_lookback,
        htf_df=htf_df,
        htf_proximity_pts=htf_proximity_pts,
        pre_open_df=pre_open_df,
        sweep_window_end=sweep_window_end,
        entry_window_end=entry_window_end,
        _external_daily_counts=base_counts,
        legacy_output=legacy_output,
    )

    return base, open_var


# ─────────────────────────────────────────────
# QUICK SANITY CHECK (run standalone to verify)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os

    data_path = r'C:\Users\Luker\strategyLabbrain\data\mnq_5min_databento.csv'
    if not os.path.exists(data_path):
        print(f'Data file not found: {data_path}')
        exit(1)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')
    df = df.set_index('datetime')
    df = df.between_time('09:30', '15:00')

    print(f'Loaded {len(df)} bars from {df.index.min().date()} to {df.index.max().date()}')

    base, open_var = ifvg_combined(df, timeframe_minutes=5)

    print(f'\nBase IFVG signals: {(base != 0).sum()} total')
    print(f'  Long  (1): {(base == 1).sum()}')
    print(f'  Short (-1): {(base == -1).sum()}')
    print(f'\nOpen Variant signals: {(open_var != 0).sum()} total')
    print(f'  Long  (1): {(open_var == 1).sum()}')
    print(f'  Short (-1): {(open_var == -1).sum()}')

    # Check shared daily limit
    combined = (base != 0).astype(int) + (open_var != 0).astype(int)
    daily_totals = combined.groupby(combined.index.date).sum()
    violations = (daily_totals > 2).sum()
    print(f'\nDaily limit check: {violations} days exceeded 2 combined signals (should be 0)')

    # Print sample signals
    all_signals = pd.DataFrame({'base': base, 'open': open_var})
    triggered = all_signals[(all_signals['base'] != 0) | (all_signals['open'] != 0)].head(20)
    print(f'\nSample signals (first 20):\n{triggered}')

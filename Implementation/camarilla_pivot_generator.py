"""
Camarilla Pivot / Session Level Features Generator
===================================================
Pre-built implementation for Agent 1 to copy into ml/signal_generators.py
(or import directly into dataset_builder.py).

Do NOT rewrite this — copy as-is and integrate with the rest of the generators.

Covers:
  - compute_camarilla()      : H3/H4/S3/S4 from prior calendar day's OHLC
  - compute_session_levels() : running session H/L for Asia, London, Pre-Market, NY AM
  - compute_prev_day_week()  : previous day high/low/close, previous week high/low
  - compute_pivot_features() : full feature set combining all of the above

Strategy spec: strategyLabbrain/Strategies/Session Level Pivots.md
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CAMARILLA PIVOT LEVELS
# ─────────────────────────────────────────────

def compute_camarilla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Camarilla H3/H4/S3/S4 levels from the PRIOR calendar day's OHLC.

    CRITICAL implementation detail:
      - Levels for day N use day N-1's OHLC (high, low, close)
      - Levels are CONSTANT within each day — they do not change bar-by-bar
      - On the first day of the dataset, levels are NaN (no prior day available)
      - After weekends/holidays, use the most recent trading day's OHLC

    Common agent mistakes this prevents:
      ❌ Using shift(1) per bar instead of per calendar day
      ❌ Computing from current day's data
      ❌ Using a rolling window that incorrectly spans day boundaries

    Formula:
      H3 = prev_close + (prev_high - prev_low) × 0.275
      H4 = prev_close + (prev_high - prev_low) × 0.55
      S3 = prev_close - (prev_high - prev_low) × 0.275
      S4 = prev_close - (prev_high - prev_low) × 0.55

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex (tz-aware, America/New_York)

    Returns
    -------
    DataFrame with columns: H3, H4, S3, S4 aligned to df's index
    """
    # Group by calendar date to get daily OHLC
    daily = df.groupby(df.index.date).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })

    # Shift by 1 calendar day to get PRIOR day's values
    prev_high = daily['high'].shift(1)
    prev_low = daily['low'].shift(1)
    prev_close = daily['close'].shift(1)

    # Compute Camarilla levels from prior day
    prev_range = prev_high - prev_low
    daily_h3 = prev_close + prev_range * 0.275
    daily_h4 = prev_close + prev_range * 0.55
    daily_s3 = prev_close - prev_range * 0.275
    daily_s4 = prev_close - prev_range * 0.55

    # Map daily levels back to each bar within that day
    bar_dates = df.index.date
    result = pd.DataFrame(index=df.index)
    result['H3'] = bar_dates
    result['H4'] = bar_dates
    result['S3'] = bar_dates
    result['S4'] = bar_dates

    # Build lookup dicts for fast mapping
    h3_map = daily_h3.to_dict()
    h4_map = daily_h4.to_dict()
    s3_map = daily_s3.to_dict()
    s4_map = daily_s4.to_dict()

    result['H3'] = pd.Series(bar_dates, index=df.index).map(h3_map)
    result['H4'] = pd.Series(bar_dates, index=df.index).map(h4_map)
    result['S3'] = pd.Series(bar_dates, index=df.index).map(s3_map)
    result['S4'] = pd.Series(bar_dates, index=df.index).map(s4_map)

    return result


# ─────────────────────────────────────────────
# SESSION HIGH/LOW LEVELS
# ─────────────────────────────────────────────

# Canonical session times (from Session Level Pivots.md)
SESSION_TIMES = {
    'asia':       ('20:00', '02:00'),   # prior day 20:00 to 02:00
    'london':     ('02:00', '07:00'),
    'premarket':  ('07:00', '09:30'),
    'ny_am':      ('09:30', '12:00'),
}


def _compute_session_hl_for_day(
    day_df: pd.DataFrame,
    session_name: str,
    session_start: str,
    session_end: str,
) -> tuple:
    """
    Compute the completed session high and low for a single day.
    Returns (session_high, session_low) or (NaN, NaN) if no data in the session.
    """
    try:
        session_bars = day_df.between_time(session_start, session_end)
    except Exception:
        return np.nan, np.nan

    if len(session_bars) == 0:
        return np.nan, np.nan

    return session_bars['high'].max(), session_bars['low'].min()


def compute_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session high/low levels as features.

    ASIA SESSION FIX (CRITICAL):
      - Asia session for trading day D = prior calendar day 20:00 ET through current day 02:00 ET
      - Do NOT use between_time('20:00', '02:00') on a per-day slice; that wraps within one calendar day
      - Instead: explicitly slice prior-day 20:00–23:59 + current-day 00:00–02:00

    For completed sessions (Asia, London, Pre-Market):
      - These are fixed by the time the NY session opens at 09:30
      - Use the prior completed session's values

    For NY AM (running session):
      - At bar N, use max/min of bars 0..N-1 within the current day's NY AM window
      - NEVER include bar N itself (that would be lookahead)

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex (tz-aware, America/New_York)
         Must include pre-market bars if available for Asia/London levels

    Returns
    -------
    DataFrame with columns: asia_high, asia_low, london_high, london_low,
                            premarket_high, premarket_low, ny_am_high, ny_am_low
    """
    result = pd.DataFrame(index=df.index, dtype=float)

    # Initialize all columns
    for session in ['asia', 'london', 'premarket', 'ny_am']:
        result[f'{session}_high'] = np.nan
        result[f'{session}_low'] = np.nan

    # Process each day
    for day in sorted(df.index.normalize().unique()):
        day_mask = df.index.normalize() == day
        day_df = df[day_mask]

        if len(day_df) == 0:
            continue

        # ─────────────────────────────────────────────────────────────────
        # ASIA SESSION: prior-day 20:00–23:59 + current-day 00:00–02:00
        # ─────────────────────────────────────────────────────────────────
        prior_day = day - pd.Timedelta(days=1)
        prior_day_mask = df.index.normalize() == prior_day
        prior_day_df = df[prior_day_mask]

        asia_bars = []
        # Evening bars from prior calendar day (20:00–23:59)
        if len(prior_day_df) > 0:
            try:
                prior_evening = prior_day_df.between_time('20:00', '23:59')
                if len(prior_evening) > 0:
                    asia_bars.append(prior_evening)
            except Exception:
                pass
        # Early morning bars from current calendar day (00:00–02:00)
        try:
            current_early = day_df.between_time('00:00', '02:00')
            if len(current_early) > 0:
                asia_bars.append(current_early)
        except Exception:
            pass

        if len(asia_bars) > 0:
            asia_combined = pd.concat(asia_bars)
            asia_high = asia_combined['high'].max()
            asia_low = asia_combined['low'].min()
            result.loc[day_mask, 'asia_high'] = asia_high
            result.loc[day_mask, 'asia_low'] = asia_low

        # ─────────────────────────────────────────────────────────────────
        # LONDON SESSION: 02:00–07:00 (current day)
        # ─────────────────────────────────────────────────────────────────
        try:
            london_bars = day_df.between_time('02:00', '07:00')
            if len(london_bars) > 0:
                sh = london_bars['high'].max()
                sl = london_bars['low'].min()
                result.loc[day_mask, 'london_high'] = sh
                result.loc[day_mask, 'london_low'] = sl
        except Exception:
            pass

        # ─────────────────────────────────────────────────────────────────
        # PRE-MARKET SESSION: 07:00–09:30 (current day)
        # ─────────────────────────────────────────────────────────────────
        try:
            premarket_bars = day_df.between_time('07:00', '09:30')
            if len(premarket_bars) > 0:
                sh = premarket_bars['high'].max()
                sl = premarket_bars['low'].min()
                result.loc[day_mask, 'premarket_high'] = sh
                result.loc[day_mask, 'premarket_low'] = sl
        except Exception:
            pass

        # ─────────────────────────────────────────────────────────────────
        # NY AM: running high/low (backward-looking only)
        # ─────────────────────────────────────────────────────────────────
        ny_am_start, ny_am_end = SESSION_TIMES['ny_am']
        try:
            ny_am_bars = day_df.between_time(ny_am_start, ny_am_end)
        except Exception:
            continue

        if len(ny_am_bars) == 0:
            continue

        # For each bar in NY AM, compute running max/min of bars BEFORE it
        for i in range(len(ny_am_bars)):
            bar_idx = ny_am_bars.index[i]
            if i == 0:
                # First bar of session — no prior bars, use NaN
                result.at[bar_idx, 'ny_am_high'] = np.nan
                result.at[bar_idx, 'ny_am_low'] = np.nan
            else:
                # Use bars 0..i-1 (NOT including bar i — no lookahead)
                prior_bars = ny_am_bars.iloc[:i]
                result.at[bar_idx, 'ny_am_high'] = prior_bars['high'].max()
                result.at[bar_idx, 'ny_am_low'] = prior_bars['low'].min()

    return result


# ─────────────────────────────────────────────
# PREVIOUS DAY/WEEK LEVELS
# ─────────────────────────────────────────────

def compute_prev_day_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute previous day high/low/close and previous week high/low.

    - Previous day = prior calendar day's session data
    - Previous week = prior week's full data (Mon-Fri)
    - All use completed data only — never current day/week

    Parameters
    ----------
    df : OHLCV DataFrame with DatetimeIndex (tz-aware, America/New_York)

    Returns
    -------
    DataFrame with columns: prev_day_high, prev_day_low, prev_day_close,
                            prev_week_high, prev_week_low
    """
    result = pd.DataFrame(index=df.index, dtype=float)

    # Daily aggregation
    daily = df.groupby(df.index.date).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })

    prev_day_high = daily['high'].shift(1)
    prev_day_low = daily['low'].shift(1)
    prev_day_close = daily['close'].shift(1)

    # Map to bars
    bar_dates = df.index.date
    result['prev_day_high'] = pd.Series(bar_dates, index=df.index).map(prev_day_high.to_dict())
    result['prev_day_low'] = pd.Series(bar_dates, index=df.index).map(prev_day_low.to_dict())
    result['prev_day_close'] = pd.Series(bar_dates, index=df.index).map(prev_day_close.to_dict())

    # Weekly aggregation
    # Use ISO week number for grouping
    weekly = df.groupby([df.index.isocalendar().year, df.index.isocalendar().week]).agg({
        'high': 'max',
        'low': 'min',
    })
    weekly.index.names = ['year', 'week']

    # Shift by 1 week
    prev_week_high = weekly['high'].shift(1)
    prev_week_low = weekly['low'].shift(1)

    # Map to bars by (year, week)
    bar_year_week = list(zip(
        df.index.isocalendar().year,
        df.index.isocalendar().week,
    ))
    pw_high_map = prev_week_high.to_dict()
    pw_low_map = prev_week_low.to_dict()

    result['prev_week_high'] = [pw_high_map.get(yw, np.nan) for yw in bar_year_week]
    result['prev_week_low'] = [pw_low_map.get(yw, np.nan) for yw in bar_year_week]

    return result


# ─────────────────────────────────────────────
# FULL PIVOT FEATURE SET
# ─────────────────────────────────────────────

def compute_pivot_features(df: pd.DataFrame, atr_period: int = 14, expose_raw: bool = False) -> pd.DataFrame:
    """
    Compute all session level pivot features for the CNN.

    For each level, produces:
      - distance: (close - level) / ATR(14) — ATR-normalized
      - above: 1 if close > level, else 0 — binary flag

    Parameters
    ----------
    df           : OHLCV DataFrame with DatetimeIndex (tz-aware, America/New_York)
    atr_period   : ATR period for normalization (default 14)
    expose_raw   : If True, also include raw level columns (e.g., camarilla_h3, asia_high, prev_day_close).
                   Default False for backward compatibility. When True, adds all raw levels used for
                   distance computation so downstream code can read them directly instead of
                   reconstructing via distance * ATR.

    Returns
    -------
    DataFrame with all pivot feature columns. If expose_raw=True, also includes raw level columns.
    """
    # Compute ATR for normalization
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # Guard: if ATR is 0 or NaN, set to NaN (will be dropped in warmup)
    atr = atr.replace(0, np.nan)

    # Get all level DataFrames
    camarilla = compute_camarilla(df)
    sessions = compute_session_levels(df)
    prev_dw = compute_prev_day_week(df)

    close = df['close']
    result = pd.DataFrame(index=df.index)

    # ── Camarilla levels ──────────────────────────────────
    for level_name in ['H3', 'H4', 'S3', 'S4']:
        level = camarilla[level_name]
        col_base = level_name.lower()
        result[f'{col_base}_dist'] = (close - level) / atr
        result[f'{col_base}_above'] = (close > level).astype(int)
        if expose_raw:
            result[f'camarilla_{col_base}'] = level

    # ── Session levels ────────────────────────────────────
    for session_name in ['asia', 'london', 'premarket', 'ny_am']:
        for hl in ['high', 'low']:
            level = sessions[f'{session_name}_{hl}']
            result[f'{session_name}_{hl}_dist'] = (close - level) / atr
            if expose_raw:
                result[f'session_{session_name}_{hl}'] = level

    # ── Previous day levels ───────────────────────────────
    result['prev_day_high_dist'] = (close - prev_dw['prev_day_high']) / atr
    result['prev_day_low_dist'] = (close - prev_dw['prev_day_low']) / atr
    if expose_raw:
        result['prev_day_high'] = prev_dw['prev_day_high']
        result['prev_day_low'] = prev_dw['prev_day_low']
        result['prev_day_close'] = prev_dw['prev_day_close']

    # ── Previous week levels ──────────────────────────────
    result['prev_week_high_dist'] = (close - prev_dw['prev_week_high']) / atr
    result['prev_week_low_dist'] = (close - prev_dw['prev_week_low']) / atr
    if expose_raw:
        result['prev_week_high'] = prev_dw['prev_week_high']
        result['prev_week_low'] = prev_dw['prev_week_low']

    return result


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

    # ── Test Camarilla ─────────────────────────────────────
    cam = compute_camarilla(df)
    print(f'\n=== Camarilla Levels ===')
    print(f'NaN count (first day expected): H3={cam["H3"].isna().sum()}')

    # Verify H3 is constant within each day
    violations = 0
    for date, group in cam.groupby(df.index.date):
        h3_vals = group['H3'].dropna()
        if len(h3_vals) > 0 and h3_vals.nunique() > 1:
            violations += 1
            print(f'  ❌ H3 changed intraday on {date}!')
    print(f'Intraday constancy check: {violations} violations (should be 0)')

    # Verify H3 uses prior day's OHLC
    dates = sorted(cam.index.normalize().unique())
    print(f'\nPrior-day verification (first 5 days):')
    for i in range(1, min(len(dates), 6)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_day = df[df.index.normalize() == prev_date]
        curr_day = cam[cam.index.normalize() == curr_date]
        if len(prev_day) == 0 or len(curr_day) == 0:
            continue
        prev_h = prev_day['high'].max()
        prev_l = prev_day['low'].min()
        prev_c = prev_day['close'].iloc[-1]
        expected_h3 = prev_c + (prev_h - prev_l) * 0.275
        actual_h3 = curr_day['H3'].iloc[0]
        match = '✓' if abs(actual_h3 - expected_h3) < 1e-6 else '❌'
        print(f'  {curr_date.date()}: expected H3={expected_h3:.2f}, actual={actual_h3:.2f} {match}')

    # ── Test Full Pivot Features ──────────────────────────
    features = compute_pivot_features(df)
    print(f'\n=== Full Pivot Features ===')
    print(f'Columns ({len(features.columns)}): {list(features.columns)}')
    warmup = 200
    post_warmup = features.iloc[warmup:]
    nan_cols = post_warmup.columns[post_warmup.isna().any()]
    print(f'Columns with NaN after warmup ({warmup} bars): {list(nan_cols)}')
    print(f'Total NaN after warmup: {post_warmup.isna().sum().sum()}')

    # Sample values
    sample = features.iloc[warmup:warmup+5]
    print(f'\nSample values (bars {warmup}-{warmup+4}):')
    print(sample.to_string())

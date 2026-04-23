"""
TTM Squeeze Signal Generator
=============================
Pre-built implementation for Agent 1 to copy into ml/signal_generators.py.
Do NOT rewrite this — copy as-is and integrate with the rest of the generators.

Covers:
  - linreg()         : linear regression value at current bar (CRITICAL — do not substitute)
  - ttm_squeeze()    : full TTM Squeeze signal generator

Strategy spec: strategyLabbrain/Strategies/TTMSqueeze.md
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# LINEAR REGRESSION — CRITICAL IMPLEMENTATION
# ─────────────────────────────────────────────

def linreg(series: pd.Series, period: int) -> pd.Series:
    """
    Linear regression value at the current bar.

    For each bar i, fits a line (y = slope * x + intercept) to the last
    `period` values of `series`, and returns the fitted value at the
    rightmost point (x = period - 1).

    This is NOT:
      - A simple difference (series.diff())
      - A moving average (series.rolling().mean())
      - The slope (np.polyfit(...)[0])
      - Deviation from mean (series - series.rolling().mean())

    It IS: the predicted Y value at the current bar from a least-squares
    linear fit of the last `period` bars.

    Equivalent to ThinkOrSwim's LinReg() or TradingView's ta.linreg().
    """
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _linreg_val(y):
        if np.any(np.isnan(y)):
            return np.nan
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return slope * (period - 1) + intercept

    return series.rolling(period).apply(_linreg_val, raw=True)


# ─────────────────────────────────────────────
# TTM SQUEEZE SIGNAL GENERATOR
# ─────────────────────────────────────────────

def ttm_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 2.0,
    min_squeeze_bars: int = 5,
    momentum_period: int = 12,
    atr_period: int = 14,
) -> pd.Series:
    """
    Generate TTM Squeeze signals bar-by-bar.

    Signal values:
        1  = Long  (squeeze fires with positive, increasing momentum)
        -1 = Short (squeeze fires with negative, decreasing momentum)
        0  = No signal

    Parameters
    ----------
    df                : OHLCV DataFrame, DatetimeIndex in America/New_York, 09:30-15:00 only
    bb_period         : Bollinger Band period (default 20)
    bb_std            : Bollinger Band standard deviation multiplier (default 2.0)
    kc_period         : Keltner Channel period (default 20)
    kc_mult           : Keltner Channel ATR multiplier (default 2.0)
    min_squeeze_bars  : minimum consecutive squeeze bars before signal is valid (default 5)
    momentum_period   : period for momentum linreg calculation (default 12)
    atr_period        : ATR period for stop/target calculation (default 14)

    Notes
    -----
    - Entry fires on the bar where squeeze transitions from ON → OFF (squeeze releases)
    - Momentum must confirm direction: positive + rising for long, negative + falling for short
    - No max signals per day — the ML model learns regime filtering
    - Warmup = max(bb_period, kc_period, momentum_period) bars — no signals before this
    """
    close = df['close']
    high = df['high']
    low = df['low']

    signals = pd.Series(0, index=df.index, dtype=int)

    # ── Bollinger Bands ────────────────────────────────────────────
    bb_mid = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    # ── Keltner Channels ───────────────────────────────────────────
    kc_mid = close.ewm(span=kc_period, adjust=False).mean()

    # True Range for ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_kc = tr.rolling(kc_period).mean()

    kc_upper = kc_mid + kc_mult * atr_kc
    kc_lower = kc_mid - kc_mult * atr_kc

    # ── Squeeze Detection ──────────────────────────────────────────
    # squeeze_on = BBs are INSIDE KCs (compressed volatility)
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # ── Momentum Histogram ─────────────────────────────────────────
    # midpoint = (highest(high, momentum_period) + lowest(low, momentum_period)) / 2
    highest_high = high.rolling(momentum_period).max()
    lowest_low = low.rolling(momentum_period).min()
    midpoint = (highest_high + lowest_low) / 2

    # SMA of close over momentum_period
    sma_momentum = close.rolling(momentum_period).mean()

    # delta = close - average(midpoint, SMA)
    delta = close - ((midpoint + sma_momentum) / 2)

    # LINEAR REGRESSION of delta — this is the critical calculation
    momentum = linreg(delta, momentum_period)

    # Momentum direction: increasing or decreasing
    momentum_prev = momentum.shift(1)
    momentum_increasing = momentum > momentum_prev
    momentum_decreasing = momentum < momentum_prev

    # ── Squeeze Bar Counter ────────────────────────────────────────
    # Count consecutive bars where squeeze is ON
    squeeze_count = pd.Series(0, index=df.index, dtype=int)
    count = 0
    for i in range(len(df)):
        if squeeze_on.iloc[i]:
            count += 1
        else:
            count = 0
        squeeze_count.iloc[i] = count

    # ── Signal Generation ──────────────────────────────────────────
    # Signal fires when squeeze transitions from ON → OFF
    # AND the squeeze was active for at least min_squeeze_bars
    # AND momentum confirms direction

    squeeze_on_prev = squeeze_on.shift(1).fillna(False)
    squeeze_count_prev = squeeze_count.shift(1).fillna(0)

    for i in range(len(df)):
        # Squeeze must have just released (was ON, now OFF)
        if not (squeeze_on_prev.iloc[i] and not squeeze_on.iloc[i]):
            continue

        # Must have been squeezing for minimum bars
        if squeeze_count_prev.iloc[i] < min_squeeze_bars:
            continue

        # Check momentum confirmation
        mom = momentum.iloc[i]
        if pd.isna(mom):
            continue

        if mom > 0 and momentum_increasing.iloc[i]:
            signals.iloc[i] = 1   # Long
        elif mom < 0 and momentum_decreasing.iloc[i]:
            signals.iloc[i] = -1  # Short

    return signals


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

    signals = ttm_squeeze(df)

    print(f'\nTTM Squeeze signals: {(signals != 0).sum()} total')
    print(f'  Long  (1): {(signals == 1).sum()}')
    print(f'  Short (-1): {(signals == -1).sum()}')

    # Verify no signals in warmup period
    warmup = max(20, 12)  # max(bb_period, momentum_period)
    warmup_signals = (signals.iloc[:warmup] != 0).sum()
    print(f'\nWarmup check: {warmup_signals} signals in first {warmup} bars (should be 0)')

    # Print sample signals
    triggered = signals[signals != 0].head(20)
    print(f'\nSample signals (first 20):')
    for ts, val in triggered.items():
        direction = 'LONG' if val == 1 else 'SHORT'
        print(f'  {ts} → {direction}')

    # Verify momentum linreg is not trivially zero or constant
    close = df['close']
    high = df['high']
    low = df['low']
    highest_high = high.rolling(12).max()
    lowest_low = low.rolling(12).min()
    midpoint = (highest_high + lowest_low) / 2
    sma_momentum = close.rolling(12).mean()
    delta = close - ((midpoint + sma_momentum) / 2)
    momentum = linreg(delta, 12)
    mom_valid = momentum.dropna()
    print(f'\nMomentum linreg stats:')
    print(f'  Mean:  {mom_valid.mean():.4f}')
    print(f'  Std:   {mom_valid.std():.4f}')
    print(f'  Min:   {mom_valid.min():.4f}')
    print(f'  Max:   {mom_valid.max():.4f}')
    print(f'  NaN count: {momentum.isna().sum()} (should be ~{12 - 1} from warmup)')

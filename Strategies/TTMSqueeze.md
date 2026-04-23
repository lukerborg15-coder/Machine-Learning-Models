# TTM Squeeze

**Type:** Mean Reversion / Momentum — Volatility Compression Breakout
**Instrument:** MNQ
**Session:** 9:30 – 15:00 ET
**Timeframe:** 5min bars (primary test timeframe)
**Related:** [[ConnorsRSI2]], [[Liquidity Levels]]

---

## Concept

The TTM Squeeze (by John Carter) identifies periods when Bollinger Bands (BB) compress inside Keltner Channels (KC) — a "squeeze." This compression indicates a coiling market where volatility has contracted and a directional move is building. When the BBs expand back outside the KCs, the squeeze releases and a momentum move begins. A momentum histogram (similar to MACD histogram) determines the direction of the trade.

The logic: BB inside KC = low volatility coil. BB crosses outside KC = coil releases. Trade in the direction the momentum histogram is pointing when the squeeze fires.

---

## Bollinger Bands

```
bb_period = 20
bb_std = 2.0
BB_Upper = SMA(20) + 2.0 × StdDev(20)
BB_Lower = SMA(20) − 2.0 × StdDev(20)
BB_Width  = BB_Upper − BB_Lower
```

---

## Keltner Channels

```
kc_period = 20
kc_mult = 2.0
KC_Upper = EMA(20) + 2.0 × ATR(20)
KC_Lower = EMA(20) − 2.0 × ATR(20)
```

---

## Squeeze Detection

```
squeeze_on = (BB_Upper < KC_Upper) AND (BB_Lower > KC_Lower)
```

When `squeeze_on = True`, the BBs are contained inside the KCs — the market is coiling.

When `squeeze_on` transitions from True → False, the squeeze **fires** — BBs have expanded outside KCs.

**Minimum squeeze duration:** `min_squeeze_bars = 5`. Only trade a squeeze that has been building for at least 5 bars. Short squeezes (1–3 bars) are noise.

---

## Momentum Histogram

The momentum oscillator determines direction after the squeeze fires:

```python
momentum_period = 12
# Linreg momentum: linear regression of (close - midpoint) over momentum_period bars
midpoint = (highest(high, momentum_period) + lowest(low, momentum_period)) / 2
delta = close - ((midpoint + SMA(close, momentum_period)) / 2)
momentum = linreg(delta, momentum_period)
```

### ⚠️ `linreg()` Implementation — CRITICAL

`linreg(series, period)` computes a **linear regression** over the last `period` bars of `series` and returns the **value of the regression line at the current bar** (i.e., the predicted Y at x = period - 1). This is NOT a simple difference, NOT a moving average, and NOT a slope.

**Correct Python implementation:**
```python
import numpy as np

def linreg(series, period):
    """
    Linear regression value at the current bar.
    For each bar i, fit a line to series[i-period+1 : i+1]
    and return the fitted value at x = period - 1 (the current bar).
    
    Equivalent to: np.polyval(np.polyfit(x, y, 1), period - 1)
    """
    result = pd.Series(np.nan, index=series.index)
    x = np.arange(period)
    for i in range(period - 1, len(series)):
        y = series.values[i - period + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        result.iloc[i] = slope * (period - 1) + intercept
    return result
```

**Vectorized alternative (faster):**
```python
def linreg_vectorized(series, period):
    """Vectorized linear regression using rolling apply."""
    x = np.arange(period)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def _linreg_val(y):
        slope = ((x - x_mean) * (y - y.mean())).sum() / x_var
        intercept = y.mean() - slope * x_mean
        return slope * (period - 1) + intercept
    
    return series.rolling(period).apply(_linreg_val, raw=True)
```

Do NOT substitute any of the following — they are common mistakes that produce subtly wrong signals:
- ❌ `series.diff()` — this is a 1-bar difference, not a regression
- ❌ `series.rolling(period).mean()` — this is a moving average, not a regression
- ❌ `np.polyfit(x, y, 1)[0]` — this returns the slope, not the regression value
- ❌ `series - series.rolling(period).mean()` — this is a deviation from mean, not regression

- `momentum > 0` and **increasing** → Long signal
- `momentum < 0` and **decreasing** → Short signal

The momentum must be moving in the direction of the trade on the bar the squeeze fires — do not trade a squeeze that fires with momentum pointing the wrong direction.

---

## Entry

Enter on the bar where `squeeze_on` transitions from True → False (squeeze fires), provided:
1. Squeeze was active for at least `min_squeeze_bars` consecutive bars
2. Momentum histogram confirms direction (positive + rising for long, negative + falling for short)

---

## Stop Loss

```
stop = entry ± 2.0 × ATR(14)
Long:  stop = entry − 2.0 × ATR(14)
Short: stop = entry + 2.0 × ATR(14)
```

---

## Target

```
target = entry + 2.0 × ATR(14)
```

Default target_mult = 2.0. The squeeze typically produces a measured move approximately equal to the compression width — ATR-based target captures this.

---

## Parameters to Backtest

| Parameter | Test Range |
|---|---|
| bb_period | 20 (fixed) |
| bb_std | 1.5, 2.0 |
| kc_period | 20 (fixed) |
| kc_mult | 1.5, 2.0, 2.5 |
| min_squeeze_bars | 3, 5, 8 |
| momentum_period | 12 (fixed) |
| stop_mult | 1.5, 2.0 |
| target_mult | 1.5, 2.0, 2.5 |

---

## Python Class Parameters

```python
class TTMSqueeze:
    def __init__(self,
                 bb_period=20,
                 bb_std=2.0,
                 kc_period=20,
                 kc_mult=2.0,
                 min_squeeze_bars=5,
                 momentum_period=12,
                 stop_mult=2.0,
                 target_mult=2.0,
                 atr_period=14)
```

---

## Logic Gaps and Fixes

**Warmup period**: BB requires `bb_period` bars, KC requires `kc_period` bars, ATR requires `atr_period` bars. Minimum warmup = `max(bb_period, kc_period, atr_period, momentum_period)` = 20 bars. No signals before this.

**Squeeze counter reset**: If squeeze_on goes True → False → True (squeeze fires then re-enters), reset the `squeeze_bars` counter. The new compression period must build its own minimum bar count before the next signal.

**Momentum direction check**: "Increasing" means the current momentum value is greater than the previous bar's momentum value. Not just that it is positive. Same logic for "decreasing" on short side.

**ATR at stop calculation**: Use ATR(14) — not ATR(kc_period=20). These are separate indicators. The stop uses the shorter ATR period for faster responsiveness to current volatility.

**Multiple signals in one day**: No hard max per day for TTMSqueeze, but the ML model can learn to skip setups in unfavorable regimes based on time of day, prior trade outcome, etc.

---

## Notes

- TTMSqueeze works best on trending days where compression breaks into a sustained move
- On range days, the squeeze fires but reverses quickly — the CNN is expected to learn this regime distinction from OHLCV and session context features
- Pairs with [[ConnorsRSI2]] as the two mean-reversion candidates — they are not identical strategies despite both being classified as "mean reversion"; TTMSqueeze is more of a volatility compression breakout, ConnorsRSI2 is a short-term overbought/oversold fade

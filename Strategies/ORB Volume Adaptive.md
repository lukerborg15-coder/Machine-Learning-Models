# ORB — Volume Adaptive

**Type:** Opening Range Breakout — Volume-Confirmation Filter
**Instrument:** MNQ
**Session:** 9:30 – 11:00 ET
**Timeframe:** 5min bars
**Related:** [[ORB IB - Initial Balance]], [[ORB Volatility Filtered]], [[ORB Wick Rejection]]

---

## Concept

A volume-adaptive ORB that only takes breakouts confirmed by above-average volume on the breakout candle itself. Where the [[ORB Volatility Filtered]] filters on market regime (ATR percentile), this strategy filters on the immediate conviction of the breakout bar. A breakout that occurs on low volume is likely a false move — institutions aren't participating. A breakout on high relative volume means the move has genuine participation behind it.

The "adaptive" element: the volume threshold is not a fixed number but a rolling multiple of the average volume during the opening range itself. This makes the filter self-calibrating — on busy days the threshold is higher, on quiet days it's lower.

---

## Opening Range Definition

| Parameter | Value |
|---|---|
| OR Start | 9:30 ET |
| OR End | 9:40 ET (first 10 minutes) |
| OR High | Highest high across 5min bars from 9:30–9:40 |
| OR Low | Lowest low across 5min bars from 9:30–9:40 |
| OR Range | OR High − OR Low |
| OR Avg Volume | Average volume of the bars that formed the OR |

---

## Volume Filter

The breakout candle must have volume above a multiple of the average OR volume:

```
or_avg_volume = mean(volume of all bars from 9:30–9:40)
breakout_bar_volume = volume of the candle that closes above/below OR

Signal valid if: breakout_bar_volume >= or_avg_volume × vol_multiplier
```

**Default:** `vol_multiplier = 1.5` (breakout bar must have 1.5× the average OR volume)

**Rationale:** The OR sets the baseline for how active the session is. A breakout bar with 1.5× that volume shows that participation accelerated on the move — consistent with institutional conviction. A breakout bar at or below OR volume suggests weak follow-through.

---

## Entry Conditions

- Price **closes** above OR High AND `breakout_volume >= or_avg_volume × 1.5` → **Long**
- Price **closes** below OR Low AND `breakout_volume >= or_avg_volume × 1.5` → **Short**
- Only 1 signal per day (first valid breakout wins)
- Entry on the close of the breakout candle
- Signal window: 9:40–11:00 ET

---

## Stop Loss

```
Long:  stop = max(OR Low, entry − 1.5 × ATR(14))
Short: stop = min(OR High, entry + 1.5 × ATR(14))
```

---

## Target

```
target = entry + 1.0 × OR Range
target_mult = 1.0 (test range: 1.0, 1.5, 2.0)
```

---

## Parameters to Backtest

| Parameter | Test Range |
|---|---|
| or_minutes | 10 (fixed) |
| vol_multiplier | 1.2, 1.5, 2.0 |
| stop_atr_mult | 1.0, 1.5, 2.0 |
| target_mult | 1.0, 1.5, 2.0 |
| atr_period | 14 (fixed) |

---

## Python Signal Generator Spec

```python
def orb_volume_adaptive(df, or_minutes=10, vol_multiplier=1.5,
                         stop_atr_mult=1.5, target_mult=1.0,
                         atr_period=14, max_signals_per_day=1):
    """
    Returns Series of 1 (long), -1 (short), 0 (no signal).

    Steps per day:
    1. Identify OR bars: 09:30 to 09:30 + or_minutes
    2. Compute OR High, OR Low, OR avg volume
    3. For each bar after OR closes (09:40 onward):
       - If close > OR High and bar volume >= or_avg_volume * vol_multiplier → signal = 1
       - If close < OR Low and bar volume >= or_avg_volume * vol_multiplier → signal = -1
       - Break after first signal (max 1/day)
    4. ATR stop and target are metadata — signal is just direction
    """
```

---

## Logic Gaps and Edge Cases

**OR has only 1 bar of data:** If data is missing and the OR window contains fewer than 2 bars, skip the day — `or_avg_volume` from a single bar is not meaningful.

**OR avg volume is zero:** Guard against zero-volume OR (data gaps). Skip day if `or_avg_volume == 0`.

**ATR warmup:** ATR(14) requires 14 prior bars. Skip signals where ATR is NaN.

**Breakout on same bar as OR close:** The OR closes at 9:40. The first valid breakout signal can only fire on the 9:40 bar close or later — not during the OR window itself.

**Volume spike from news:** On major economic release days (NFP, CPI at 8:30 ET), the 9:30 open can have anomalously high volume, making the OR avg volume very high and the 1.5× threshold harder to reach. This is intentional — on chaotic news days, the volume filter self-calibrates to be stricter.

---

## How This Differs from the Other ORBs

| Strategy | Filter | What it catches |
|---|---|---|
| [[ORB Volatility Filtered]] | ATR regime filter | Only trades in healthy volatility environments |
| [[ORB Wick Rejection]] | Body pct of breakout candle | Ensures directional conviction on the candle itself |
| [[ORB Volume Adaptive]] | Volume vs OR avg | Ensures institutional participation on the breakout |
| [[ORB IB - Initial Balance]] | 60min range | Uses wider, more significant range level |

These four filters catch different failure modes of the raw ORB. The ML model can learn which filter(s) produce the best signal quality across different market regimes.

---

## Notes

- Volume Adaptive pairs well with Wick Rejection — a high-volume body candle breakout is a very high-conviction signal
- The multiplier of 1.5× is a starting point. If MNQ consistently shows breakouts at 1.2× in backtests, lower the threshold
- Like all ORBs, pairs with [[Liquidity Levels]]: a volume-confirmed breakout that also clears a previous day high/low is the highest probability setup

# ORB — Volatility Filtered ⭐ TOP PERFORMER

**Type:** Opening Range Breakout — Volatility Regime Filter
**Instrument:** MNQ
**Session:** 9:30 – 11:00 ET
**Timeframe:** 5min bars
**Tested Result:** One of the two best performing ORB variants (alongside [[ORB IB - Initial Balance]])
**Related:** [[ORB IB - Initial Balance]], [[ORB Wick Rejection]], [[Liquidity Levels]]

---

## Concept

A volatility-filtered ORB that only fires when the current market volatility is within a healthy operating range. The filter uses ATR percentile rank: if today's ATR is too low (compressed, choppy day) or too high (news event, violent whipsaw), the setup is skipped. This produces a regime filter that ensures you are trading breakouts during the conditions most favorable to momentum continuation.

The Opening Range is a fixed 10-minute window (9:30–9:40). The ATR percentile filter is applied as a gate before any signal is considered.

---

## Opening Range Definition

| Parameter | Value |
|---|---|
| OR Start | 9:30 ET |
| OR End | 9:40 ET (first 10 minutes) |
| OR High | Highest high across all 5min bars from 9:30–9:40 |
| OR Low | Lowest low across all 5min bars from 9:30–9:40 |
| OR Range | OR High − OR Low |

---

## Volatility Filter

Before any signal is considered, check ATR percentile rank:

```
atr_period = 14
atr_lookback = 100 bars (rolling window for percentile calculation)
min_atr_pct = 25th percentile  ← below this = too compressed, skip
max_atr_pct = 85th percentile  ← above this = too volatile, skip
```

**Calculation:**
1. Compute ATR(14) on the 5min chart
2. Collect the last 100 ATR values (rolling window)
3. Calculate the percentile rank of the current ATR within that window
4. Only trade if: `25th pct ≤ current ATR ≤ 85th pct`

**Why these bounds:**
- Below 25th pct: market is compressed, range is too small, breakouts are false more often than not
- Above 85th pct: market is in a news-driven or chaotic regime, breakouts overshoot then violently reverse, stops get hit before targets

---

## Entry Conditions

- Price **closes** above OR High → **Long**
- Price **closes** below OR Low → **Short**
- Only 1 signal per day (first valid breakout wins)
- Entry on the close of the breakout candle
- Signal window: 9:40–11:00 ET (breakout after OR closes)
- ATR percentile filter must be satisfied at entry time

### Range Quality Filter
Minimum OR range = `0.0 × ATR(14)` (no minimum by default — volatility filter handles this implicitly)

---

## Stop Loss

```
Long:  stop = max(OR Low, entry − 1.5 × ATR(14))
Short: stop = min(OR High, entry + 1.5 × ATR(14))
```

The OR boundary serves as the structural stop; the ATR-based floor/ceiling prevents an excessively wide stop when the OR is very narrow.

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
| stop_atr_mult | 1.0, 1.5, 2.0 |
| target_mult | 1.0, 1.5, 2.0 |
| min_atr_pct | 20, 25, 30 |
| max_atr_pct | 75, 80, 85 |
| atr_lookback | 80, 100, 120 |

---

## Python Class Parameters

```python
class VolatilityFilteredORB:
    def __init__(self,
                 or_minutes=10,
                 stop_atr_mult=1.5,
                 target_mult=1.0,
                 atr_period=14,
                 atr_lookback=100,
                 min_atr_pct=25,
                 max_atr_pct=85,
                 max_signals_per_day=1,
                 min_range_atr=0.0)
```

---

## Logic Gaps and Edge Cases

**ATR warmup**: ATR(14) requires 14 prior bars to be valid. The first 14 bars of the dataset produce NaN ATR — guard against using these as signals. Warmup period = `max(atr_period, atr_lookback)` bars.

**Percentile with insufficient history**: Before 100 bars of ATR history accumulate, percentile rank is unreliable. Skip signals until `len(atr_history) >= atr_lookback`.

**OR closes on NaN ATR**: If ATR is NaN during OR formation, skip the day entirely.

**Session boundary**: OR must contain at least 2 complete 5min bars (9:30 and 9:35 close). If a bar is missing due to data gaps, skip that day.

---

## Notes

- Pairs well with [[Liquidity Levels]]: a breakout that also clears a previous day high/low is a much stronger setup
- On major macro release days (NFP, CPI, FOMC), ATR can spike beyond the 85th pct filter and the strategy correctly self-excludes
- The 10-minute OR is intentionally short — it captures the initial auction without absorbing too much directional price action

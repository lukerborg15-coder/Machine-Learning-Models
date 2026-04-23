# ORB — Wick Rejection ⭐ TOP PERFORMER

**Type:** Opening Range Breakout — Momentum Quality Filter
**Instrument:** MNQ
**Session:** 9:30 – 11:00 ET
**Timeframe:** 5min bars
**Tested Result:** One of the top performing ORB variants
**Related:** [[ORB IB - Initial Balance]], [[ORB Volatility Filtered]], [[Liquidity Levels]]

---

## Concept

A wick rejection filter applied to the OR breakout candle. The filter requires that the breakout candle have a **real body** of at least a minimum percentage of the total bar range. This ensures you are entering on candles with genuine directional conviction — not a wick spike that momentarily closes beyond the OR boundary before reversing.

A long upper wick on a long signal means sellers pushed back aggressively — the "breakout" was rejected. The wick filter eliminates these entries. A strong body candle closing above the OR high means buyers controlled the close with minimal rejection.

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

## Wick Rejection Filter

For a **Long signal** (close above OR High):
```
body = abs(close - open)
bar_range = high - low
body_pct = body / bar_range
```
Signal is valid only if: `body_pct >= min_body_pct`

**Default:** `min_body_pct = 0.55` (body must be at least 55% of total bar range)

For a **Short signal** (close below OR Low): same calculation. The body must dominate the bar — not a wick spike down that recovered.

**Edge case — zero range bar:** If `high == low` (doji with no range), `bar_range = 0` causes division by zero. Guard: if `bar_range == 0`, skip the signal entirely (no body, no valid breakout).

---

## Entry Conditions

- Price **closes** above OR High → check body_pct ≥ min_body_pct → **Long**
- Price **closes** below OR Low → check body_pct ≥ min_body_pct → **Short**
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
| min_body_pct | 0.45, 0.55, 0.65 |
| stop_atr_mult | 1.0, 1.5, 2.0 |
| target_mult | 1.0, 1.5, 2.0 |

---

## Python Class Parameters

```python
class WickRejectionORB:
    def __init__(self,
                 or_minutes=10,
                 stop_atr_mult=1.5,
                 target_mult=1.0,
                 atr_period=14,
                 max_signals_per_day=1,
                 min_body_pct=0.55)
```

---

## Logic Gaps and Edge Cases

**Zero range bar**: `high == low` on the breakout candle → `bar_range = 0` → division by zero. Guard: `if (high - low) == 0: skip signal`.

**ATR warmup**: ATR(14) requires 14 prior bars. Skip any signal where ATR is NaN.

**Body direction**: The body should be in the direction of the breakout. A close above OR High with a red candle (close < open) is a bearish candle that happened to close above OR High — still a valid close, but directional body confirmation is optional as an additional filter.

**Minimum body threshold**: At min_body_pct = 0.55, a candle with 45% wick is excluded. On volatile news opens, candles may have wider wicks — this filter naturally reduces news-day entry risk.

---

## Notes

- Wick rejection is a complementary filter to the volatility filter — combining both (VolatilityFilteredORB + WickRejectionORB logic) as a single strategy is a valid test
- Pairs well with [[Liquidity Levels]]: a wick rejection breakout that also clears a structural level is highest probability
- The IB strategy ([[ORB IB - Initial Balance]]) does not use this filter — it relies on the 60-minute range to self-filter quality

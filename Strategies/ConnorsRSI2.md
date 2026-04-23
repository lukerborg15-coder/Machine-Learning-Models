# ConnorsRSI2

**Type:** Mean Reversion — Short-Term Overbought/Oversold Fade
**Instrument:** MNQ
**Session:** 9:30 – 15:00 ET
**Timeframe:** 5min bars (primary test timeframe)
**Related:** [[TTMSqueeze]], [[Liquidity Levels]]

---

## Concept

ConnorsRSI2 is a short-term mean reversion strategy using RSI with a very short period (2). RSI(2) is extremely sensitive — it oscillates between near-0 and near-100 frequently, giving "oversold" and "overbought" readings on short pullbacks within trends. The strategy fades short-term exhaustion while staying in the direction of the longer-term trend (200 SMA filter). The exit is aggressive: once RSI(2) recovers to the overbought/oversold exit level (or a fast MA is crossed), the trade closes.

---

## Trend Filter

```
trend_ma = SMA(200)
```

- Only go **Long** when `close > SMA(200)` — uptrend
- Only go **Short** when `close < SMA(200)` — downtrend

The 200 SMA on a 5min chart is a significant moving average covering approximately 16+ hours of bar data. It represents the dominant intraday trend.

**Critical**: The 200 SMA requires 200 bars of warmup. On a 5min chart, that is 1000 minutes = ~16.7 hours of data. At the start of a session, there is sufficient history from prior days as long as the rolling dataset is loaded. Never slice data to just today's bars when computing the 200 SMA.

---

## Entry Signal

```
rsi_period = 2
entry threshold = 10  (oversold for long)
```

**Long entry:**
1. `close > SMA(200)` (uptrend)
2. `RSI(2) < 10` (short-term oversold pullback within uptrend)
3. Enter on the close of that bar

**Short entry:**
1. `close < SMA(200)` (downtrend)
2. `RSI(2) > 90` (short-term overbought bounce within downtrend)
3. Enter on the close of that bar

---

## Exit Signal

```
exit RSI threshold: RSI(2) > 90 for long exits (overbought recovery)
                    RSI(2) < 10 for short exits (oversold recovery)
exit_ma = SMA(5)  ← fast exit trigger
```

**Long exit (whichever comes first):**
- `RSI(2) > 90` — mean reversion complete, momentum recovered
- `close > SMA(5)` — price crossed above fast MA (momentum confirmed)
- Stop hit

**Short exit (whichever comes first):**
- `RSI(2) < 10` — mean reversion complete
- `close < SMA(5)` — price crossed below fast MA
- Stop hit

---

## Stop Loss

```
stop = entry ± 1.5 × ATR(14)
Long:  stop = entry − 1.5 × ATR(14)
Short: stop = entry + 1.5 × ATR(14)
```

---

## Target

```
target = entry + 1.0 × ATR(14)
```

ConnorsRSI2 is a fast mean-reversion trade. The target is intentionally modest — the edge comes from high win rate on tight targets, not large R multiples.

---

## Parameters to Backtest

| Parameter | Test Range |
|---|---|
| rsi_period | 2 (fixed) |
| rsi_entry | 5, 10, 15 |
| rsi_exit | 85, 90, 95 |
| exit_ma | 3, 5, 8 |
| trend_ma | 200 (fixed) |
| stop_mult | 1.0, 1.5, 2.0 |
| target_atr_mult | 0.75, 1.0, 1.5 |

---

## Python Class Parameters

```python
class ConnorsRSI2:
    def __init__(self,
                 rsi_period=2,
                 rsi_entry=10,
                 rsi_exit=90,
                 exit_ma=5,
                 trend_ma=200,
                 stop_mult=1.5,
                 target_atr_mult=1.0,
                 atr_period=14)
```

---

## Logic Gaps and Fixes

**200 SMA warmup is the longest warmup in the entire system**: 200 bars on 5min = 1000 minutes of data required. When loading data for any day, always include at least 200 bars from prior sessions. Never start from the first bar of the current day's open without trailing session data.

**RSI(2) computation**: RSI(2) with only 2 bars of history is extremely volatile. It can swing 0→100 in a single bar. This is intentional. Do not smooth it or add additional filters — the sensitivity is the edge.

**Consecutive RSI signals**: Multiple consecutive bars can show RSI(2) < 10. Only take the **first** entry signal in each sequence. Require RSI to recover above 10 (exit threshold) before re-entering long. Track `in_trade` state carefully.

**Off-by-one on SMA(5) exit**: The exit fires when `close > SMA(5)`. The SMA(5) on bar N uses bars N-4 through N. Make sure to use the completed SMA value at bar N, not a forward-looking average.

**ATR at stop and target**: ATR(14) requires 14 bars of True Range history. Guard against NaN ATR on early bars.

**SMA(200) and SMA(5) both need warmup**: Warmup = `max(trend_ma, exit_ma, rsi_period, atr_period)` = 200 bars minimum before any signal.

---

## Notes

- ConnorsRSI2 has a fundamentally different character from the ORB and IFVG strategies — it fades short-term exhaustion rather than trading breakouts or reversals. The CNN can weight signals from all strategy types together for ensemble-style prediction
- Mean reversion strategies generally have higher win rates but lower average R per trade. The CNN's goal is to identify which RSI(2) dips are genuine pullbacks vs. the start of a larger trend move (which should be avoided)
- Works best in trending conditions where pullbacks are shallow and recover quickly — the 200 SMA filter enforces this regime requirement

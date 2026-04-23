# ORB IB — Initial Balance ⭐ TOP PERFORMER

**Type:** Opening Range Breakout — Initial Balance
**Instrument:** MNQ
**Session:** 9:30 – 11:00 ET (IB forms 9:30–10:30, breakout after)
**Timeframe:** 5min bars
**Tested Result:** One of the two best performing ORB variants
**Related:** [[ORB Volume Adaptive]], [[ORB Volatility Filtered]], [[ORB Wick Rejection]], [[Liquidity Levels]]

---

## Concept

The Initial Balance (IB) is a concept borrowed from Market Profile theory. It defines the high and low established during the **first 60 minutes** of the regular session (9:30–10:30). The IB represents the range where the market does its initial price discovery and establishes value for the day. A breakout beyond the IB high or low — especially with momentum — signals that the market has found directional conviction and is ready to extend beyond the opening range.

The IB is wider than the 10-minute ORB variants by design. It absorbs more noise and volatility, which means the breakout signal is less frequent but carries higher conviction. The IB High and Low also act as natural extension targets — traders using Market Profile methodology expect price to reach 1.5× the IB range beyond the breakout level on trending days.

---

## Initial Balance Definition

| Parameter | Value |
|---|---|
| IB Start | 9:30 ET |
| IB End | 10:30 ET (first 60 minutes) |
| IB High | Highest high across all bars from 9:30–10:30 |
| IB Low | Lowest low across all bars from 9:30–10:30 |
| IB Range | IB High − IB Low |
| IB Extension Target | IB High/Low ± (1.5 × IB Range) |

---

## Entry Conditions

- Price **closes** above IB High → **Long**
- Price **closes** below IB Low → **Short**
- Only 1 signal per day (first valid breakout wins)
- Entry on the close of the breakout candle
- Signal window: 10:30–11:00 ET (first breakout after IB closes)

### Range Quality Filter
Minimum IB range = **ATR(14)**
The IB should be at least 1× ATR to represent meaningful structure. Narrow IBs indicate indecision and produce lower quality breakouts.

---

## Stop Loss

`stop = max(IB Low, entry − 1.5 × ATR(14))` for longs
`stop = min(IB High, entry + 1.5 × ATR(14))` for shorts

---

## Target

**Primary target:** measured from the broken IB level:
- Long: `IB High + extension_mult × IB Range`
- Short: `IB Low - extension_mult × IB Range`

Default extension_mult: **1.5** (IB extension level — a widely watched Market Profile target)

Canonical implementation-backed target origin: [StrategyLab/docs/INITIAL_BALANCE_ORB_STRATEGY.md](../../StrategyLab/docs/INITIAL_BALANCE_ORB_STRATEGY.md).

---

## Parameters to Backtest

| Parameter | Test Range |
|---|---|
| extension_mult | 1.0, 1.5, 2.0 |
| stop_atr_mult | 1.0, 1.5, 2.0 |
| min_ib_atr | 0.5, 0.75, 1.0 |

---

## Why the IB Works

The first 60 minutes of trading concentrates the most volume and the highest participation from institutional traders. The IB high and low are therefore levels that many market participants have observed and are aware of. A clean break of the IB — especially on expanding volume — signals that the day's dominant side has won the initial auction and is driving price away from value. This is one of the oldest and most studied intraday patterns in futures.

---

## IB Days vs Non-IB Days

| Day Type | Characteristic | IB Strategy |
|---|---|---|
| Trend day | Clean break of IB, no return | ✅ High win rate, large R |
| Normal day | Breaks IB then returns to range | ⚠️ Requires good stop placement |
| Neutral day | Never clearly breaks IB | ❌ No signal fires |

The IB strategy is self-filtering — on neutral range days, price simply never breaks the IB cleanly and no trade triggers.

---

## Notes

- The IB is a **delayed** strategy — entry cannot happen before 10:30, unlike the 10-minute ORB variants
- Pairs very well with [[Liquidity Levels]]: an IB breakout that also clears a previous day high/low is a much stronger setup
- On days with major economic releases at 10:00 ET (e.g. ISM, NFP), the 10:00 candle can distort the IB — consider flagging those days separately

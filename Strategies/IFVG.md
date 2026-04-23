# Inversion Fair Value Gap (IFVG)

**Type:** Price Action / Smart Money Concepts (SMC / ICT)
**Instrument:** MNQ
**Session:** 9:30 – 15:00 ET
**Max Setups/Day:** 2 (shared with [[IFVG - Open Variant]])
**Related:** [[IFVG - Open Variant]], [[Liquidity Levels]]

---

## Background — What Is a Fair Value Gap (FVG)?

Before understanding an IFVG, you must understand a **Fair Value Gap (FVG)**.

A Fair Value Gap is a **3-candle imbalance pattern** that appears on a price chart. It forms when the market moves so aggressively in one direction that it leaves a price range completely untouched — no buying and selling occurred there because the move was too fast. This gap represents an imbalance between buyers and sellers.

### How to Identify a FVG

Look at 3 consecutive candles:
- **Candle 1** — any candle
- **Candle 2** — the large, aggressive "impulse" candle
- **Candle 3** — the candle that follows

A **Bullish FVG** exists when:
- The **low of Candle 3** is **above** the **high of Candle 1**
- The gap between Candle 1's high and Candle 3's low is the FVG zone
- Price moved up so fast that the space between those two wicks was never traded
- This zone acts as potential **support** — buyers are expected to defend it if price returns

A **Bearish FVG** exists when:
- The **high of Candle 3** is **below** the **low of Candle 1**
- The gap between Candle 1's low and Candle 3's high is the FVG zone
- Price moved down so fast that the space was never traded
- This zone acts as potential **resistance** — sellers are expected to defend it if price returns

```
Bullish FVG Example (price moved up):
         ┌──┐
         │C2│  ← large bullish candle
    ┌──┐ │  │
    │C1│ │  │       ┌──┐
    └──┘ │  │       │C3│
    high └──┘  gap  └──┘ low
              ←FVG zone→
              (C1 high to C3 low — price never traded here)

Bearish FVG Example (price moved down):
    ┌──┐
    │C1│ low
    └──┘       ┌──┐
         ┌──┐  │C3│ high
         │C2│  └──┘
         │  │
         └──┘
              ←FVG zone→
              (C1 low to C3 high — price never traded here)
```

---

## What Is an Inversion Fair Value Gap (IFVG)?

An **Inversion FVG** forms when a Fair Value Gap is **violated** — price trades through it completely, invalidating its original bias. When this happens, the zone does not disappear. Instead, it **flips its role**: what was support becomes resistance, and what was resistance becomes support.

### Bullish IFVG Formation
1. A **Bearish FVG** exists on the chart (a resistance zone pointing downward)
2. Price rallies and a candle **wicks through or closes above** the top of the Bearish FVG
3. The Bearish FVG is now **inverted** — it has become a **Bullish IFVG**
4. The zone is now expected to act as **support** — if price pulls back into it, look for longs

### Bearish IFVG Formation
1. A **Bullish FVG** exists on the chart (a support zone pointing upward)
2. Price drops and a candle **wicks through or closes below** the bottom of the Bullish FVG
3. The Bullish FVG is now **inverted** — it has become a **Bearish IFVG**
4. The zone is now expected to act as **resistance** — if price rallies back into it, look for shorts

### Why Does Inversion Work?

When a FVG is violated, it signals that the original imbalance has been absorbed and the momentum has shifted. Traders who were positioned expecting the FVG to hold as support/resistance are now trapped on the wrong side. As price returns to the inverted zone, those trapped traders add selling/buying pressure at exactly the point where you want to enter — creating a self-fulfilling resistance/support zone.

---

## IFVG Validity

An IFVG zone remains active and tradeable until:
- A candle **closes through the zone in the opposite direction** to the inversion

Example: A Bullish IFVG (formed by inverting a Bearish FVG) is invalidated when a candle closes **below** the bottom of the zone. Once invalidated, do not use that zone for entries.

---

## Setup Requirements

All three conditions must be present before entering:

### Condition 1 — Liquidity Sweep (Required)
Before the IFVG forms, price must first **sweep a significant liquidity level** — meaning price wicks through or takes out a resting high or low where stop orders are clustered. This sweep triggers those orders, often causing the sharp reversal that creates the IFVG.

Valid liquidity levels (see [[Liquidity Levels]]):
- 1H high or low
- 4H high or low
- Session high or low (Asia, London, NY AM, NY Lunch, NY PM)
- Previous day high or low
- Previous week high or low

Without a liquidity sweep, the IFVG formation is lower probability — you are likely fading a continuation move rather than catching a reversal.

### Condition 2 — Higher Timeframe FVG Confluence (Required)
At the time the IFVG forms, price must be **inside or rejecting from a higher timeframe FVG zone**. This means price displaced aggressively into a HTF imbalance zone and then rapidly displaced back out — the IFVG forms during or immediately after this HTF rejection.

This is the most important filter. The HTF gap acts as the "reason" for the reversal — the IFVG is your precision entry within that larger move.

| Entry Timeframe | Required HTF Gap Rejection |
|---|---|
| 1min / 2min / 3min | 5min, 15min, 1H, or 4H FVG must be wicking/rejecting |
| 5min | 15min, 1H, or 4H FVG must be wicking/rejecting |

**What "rejection" looks like:** Price aggressively enters the HTF FVG zone (wicks deep into it or closes into it), then on the very next bar or within 1–3 bars, price displaces back out of the zone sharply. The IFVG on your entry timeframe forms during this displacement candle.

### Condition 3 — Gap Size Minimum
The original FVG that gets inverted must be large enough to be structurally significant. Small FVGs are noise — they form constantly and most do not hold.

| Entry Timeframe | Minimum FVG Size |
|---|---|
| 1min / 2min | 5 points (on MNQ) |
| 3min / 5min | 7 points (on MNQ) |

---

## Entry

**Enter on the close of the candle that forms the IFVG** — the candle that violates the original FVG and completes the inversion. You do not wait for a retest. The entry is immediate, on the close of that invalidation candle.

For a **Bullish IFVG**: the entry candle closes above the top of the original Bearish FVG → you are long at that close price.

For a **Bearish IFVG**: the entry candle closes below the bottom of the original Bullish FVG → you are short at that close price.

---

## Stop Loss

Place stop at the **swing low** (for longs) or **swing high** (for shorts) that was formed by the liquidity sweep immediately preceding the IFVG.

This swing is logical: it was the extreme of the manipulation move. If price returns to that level after the IFVG has formed, the setup is invalidated — the reversal did not hold.

---

## Break-Even Rule

When price reaches **50% of the distance to the target**, move the stop loss to the entry price (break-even). This protects the trade from turning a winner into a loser once momentum has been confirmed.

---

## Target

**1R to 1.5R** from entry.

Calculate 1R as: `entry price − stop price` (distance to stop). Target = `entry + 1R` to `entry + 1.5R`.

---

## Timeframes to Test

The ML agent will test 1min, 2min, 3min, and 5min to find the best performing entry timeframe. Each timeframe requires its own HTF confluence level per the table above.

---

## Full Setup Checklist (for AI implementation)

```
□ 1. A FVG has been identified on the entry timeframe (1/2/3/5min)
□ 2. A liquidity level has been swept before or as the FVG forms
□ 3. A higher timeframe FVG is being wicked/rejected at the same time
□ 4. The FVG size meets the minimum (5pts for 1/2min, 7pts for 3/5min)
□ 5. The current time is between 9:30 and 15:00 ET
□ 6. Fewer than 2 IFVG trades have been taken today
□ 7. The FVG has not already been invalidated (no close through it in opposite direction)
→ Enter on the close of the inversion candle
→ Stop at swing low/high
→ Move to break-even at 50% of target
→ Exit at 1–1.5R
```

---

## Notes

- The highest probability setups have ALL THREE conditions clearly present — liquidity sweep is clean, HTF gap is obvious, and the IFVG forms on a strong momentum candle
- Avoid setups where the liquidity sweep is questionable (barely clips the level) or the HTF FVG is small/old
- The 2 trades/day limit is strict — take only the cleanest setups, do not force entries
- After 14:30, volume thins and spread widens on MNQ — avoid late-session entries unless the setup is extremely clear

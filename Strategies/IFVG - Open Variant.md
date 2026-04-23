# IFVG — Open Variant

**Type:** Price Action / Smart Money Concepts (SMC / ICT)
**Instrument:** MNQ
**Session:** 9:30 – 15:00 ET (manipulation window: 9:30–9:45 ET, tunable)
**Max Setups/Day:** 2 (shared with [[IFVG]])
**Related:** [[IFVG]], [[Liquidity Levels]]

---

## Concept

The Open Variant targets the manipulation move that occurs at the 9:30 ET open. In the first 5 minutes, institutional players frequently push price through nearby highs or lows to trigger stop orders and fill their own positions at better prices. This sweep creates a sharp move in one direction followed by an immediate and aggressive reversal — the reversal leg is the IFVG that you trade.

The logic is identical to the base [[IFVG]]. The only difference is the timing constraint: the liquidity sweep must occur between 9:30 and 9:35 ET. This concentrates entries at the most manipulated moment of the session.

---

## Background — IFVG Mechanics

See [[IFVG]] for the full FVG and IFVG explanation.

**Summary:**
- FVG = 3-candle imbalance zone where price moved too fast to leave any traded price in that range
- When a FVG is violated (price closes through it), it **inverts** — former support becomes resistance and vice versa
- Bullish IFVG: Bearish FVG violated upward → zone becomes support → long
- Bearish IFVG: Bullish FVG violated downward → zone becomes resistance → short

---

## What Makes the Open Unique

The manipulation sweep at the open is structurally different from mid-session sweeps:

1. **Pre-positioned stops**: Overnight and pre-market traders have stops just above/below visible levels — at 9:30 with full volume, these are immediately hunted
2. **Speed**: The sweep and reversal happen within 1–5 bars on a 1min chart — the IFVG forms extremely fast
3. **Clarity**: Both the sweep and the IFVG are structurally obvious at the open — less ambiguity than mid-session setups

---

## Setup Requirements

All three conditions from base [[IFVG]] apply, plus the timing constraint:

### Condition 0 — Timing Gate (Open Variant Only)
The liquidity sweep must occur between **9:30 and 9:45 ET** (default). If the sweep happens at 9:46 or later, this is no longer an Open Variant — it may still be a valid base IFVG but not this variant.

The IFVG formation candle can close any time up to **10:00 ET**, provided the sweep that triggered it was within the sweep window. The 9:45 cutoff is the default because it captures the full opening-drive manipulation phase on MNQ without bleeding into mid-session chop. Tunable via `sweep_window_end` / `entry_window_end` parameters on `ifvg_open_signals()` — test 09:35 (strict original spec) vs 09:45 (default) vs 10:00 (wider) to find the hit-rate cliff.

Inversion detection works on any of the configured entry timeframes (1 / 2 / 3 / 5-min). On 5-min bars, the typical sequence is: sweep on the 9:30 or 9:35 bar → inversion candle closes on the 9:40 / 9:45 / 9:50 bar. On 1-min bars the whole sequence can complete inside the first 5–10 bars.

### Condition 1 — Liquidity Sweep (Required)
Price must sweep a significant liquidity level. Valid levels at the open (see [[Liquidity Levels]]):
- Previous day high or low
- Previous week high or low
- Overnight (Globex) session high or low
- Asia session high or low
- Pre-market high or low
- Any obvious structural high or low visible on the 5min or 15min chart

**The sweep size does not matter** — a single tick through the level qualifies. What matters is that resting stop orders were triggered.

### Condition 2 — Higher Timeframe FVG Confluence (Required)
At the time the IFVG forms, price must be inside or rejecting from a higher timeframe FVG zone.

| Entry Timeframe | Required HTF Gap Rejection |
|---|---|
| 1min / 2min / 3min | 5min, 15min, 1H, or 4H FVG must be wicking/rejecting |
| 5min | 15min, 1H, or 4H FVG must be wicking/rejecting |

### Condition 3 — Gap Size Minimum
The FVG that gets inverted must meet the minimum size:

| Entry Timeframe | Minimum FVG Size |
|---|---|
| 1min / 2min | 5 points (on MNQ) |
| 3min / 5min | 7 points (on MNQ) |

---

## Entry

**Enter on the close of the candle that forms the IFVG** — the candle that violates the original FVG and completes the inversion. No retest required. Entry is immediate.

- Bullish IFVG: entry candle closes above the top of the original Bearish FVG → long at that close
- Bearish IFVG: entry candle closes below the bottom of the original Bullish FVG → short at that close

---

## Stop Loss

Place stop at the **swing low** (for longs) or **swing high** (for shorts) created by the manipulation sweep. That extreme is the logical invalidation — if price returns there, the reversal failed.

---

## Break-Even Rule

When price reaches **50% of the distance to the target**, move stop to entry price (break-even).

---

## Target

**1R to 1.5R** from entry.

`1R = |entry − stop|`
`target = entry + 1R` to `entry + 1.5R`

---

## IFVG Validity

The IFVG zone remains active until a candle closes through it in the opposite direction of the inversion. Once invalidated, do not use that zone for entries.

---

## Setup Sequence

```
9:30 open
  ↓
Price sweeps a key level (prev day H/L, overnight H/L, pre-market H/L) — must be before 9:45 (default)
  ↓
Sharp reversal with momentum in the opposite direction
  ↓
A FVG on the entry timeframe (1/2/3/5-min) gets violated → IFVG forms
  ↓
Confirm HTF FVG is rejecting price at the same time
  ↓
Enter on the close of the IFVG formation candle (must close by 10:00 default)
```

---

## Full Setup Checklist (for AI implementation)

```
□ 1. The liquidity sweep occurred between 9:30 and 9:45 ET (default window; tunable)
□ 2. A significant liquidity level was swept (prev day H/L, overnight H/L, pre-market H/L, structural level)
□ 3. A FVG has been identified on the entry timeframe (1/2/3/5min)
□ 4. The FVG size meets the minimum (5pts for 1/2min, 7pts for 3/5min)
□ 5. A higher timeframe FVG is being wicked/rejected at the same time
□ 6. The FVG has not already been invalidated (no close through it in opposite direction)
□ 7. Fewer than 2 IFVG trades have been taken today (shared limit with base IFVG)
→ Enter on the close of the inversion candle
→ Stop at sweep swing low/high
→ Move to break-even at 50% of target
→ Exit at 1–1.5R
```

---

## Notes

- The 9:30–9:35 window means this setup fires in the first 1–5 bars on a 1min chart — the ML agent must timestamp the sweep trigger, not the IFVG formation close
- Sweep size does not matter — a single tick through the level qualifies
- The 2 trades/day limit is shared with base IFVG — if an Open Variant fires first, only 1 more IFVG trade remains for the session
- On high-impact news days (CPI, NFP, FOMC), opening sweeps are more violent — flag these days in the ML feature set as they may have different statistical properties
- If the manipulation leg extends past 9:35 without producing an IFVG, skip the variant — wait for a standard base IFVG setup later in the session

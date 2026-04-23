# Session Level Pivots Break

**Type:** Active entry strategy + feature engineering
**Instrument:** MNQ
**Timeframe:** 5min (primary), compatible with 1min/3min
**Signal:** Camarilla H4/S4 close-through continuation
**Signal column:** `session_pivot_break_signal`
**Max signals/day:** 2 (shared with Session Level Pivots rejection)
**Status:** Fully defined companion signal
**ONNX Model:** Model 3
**Related:** [[Session Level Pivots]], [[ORB Volatility Filtered]], [[Architecture Overview]]

---

## Role in the ML Pipeline

Session Level Pivots Break is the continuation companion to [[Session Level Pivots]]. It produces `session_pivot_break_signal` for Model 3, alongside `orb_vol_signal` and `session_pivot_signal`.

The signal captures the opposite behavior from pivot rejection. Instead of fading a defended Camarilla level, it follows the first close through the outer Camarilla boundary when price accepts beyond H4 or S4.

---

## Core Concept

Camarilla H4 and S4 are the outer daily pivot levels computed from the prior day's OHLC. Rejection at these levels is a mean-reversion setup; a close through them is a directional continuation setup.

This strategy only uses H4/S4. H3/S3 and session highs/lows remain part of the rejection strategy and the shared level feature set, but they are not break-signal triggers.

---

## Signal Generation Rules

### Long Signal (+1)

All conditions must be true on the same bar:

1. **Close-through:** Current bar closes above H4: `bar.close > camarilla_h4`
2. **First close through:** Prior bar close was at or below H4: `prev_close <= prev_camarilla_h4`
3. **ATR valid:** `atr_14` is present and greater than 0
4. **Daily cap:** Fewer than 2 total Session Level Pivot signals have fired today

### Short Signal (-1)

All conditions must be true on the same bar:

1. **Close-through:** Current bar closes below S4: `bar.close < camarilla_s4`
2. **First close through:** Prior bar close was at or above S4: `prev_close >= prev_camarilla_s4`
3. **ATR valid:** `atr_14` is present and greater than 0
4. **Daily cap:** Fewer than 2 total Session Level Pivot signals have fired today

Only one break signal fires per bar. If both directions somehow qualify because of bad or crossed level data, no directional conflict should be emitted; upstream level generation must be corrected.

---

## Relationship to Rejection Signal

`session_pivot_signal` and `session_pivot_break_signal` are mutually exclusive interpretations of the same outer pivot levels:

- **Rejection:** price touches or probes H4/S4, then closes back on the inside of the level.
- **Break:** price closes through H4/S4 after the prior bar was still on the inside of the level.

The rejection signal is a fade. The break signal is continuation. They must share the same daily pivot-signal budget, not receive separate 2-signal budgets.

---

## Required Inputs

The generator requires:

```python
close
atr_14
camarilla_h4 or h4_dist with atr_14 and close
camarilla_s4 or s4_dist with atr_14 and close
```

Camarilla levels must be computed from the prior day's confirmed OHLC only. Do not compute H4/S4 from current-day intraday data.

---

## Parameters

```python
ATR_PERIOD  = 14  # validation only; the break condition is level/close based
MAX_PER_DAY = 2   # shared Session Level Pivot cap
```

---

## Notes

- This signal is intended for Model 3 as a structural continuation input, not as a standalone replacement for the rejection setup.
- Because it uses a prior-close transition, it fires only on the first close through H4 or S4, not every bar that remains beyond the level.
- If raw Camarilla levels are absent, the implementation may reconstruct them from ATR-normalized distance features; this is acceptable only when `close` and `atr_14` are aligned to the same bar.

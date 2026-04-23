# Session Level Pivots

**Type:** Active entry strategy + feature engineering
**Instrument:** MNQ
**Timeframe:** 5min (primary), compatible with 1min/3min
**Signal:** Level-touch rejection at Camarilla H3/H4/S3/S4 and session highs/lows
**Max signals/day:** 2 (shared across all levels and the break companion)
**Status:** ✅ Fully defined
**ONNX Model:** Model 3 (paired with ORB IB)
**Companion Signal:** [[Session Level Pivots Break]] (`session_pivot_break_signal`)
**Related:** [[Liquidity Levels]], [[ORB IB - Initial Balance]], [[Architecture Overview]]

---

## Role in the ML Pipeline

Session Level Pivots serves two roles simultaneously:

1. **Active strategy:** Generates directional signals (`session_pivot_signal` column = +1/-1/0) that become training labels and live trade signals. Covered by Model 3 alongside ORB IB.
2. **Feature engineering:** Camarilla H3/H4/S3/S4 and session high/low distance columns are input features fed to ALL four models. Every model sees the level context whether or not it is responsible for the pivot signal.

This dual role is intentional. The pivot levels provide structural context that improves every strategy's decision-making, while the pivot signal itself captures the specific mean-reversion setup at key levels.

---

## Core Concept

Camarilla pivot levels define mathematically-derived intraday support and resistance zones computed from the prior day's OHLC. Approximately 70% of trading days price remains within the H3–S3 range, making H3/S3 natural mean-reversion zones. When price reaches H4 or S4, the setup becomes a high-probability rejection play — institutions are aware of these levels and frequently defend them.

The strategy fires when price touches a key level AND shows a rejection candle (closes back on the correct side of the level on the same bar). This confirms participation rather than just proximity.

---

## Camarilla Level Formulas

**Source: prior day's OHLC only. Never use current day's data.**

```python
prev_range = prev_high - prev_low

H4 = prev_close + prev_range * 0.55
H3 = prev_close + prev_range * 0.275
S3 = prev_close - prev_range * 0.275
S4 = prev_close - prev_range * 0.55
```

**Timing:** Levels are fixed at the start of each session using the confirmed prior day close (after 16:00 ET prior day). They do not update intraday.

**Common error to avoid:** Computing Camarilla from today's intraday data introduces lookahead. Always use yesterday's confirmed OHLC.

---

## Signal Generation Rules

### Long Signal (+1)

All four conditions must be true on the same bar:

1. **Level touch:** Bar low reaches within `proximity_atr * ATR(14)` of S4 OR S3 OR a key session low (Asia low, London low, Pre-Market low, previous day low)
2. **Rejection candle:** Bar closes ABOVE the touched level (close > level). A wick below the level is fine — the close must be above it. If close <= level, the bar is not a rejection, it is a breakdown.
3. **Context filter:** Current bar close is BELOW the prior day close (price is in a relative oversold/pullback context, mean reversion is plausible)
4. **Daily cap:** Fewer than 2 Session Level Pivot signals have fired today, counting both rejection and break companion signals

### Short Signal (-1)

All four conditions must be true on the same bar:

1. **Level touch:** Bar high reaches within `proximity_atr * ATR(14)` of H4 OR H3 OR a key session high (Asia high, London high, Pre-Market high, previous day high)
2. **Rejection candle:** Bar closes BELOW the touched level (close < level). A wick above is fine — the close must be below it. If close >= level, the bar is not a rejection, it is a breakout.
3. **Context filter:** Current bar close is ABOVE the prior day close (price is in a relative overbought context)
4. **Daily cap:** Fewer than 2 Session Level Pivot signals have fired today, counting both rejection and break companion signals

### Level Priority (if multiple levels touched on same bar)

H4 and S4 take priority over H3/S3. S4 and H4 take priority over session highs/lows.
Priority order: H4/S4 > H3/S3 > prev_day_high/prev_day_low > Asia high/low > London high/low > Pre-Market high/low

Only one signal fires per bar even if multiple levels are touched simultaneously.

### Parameters

```python
PROXIMITY_ATR = 0.5    # level must be within 0.5 × ATR(14) of the bar's high or low
ATR_PERIOD    = 14     # ATR lookback
MAX_PER_DAY   = 2      # shared cap — resets at 09:30 each day
```

### Touch Semantics

**Definition of a "level touch" rejection entry:**

The entry condition uses a **wick-tag interpretation**, not deep penetration:

- **Long condition:** `bar.low ≤ level + proximity` AND `bar.close > level`
  - The wick may extend below the level; this is fine and shows institutional testing.
  - However, if the bar closes **at or below** the level (no recovery), it is NOT a rejection—it is a breakdown. No signal fires.
  - Only a bar that penetrates the level's wick but recovers to close above the level qualifies.

- **Short condition:** `bar.high ≥ level - proximity` AND `bar.close < level`
  - The wick may extend above the level.
  - If close closes **at or above** the level, it is a breakout, not a rejection. No signal fires.
  - Recovery below the level is required.

**Rationale:** Institutions probe levels with wicks to detect liquidity, then reverse. The rejection candle proves they defend the level (close on the correct side). A close that penetrates far below the level (deep breakdown) invalidates the mean-reversion setup — that is directional movement, not rejection.

---

## Implementation — signal_generators.py

```python
def session_pivot_signal(
    df: pd.DataFrame,
    proximity_atr: float = 0.5,
    atr_period: int = 14,
    max_per_day: int = 2,
) -> pd.Series:
    """Generate session level pivot signals.

    Returns Series of +1 (long), -1 (short), 0 (no signal), indexed same as df.

    CRITICAL REQUIREMENTS:
    - df must have columns: open, high, low, close, volume
    - df must have columns: camarilla_h3, camarilla_h4, camarilla_s3, camarilla_s4
      (computed from PRIOR day OHLC — see camarilla_pivot_generator.py)
    - df must have columns: asia_high, asia_low, london_high, london_low,
      premarket_high, premarket_low, prev_day_high, prev_day_low, prev_day_close
    - df must have column: atr_14 (ATR computed from lookback-only bars, never forward)
    - df index must be tz-aware DatetimeIndex in America/New_York
    - df must be pre-filtered to 09:30–15:00 ET session bars only

    Logic:
    1. For each bar, check if bar.low penetrates any support level within proximity
    2. If yes, check rejection: bar.close must be ABOVE the touched level
    3. Check context: bar.close < prev_day_close (oversold context for longs)
    4. If all pass and daily count < max_per_day: signal = +1
    5. Symmetric for shorts: bar.high near resistance, close BELOW level, close > prev_day_close
    6. Max 2 signals per calendar day (resets at session open)
    """
    signal = pd.Series(0, index=df.index, dtype=int)
    daily_count: dict = {}  # date → count of signals fired

    # Level columns in priority order — highest priority first
    long_levels  = ["camarilla_s4", "camarilla_s3", "prev_day_low",
                     "asia_low", "london_low", "premarket_low"]
    short_levels = ["camarilla_h4", "camarilla_h3", "prev_day_high",
                     "asia_high", "london_high", "premarket_high"]

    for idx, row in df.iterrows():
        date_key = idx.date()
        count = daily_count.get(date_key, 0)
        if count >= max_per_day:
            continue

        atr = row["atr_14"]
        if pd.isna(atr) or atr <= 0:
            continue  # no ATR yet (warmup period), skip

        proximity = proximity_atr * atr
        fired = False

        # --- Long check ---
        for level_col in long_levels:
            level = row.get(level_col)
            if pd.isna(level):
                continue
            # Touch condition: bar.low <= level + proximity (came within range of level)
            if row["low"] <= level + proximity:
                # Rejection condition: close must be ABOVE the level
                if row["close"] > level:
                    # Context condition: close below prior day close (mean reversion context)
                    if row["close"] < row["prev_day_close"]:
                        signal[idx] = 1
                        daily_count[date_key] = count + 1
                        fired = True
                        break  # highest-priority level matched, stop checking

        if fired:
            continue

        # --- Short check ---
        for level_col in short_levels:
            level = row.get(level_col)
            if pd.isna(level):
                continue
            # Touch condition: bar.high >= level - proximity
            if row["high"] >= level - proximity:
                # Rejection condition: close must be BELOW the level
                if row["close"] < level:
                    # Context condition: close above prior day close
                    if row["close"] > row["prev_day_close"]:
                        signal[idx] = -1
                        daily_count[date_key] = count + 1
                        break

    return signal
```

---

## Feature Engineering (Input to ALL Models)

These features are produced by `camarilla_pivot_generator.py` and included in every parquet. All four models receive them as input features regardless of which strategies they cover.

### Camarilla Distance Features

For each level (H3, H4, S3, S4):

```python
# Distance in ATR units — positive means price is above the level
camarilla_h3_dist = (close - camarilla_h3) / atr_14
camarilla_h4_dist = (close - camarilla_h4) / atr_14
camarilla_s3_dist = (close - camarilla_s3) / atr_14
camarilla_s4_dist = (close - camarilla_s4) / atr_14
```

ATR normalization makes distances scale-invariant across volatility regimes. Raw point distances would make the model regime-dependent.

### Session High/Low Distance Features

For each session (Asia, London, Pre-Market):

```python
# Distance in ATR units
asia_high_dist     = (close - asia_high)     / atr_14
asia_low_dist      = (close - asia_low)      / atr_14
london_high_dist   = (close - london_high)   / atr_14
london_low_dist    = (close - london_low)    / atr_14
premarket_high_dist = (close - premarket_high) / atr_14
premarket_low_dist  = (close - premarket_low)  / atr_14
```

### Previous Day/Week Distances

```python
prev_day_high_dist  = (close - prev_day_high)  / atr_14
prev_day_low_dist   = (close - prev_day_low)   / atr_14
prev_week_high_dist = (close - prev_week_high) / atr_14
prev_week_low_dist  = (close - prev_week_low)  / atr_14
```

### NY AM Running High/Low

NY AM high and low are computed as running max/min from 09:30 to the current bar. **Use bars up to bar N-1 only — never include the current bar in the running max/min calculation.** A common error is including bar N, which makes the feature look-ahead.

```python
# Correct: rolling max up to but not including current bar
nyam_high = df["high"].expanding().max().shift(1)
nyam_low  = df["low"].expanding().min().shift(1)
```

---

## Logic Gaps and Guards

**1. Camarilla uses prior day data only.** H3/H4/S3/S4 must be computed from yesterday's confirmed OHLC. If today's OHLC is used (even after session close), it introduces lookahead when the model is evaluated bar-by-bar during training.

**2. Rejection candle requires close on correct side.** A bar that touches S4 but closes below S4 is NOT a rejection — it is a breakdown. The signal only fires if close > level (long) or close < level (short). Checking only for "bar touched the level" without verifying the close is a silent logic error.

**3. Proximity check uses ATR at signal bar.** The 0.5 × ATR proximity threshold uses the ATR at that specific bar, not a rolling average and not a fixed point value. ATR must be computed from lookback bars only.

**4. Daily cap is shared across all levels and the break companion.** H3, H4, S3, S4, Asia, London, Pre-Market highs/lows, and `session_pivot_break_signal` all share the same 2-signal-per-day counter. It is NOT 2 signals per level or 2 signals per pivot variant. One long at S4 and one short at H4 = cap reached for the day.

**5. NY AM running H/L excludes current bar.** Use `.shift(1)` after `.expanding().max()`. Including the current bar makes the model look ahead.

**6. Zero ATR guard.** During warmup (first 14 bars) ATR is NaN. Skip signal computation on these bars and set distance features to 0.0 rather than dividing by zero.

**7. Session boundary.** The signal generator only runs on bars within the 09:30–15:00 session filter. Camarilla levels are computed from prior-session data (which is outside this window). The generator must have access to prior-day OHLC separately from the session-filtered input DataFrame.

---

## Notes

- Session levels are particularly relevant for IFVG setups — IFVG requires a liquidity sweep, and pivot levels define where liquidity pools. Model 3 benefits from ORB IB and Session Pivots sharing structural level context.
- Camarilla H4/S4 are among the most watched intraday levels. Institutional players are aware of them and may use them as sweep targets. The CNN can learn this behavior from data.
- The proximity_atr parameter (default 0.5) can be tuned during HPO. Wider proximity catches more signals but reduces precision.

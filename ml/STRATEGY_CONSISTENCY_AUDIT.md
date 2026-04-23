# Strategy Consistency Audit — strategyLabbrain vs StrategyLab

**Auditor:** independent review
**Date:** 2026-04-18
**Scope:** 9 strategies across `strategyLabbrain/Strategies/*.md` vs `strategyLabbrain/ml/signal_generators.py`, `strategyLabbrain/Implementation/*_generator.py`, `StrategyLab/strategies/all_strategies.py`, and the docs `STRATEGYLAB_MASTER_SPEC.md` / `INITIAL_BALANCE_ORB_STRATEGY.md`.

**Reading note:** "Match?" uses ✅ (matches), ⚠️ (partial / soft deviation), ❌ (wrong or missing). "Silent" means the code runs without error but produces signals that do not conform to the written strategy.

---

## 1. IFVG (base)

| Item | Spec Says | Implementation Has | Match? |
|---|---|---|---|
| FVG detection (3-candle imbalance) | `C3.low > C1.high` (bull) / `C3.high < C1.low` (bear) | `highs[i-2] < lows[i]` / `lows[i-2] > highs[i]` in `detect_fvgs` | ✅ |
| Min FVG gap size | 5 pts for 1/2min, 7 pts for 3/5min | `5.0 if timeframe_minutes <= 2 else 7.0` | ✅ |
| Liquidity sweep (Condition 1) | Sweep a **structural** level: 1H/4H H/L, session H/L (Asia, London, NY AM/Lunch/PM), prev-day H/L, prev-week H/L | `_detect_sweep()` tests whether the last 3 bars made a new N-bar (default 10) rolling extreme on the **entry timeframe** | ❌ |
| HTF FVG confluence (Condition 2) | Price must be **inside or rejecting** from an HTF FVG on 15m/1h/4h (for 5m entries), with explicit HTF TF selection | `_check_htf_confluence()` only checks price is within ±`htf_proximity_pts` of an HTF zone; HTF loaded by `_resolve_ifvg_htf()` with preference 1h → 15min; **4h not generated**; HTF df loaded with `session_only=True`, which kills overnight / pre-market FVGs on the HTF | ⚠️ (loose + HTF session-only is wrong for HTF) |
| HTF min-gap threshold | Should scale with the HTF bar (a 15m/1h FVG is much larger) | Same 5 / 7 pt threshold as the entry TF — catches many noise gaps on HTF charts | ❌ |
| Entry | Close of the inversion candle | `signals.at[bar_time] = signal` on the close where `close > zone_top` (bull) / `close < zone_bottom` (bear) | ✅ |
| FVG invalidation before inversion | FVG becomes invalid if a prior candle closes through it in the opposite direction | Not checked — any still-`active` FVG is a candidate regardless of prior closes | ❌ (silent) |
| Stop loss | Swing low/high formed by the liquidity sweep preceding the IFVG | Signal generator returns only direction (+1/-1/0). No stop produced. No downstream code computes the sweep-swing stop either | ❌ |
| Break-even at 50% of target | Move stop to entry at 50% | Not emitted; not implemented downstream | ❌ |
| Target | 1R – 1.5R from entry | Not emitted | ❌ |
| Daily cap | 2/day, shared with Open Variant | `ifvg_combined()` runs base first, passes `_external_daily_counts` into open variant — both respect combined `max_signals_per_day=2` | ✅ |
| Session filter 09:30–15:00 ET | Yes | Enforced twice: `dataset_builder.load_data` applies `between_time(09:30, 15:00)`, and `ifvg_signals` reapplies inside each day loop | ✅ |
| StrategyLab version | n/a | **No IFVG implementation in `all_strategies.py`.** Strategy exists only in strategyLabbrain. | ❌ |

**Issues found:**
1. **Liquidity sweep gate is a proxy, not the spec gate** — `_detect_sweep` only asks "did the last 3 bars hit a new local N-bar extreme?". That fires on random intraday choppiness that has no relationship to 1H/4H/session/prev-day/prev-week levels. The model is learning from signals that were never filtered by the real condition. Critical.
2. **HTF confluence semantics are weaker than spec** — "near the zone" replaces "inside or rejecting". A rejection is a specific structural event (aggressive entry → reversal within 1–3 bars); a proximity check fires even on drift-throughs.
3. **HTF DataFrame is loaded session-only (09:30–15:00).** 15m and 1h FVGs that form overnight, in pre-market, or into the close are discarded — precisely the HTF zones a 9:30+ entry would be rejecting from. Wrong for HTF.
4. **HTF min-gap threshold is hard-coded to the entry-TF value** (5 or 7 pts). A 1h chart FVG of 7 pts is noise. Should scale with TF (e.g., 15pts on 15m, 30pts on 1h).
5. **FVG invalidation check missing.** An FVG that has already been closed through before the inversion bar can still produce a signal because the `active` flag is only toggled after firing, not after a contrary close.
6. **No stop / target / break-even emitted.** The spec's swing-low/sweep-swing stop is the whole reason the strategy has a defined R — without it, R-based labeling and walk-forward labeling are guessing.
7. For 1min entries, `_resolve_ifvg_htf` only tries `5min` — spec allows 5/15/1h/4h. Narrower than spec.

**Recommendation:** critical fixes needed before any trading or ML labeling is trustworthy. Minimum viable fix set: (a) replace `_detect_sweep` with a real level-sweep check against the already-computed session/prev-day/prev-week columns; (b) remove `session_only=True` from the HTF load; (c) scale `min_gap` by HTF bar size; (d) add invalidation check (deactivate FVG once any subsequent candle closes through it in the opposite direction); (e) emit stop = swing low/high of the sweep leg as signal metadata.

---

## 2. IFVG — Open Variant

| Item | Spec Says | Implementation Has | Match? |
|---|---|---|---|
| Timing gate | The **liquidity sweep** must occur between 09:30 and 09:35 ET; the IFVG close can be at/slightly after 09:35 | Uses 09:30–09:35 only to compute `open_high` / `open_low`; signals can fire **any time from 09:30–15:00** | ❌ |
| Sweep reference level | Prev-day H/L, prev-week H/L, overnight H/L, Asia H/L, pre-market H/L, structural 5m/15m level | Checks `open_low < first_open_close` (long) or `open_high > first_open_close` (short) — i.e., "did any price action occur in the 09:30–09:35 window?" No structural level referenced | ❌ |
| Sweep size | Any tick through the level qualifies | No level, so size is meaningless; essentially any non-flat first-5min window passes | ❌ (silent) |
| FVG detection / min gap | Same as base | Same as base | ✅ |
| HTF confluence | Same as base | Same (inherits base's weaknesses) | ⚠️ |
| Entry | Close of inversion candle | ✅ | ✅ |
| Stop | Swing low/high of the manipulation sweep | Not emitted | ❌ |
| Daily cap | 2/day, **shared with base IFVG** | `ifvg_combined` shares budget via `_external_daily_counts` | ✅ |
| Session filter 09:30–15:00 | Yes | Applied in `day_df.between_time('09:30', '15:00')` | ✅ |

**Issues found:**
1. **The "Open Variant" is not actually an open variant.** The time gate on the sweep is missing. The code accepts any IFVG that fires later in the session as long as the first-5min window wasn't flat. That's just a slightly-filtered base IFVG, not an opening manipulation play.
2. **No structural level reference in the sweep check.** Spec is emphatic: "prev day H/L, overnight H/L, pre-market H/L, structural level". None of these are in the logic.
3. **Sweep-direction "proxy" is trivially true on most days.** `open_low < first_open_close` is satisfied on almost any 5-min window with any downward tick after the first bar.
4. Stop not emitted (same as base).

**Recommendation:** critical. This variant as written is effectively duplicating base IFVG output, which is why `AGENT2_STATUS.md` shows `ifvg_open` with 893 signals — that's far too many for a 9:30–9:35 manipulation setup. Rewrite:
```
sweep_long  = open_window['low']  < max(prev_day_low,  overnight_low,  premarket_low)
sweep_short = open_window['high'] > min(prev_day_high, overnight_high, premarket_high)
```
and require the IFVG formation bar's timestamp to be at or after the sweep bar's timestamp, with the sweep timestamp ≤ 09:35.

---

## 3. ConnorsRSI2

| Item | Spec Says | strategyLabbrain (`connors_rsi2`) | StrategyLab (`ConnorsRSI2`) | Match? |
|---|---|---|---|---|
| RSI period | 2 | EWM α=1/2, min_periods=2 | Wilder smoothing on `rsi()` | ✅ / ✅ |
| Trend filter | Long if close > SMA(200); short if close < SMA(200) | `trend = close.rolling(200).mean()` → `price > trend_value` | `close[i] > trend[i]` | ✅ / ✅ |
| Entry trigger | `RSI(2) < 10` (long), `>90` (short) on close | Uses crossing `prev_rsi >= 10 and rsi < 10` (plus position state) | `r[i] < 10 and r[i-1] >= 10` | ✅ / ✅ (crossing is spec-equivalent to "first in sequence") |
| Re-entry guard | Require RSI to recover above exit threshold before re-entering | `position` state variable enforces it | Implicit via crossing; no explicit state | ✅ / ⚠️ (StrategyLab would re-fire if RSI bounces between 9 and 10 within a held trade; strategyLabbrain won't) |
| Exit on RSI recovery | `RSI(2) > 90` (long) | `rsi_value > rsi_exit` while position==1 | Not emitted (dict only has stop/target) | ✅ / ❌ |
| Exit on SMA(5) cross | `close > SMA(5)` (long) | `price > exit_value` while position==1 | Only used indirectly: `target = max(SMA5, entry + ATR)` | ✅ / ❌ |
| Stop loss | entry ± 1.5 × ATR(14) | Not emitted (Series only) | `entry ± 1.5 × ATR` | ❌ / ✅ |
| Target | entry + 1.0 × ATR(14) | Not emitted | `max(SMA5, entry + 1.0×ATR)` for long, `min(SMA5, entry − 1.0×ATR)` for short | ❌ / ⚠️ (adds SMA5 clamp that is not in spec) |
| Daily cap | Spec does not set one | None | None | ✅ / ✅ |
| Session filter 09:30–15:00 | Yes | Not applied internally; relies on `dataset_builder.load_data` which does enforce it | Not applied internally; **StrategyLab has no consistent session filter at load time** — depends on the caller | ✅ / ⚠️ |

**Issues found:**
1. **strategyLabbrain generator returns direction only** — stop, target, exit-on-RSI, exit-on-SMA(5), stop-hit are not emitted. Downstream labeling has to re-derive them, which may or may not match spec.
2. **StrategyLab's target is non-spec.** `max(SMA5, entry + 1.0×ATR)` widens the target on a long whenever `SMA5 > entry + ATR`. Spec says the target is purely `entry + 1.0 × ATR`. Same issue for short with `min(...)`.
3. **Neither version enforces the 9:30–15:00 filter internally.** strategyLabbrain is saved by `dataset_builder` pre-filtering; StrategyLab's backtest path has to pre-filter upstream.
4. StrategyLab does not implement the re-entry guard explicitly, but the strict crossing condition approximates it — fine.

**Recommendation:** moderate. Make `connors_rsi2` emit stop / target / exit-signal metadata so the ML labeler isn't synthesizing them. Remove the `max(SMA5, ...)` clamp from StrategyLab's `ConnorsRSI2.target` — it's not in the spec and it shifts R.

**Which is more correct?** strategyLabbrain for entry state tracking and exit conditions. StrategyLab for producing full trade dicts (but with the target deviation).

---

## 4. TTM Squeeze

| Item | Spec Says | strategyLabbrain (`Implementation/ttm_squeeze_generator.py`) | StrategyLab (`TTMSqueeze` in `all_strategies.py`) | Match? |
|---|---|---|---|---|
| Bollinger Bands | SMA(20) ± 2.0 × StdDev(20) | ✅ | ✅ | ✅ / ✅ |
| Keltner Channels | EMA(20) + 2.0 × ATR(20) | ✅ (`close.ewm(span=20)` + `atr.rolling(20)`) | ✅ | ✅ / ✅ |
| Squeeze on | BB upper < KC upper AND BB lower > KC lower | ✅ | ✅ | ✅ / ✅ |
| Min squeeze bars | 5 consecutive | ✅ via `squeeze_count_prev` | ✅ via backwards count | ✅ / ✅ |
| **Momentum formula** | **`linreg(delta, 12)` where `delta = close - (midpoint + SMA(close,12))/2`** — spec **explicitly** says "❌ `series - series.rolling(period).mean()`" is wrong | `linreg()` implemented correctly (value of regression at x=period-1); `midpoint = (high.rolling(12).max() + low.rolling(12).min())/2`; `delta = close - (midpoint + SMA)/2`; `momentum = linreg(delta, 12)` | **`mom = close - sma(close, self.mom_period)`** — exactly the mistake the spec calls out | ✅ / ❌ CRITICAL |
| Trigger | Squeeze ON → OFF **with** momentum confirming direction and direction increasing | `squeeze_on_prev and not squeeze_on` + `mom > 0 and momentum_increasing` | `not squeeze[i] and squeeze[i-1]` + `mom[i] > 0 and mom[i] > mom[i-1]` (but on the wrong `mom`) | ✅ / ⚠️ (right pattern, wrong variable) |
| Stop loss | entry ± 2.0 × ATR(14) | Not emitted | `entry ± stop_mult × ATR(14)`, default 2.0 | ❌ / ✅ |
| Target | entry + 2.0 × ATR(14) | Not emitted | `entry + target_mult × ATR`, default 2.0 | ❌ / ✅ |
| Daily cap | None | None | None | ✅ / ✅ |
| Session filter | 09:30–15:00 | Not applied internally (trusts caller) | Not applied | ⚠️ / ⚠️ |

**Issues found:**
1. **StrategyLab's TTM Squeeze uses the wrong momentum formula.** `close − SMA(close, 12)` is one of the four "❌ do not substitute" examples in the spec (TTMSqueeze.md lines 112–116). The strategyLabbrain `linreg` momentum is correct. **This is a critical signal-logic bug**: the direction and magnitude of momentum fire differently, so StrategyLab is firing different long/short signals than strategyLabbrain on the same data.
2. **strategyLabbrain returns direction only** — stop/target not emitted. (StrategyLab's version does emit correct stops/targets, so the signal dict side is fine there.)
3. Neither enforces 9:30–15:00 internally.

**Recommendation:** critical. Fix StrategyLab's `TTMSqueeze.generate_signals`: import the strategyLabbrain `linreg` implementation and replace the one-liner `mom = close - sma(close, ...)` with the full linreg-of-delta computation. Until this is fixed, StrategyLab's TTM Squeeze results are not comparable to the strategyLabbrain ML pipeline and are not faithful to the spec.

**Which is more correct?** strategyLabbrain, by a wide margin.

---

## 5. ORB IB (Initial Balance)

| Item | Spec Says | strategyLabbrain (`orb_initial_balance`) | StrategyLab (`InitialBalanceORB`) | Match? |
|---|---|---|---|---|
| IB window | 09:30–10:30 (60 min) | 60 minute window, checks first bar is 09:30 | `rth_start + 60min`, uses first bar of data as rth_start (assumes session-filtered input) | ✅ / ⚠️ |
| Signal window | **10:30–11:00 only** | `between_time("10:30", "11:00", inclusive="both")` | `day_df[day_df.index > ib_end]` — **no upper bound** | ✅ / ❌ |
| Entry trigger | Close above IB high (long) / below IB low (short) | ✅ on close | ✅ on close | ✅ / ✅ |
| Max signals/day | 1 | 1 | 1 | ✅ / ✅ |
| **Min IB range filter** | **IB range ≥ ATR(14)** — skip day if less | `if ib_range < first_atr: continue` | **Missing**. Nothing rejects narrow IB days | ✅ / ❌ |
| Stop | long = `max(IB low, entry − 1.5×ATR)`; short = `min(IB high, entry + 1.5×ATR)` | Not emitted | ✅ matches | ❌ / ✅ |
| Target | Both ORB IB specs define the target from the broken IB level: long = `IB High + extension_mult × IB Range`, short = `IB Low − extension_mult × IB Range`. | Not emitted | `ib_high + 1.5 × ib_range` / `ib_low − 1.5 × ib_range` | ❌ / ✅ (strategyLabbrain emits direction only; StrategyLab target matches the spec) |
| Session filter 09:30–15:00 | Yes | Data pre-filtered by `dataset_builder`; candidate window is 10:30–11:00 | No internal filter; depends on caller | ✅ / ⚠️ |

**Issues found:**
1. **Spec target-origin contradiction is resolved.** `Strategies/ORB IB - Initial Balance.md` now follows `docs/INITIAL_BALANCE_ORB_STRATEGY.md` and the StrategyLab implementation: targets are measured from the broken IB level.
2. **StrategyLab skips the min-IB-range quality filter entirely.** A day with a flat first hour (IB range << ATR) still generates a signal. The spec's self-filtering property ("on neutral range days no trade triggers") is gone.
3. **StrategyLab has no 11:00 upper bound on the signal window.** A breakout at 14:45 still gets tagged as IB_ORB. Spec is clear: signal window 10:30–11:00.
4. strategyLabbrain emits no stop/target; StrategyLab emits both — so for labeling purposes StrategyLab is more complete, but the narrow-range day leakage biases the edge.

**Recommendation:** moderate. Keep the ORB IB target origin pinned to the implementation-backed `INITIAL_BALANCE_ORB_STRATEGY.md`. Add `ib_range >= av[or_end_idx]` guard and `idx.time() <= time(11,0)` guard to StrategyLab's `InitialBalanceORB`.

**Which is more correct?** strategyLabbrain for the filters, StrategyLab for emitting stop/target. Neither is complete.

---

## 6. ORB Volatility Filtered

| Item | Spec Says | strategyLabbrain (`orb_volatility_filtered`) | StrategyLab (`VolatilityFilteredORB`) | Match? |
|---|---|---|---|---|
| OR window | 09:30–09:40 (10 min) | ✅ | ✅ | ✅ / ✅ |
| ATR(14) + 100-bar rolling percentile | Yes; require ≥100 bars before percentiles are valid | `atr.rolling(100, min_periods=100).quantile(...)` | `atr_window = av[start_i:or_end_idx+1]` with guard `len(atr_window) < 20` — **accepts as few as 20 valid ATR bars** | ✅ / ❌ (warmup too permissive) |
| **Percentile gate bar** | **At entry time** (per breakout candle) | Checked per candidate bar inside 09:40–11:00 loop | Checked **once at OR end**; if pass, all subsequent breakouts in the day qualify regardless of whether ATR moved into/out of the band later | ✅ / ❌ |
| 25 ≤ pct ≤ 85 | Yes (tunable) | ✅ | ✅ but uses `np.searchsorted` approximation over a short window | ✅ / ⚠️ |
| Signal window | 09:40–11:00 | `between_time("09:40", "11:00")` | No upper bound | ✅ / ❌ |
| 1 signal/day | Yes | ✅ | ✅ | ✅ / ✅ |
| Stop | `max(OR low, entry − 1.5×ATR)` | Not emitted | ✅ | ❌ / ✅ |
| Target | `entry + 1.0 × OR range` | Not emitted | `c + target_mult × or_range` ✅ | ❌ / ✅ |
| Session filter 09:30–15:00 | Yes | Via `between_time(09:40, 11:00)` | Not applied upstream in-function | ✅ / ⚠️ |

**Issues found:**
1. **StrategyLab lowers the percentile warmup from 100 to 20 bars.** That means the percentile filter is essentially noise for the first few months of any backtest. Spec is explicit: "Skip signals until `len(atr_history) >= atr_lookback`."
2. **StrategyLab gates at OR-end, not at breakout bar.** ATR can change meaningfully between 09:40 and 10:45. A setup that would fail the regime filter at the actual breakout still gets through.
3. No 11:00 upper bound on signal window in StrategyLab.

**Recommendation:** moderate-to-critical. The warmup loosening is a real issue because it silently changes the strategy's behavior during the very training period the optimizer cares about most (recent history). Align to spec: require ≥100 ATR values, gate at breakout bar.

**Which is more correct?** strategyLabbrain, clearly.

---

## 7. ORB Wick Rejection

| Item | Spec Says | strategyLabbrain (`orb_wick_rejection`) | StrategyLab (`WickRejectionORB`) | Match? |
|---|---|---|---|---|
| OR window | 09:30–09:40 | ✅ | ✅ | ✅ / ✅ |
| body_pct = \|close−open\| / (high−low) ≥ 0.55 | Yes | ✅ | ✅ | ✅ / ✅ |
| Zero-range guard | Skip if high == low | `if bar_range == 0: continue` | `if bar_range <= 0: continue` | ✅ / ✅ |
| Directional body (bull close for long) | **Optional** per spec ("directional body confirmation is optional") | Not enforced | **Enforced**: `c > o` for long, `c < o` for short | ✅ / ⚠️ (stricter than spec default) |
| Signal window | 09:40–11:00 | ✅ | No upper bound | ✅ / ❌ |
| 1 signal/day | Yes | ✅ | ✅ | ✅ / ✅ |
| Stop | `max(OR low, entry − 1.5×ATR)` | Not emitted | ✅ | ❌ / ✅ |
| Target | `entry + 1.0 × OR range` | Not emitted | ✅ | ❌ / ✅ |
| Session filter | Yes | Via candidate window | Not in-function | ✅ / ⚠️ |

**Issues found:**
1. **StrategyLab adds the directional-body constraint by default.** Spec explicitly says this is optional. That means StrategyLab will have fewer signals than strategyLabbrain on the same data and the optimizer's walk-forward metrics are on a stricter variant.
2. No 11:00 upper bound in StrategyLab.

**Recommendation:** minor-to-moderate. Expose `require_directional_body` as a parameter (default False to match spec); add 11:00 cap.

**Which is more correct?** strategyLabbrain matches the spec default more faithfully.

---

## 8. ORB Volume Adaptive

| Item | Spec Says | strategyLabbrain (`orb_volume_adaptive`) | StrategyLab (`VolumeAdaptiveORB`) | Match? |
|---|---|---|---|---|
| OR window | **Fixed 10 minutes** (09:30–09:40) | ✅ | **Dynamic**: OR ends when bar volume drops below `vol_decay × running avg` (or max 24 bars) | ✅ / ❌ different strategy entirely |
| OR avg volume | Mean volume of the OR bars | ✅ | Uses running average, not the same thing | ✅ / ❌ |
| Volume filter on breakout bar | `breakout_volume ≥ or_avg_volume × 1.5` | ✅ | **Missing** — no volume check on the breakout bar itself | ✅ / ❌ |
| OR avg volume zero guard | Skip day if avg vol = 0 | `if or_avg_volume == 0: continue` | n/a (different mechanism) | ✅ / ❌ |
| 1 signal/day | Yes | ✅ | ✅ | ✅ / ✅ |
| Min range filter | Not in spec | Not enforced | `min_range_atr=0.3` extra filter — not in spec | ✅ / ❌ (spec-divergent addition) |
| Stop / target | Standard ORB | Not emitted | Emits correctly | ❌ / ✅ |

**Issues found:**
1. **These are two different strategies with the same name.** The spec (`Strategies/ORB Volume Adaptive.md`) describes a fixed 10-min OR with a volume filter on the breakout candle. StrategyLab's `VolumeAdaptiveORB` implements a dynamic OR that ends on volume decay — a totally different concept. If both are run in a shared study, their results cannot be compared.
2. **StrategyLab adds `min_range_atr=0.3` which is not in the spec.**
3. strategyLabbrain matches the spec.

**Recommendation:** critical naming/conceptual collision. Either:
- rename the StrategyLab class `VolumeDecayORB` and write a new spec for it, or
- replace its body with the strategyLabbrain-style fixed-OR + breakout-volume-filter logic.

Leaving both under the name "ORB Volume Adaptive" guarantees confusion in reports and ML feature engineering.

**Which is more correct?** strategyLabbrain follows the written spec. StrategyLab is a different strategy altogether.

---

## 9. Session Level Pivots

| Item | Spec Says | Implementation (`session_pivot_signal` + `camarilla_pivot_generator.py`) | Match? |
|---|---|---|---|
| Camarilla formulas | H3/H4 = prev_close + range × (0.275 / 0.55); S3/S4 = prev_close − range × (0.275 / 0.55); computed from **prior day** OHLC only | ✅ `compute_camarilla` aggregates by `df.index.date`, shifts by 1, maps back | ✅ |
| Levels constant within day | Yes | ✅ (mapping dict) | ✅ |
| **Asia session window** | Prior day 20:00 → current day 02:00 (note on page: "session highs/lows of Asia, London…") | **`day_df.between_time('20:00', '02:00')` on a single calendar day.** Returns *current day* 00:00–02:00 + *current day* 20:00–23:59 — the evening bars belong to the *next* trading day's Asia session. For Tuesday's NY session the code pulls Tuesday evening rather than Monday evening. | ❌ |
| London window | 02:00–07:00 of the current trading day | ✅ (correct interpretation) | ✅ |
| Pre-market | 07:00–09:30 | ✅ | ✅ |
| NY AM running H/L excludes current bar | Yes (use `.shift(1)` or equivalent) | ✅ `prior_bars = ny_am_bars.iloc[:i]` — excludes current | ✅ |
| Long condition 1 (level touch) | `bar.low` within `proximity_atr × ATR(14)` of S4/S3/session low/prev-day low | `low.le(level.add(proximity))` — any low ≤ level + proximity, i.e., any low at or below `level + 0.5×ATR`. A deep sell-off far below the level still satisfies this. | ⚠️ (too permissive for interpretation of "reaches within", but defensible) |
| Long condition 2 (rejection close) | `close > level` | ✅ `close.gt(level)` | ✅ |
| Long condition 3 (context) | `close < prev_day_close` | ✅ `close.lt(prev_day_close)` | ✅ |
| prev_day_close source | Must be available | Not output by `compute_pivot_features` directly; `_prev_day_close_series` falls back to `(H3+S3)/2`. Algebraically `(H3+S3)/2 ≡ prev_close` so this is exact — but relies on Camarilla levels being present | ✅ (works, but fragile; if Camarilla not joined, signal silently returns 0) |
| Priority order | H4/S4 > H3/S3 > prev-day > Asia > London > Pre-market | ✅ implemented in priority list order | ✅ |
| Max 2/day shared across all levels | Yes | ✅ via `daily_signal_number = active.cumsum().groupby(date)` and `le(max_per_day)` | ✅ |
| Session filter 09:30–15:00 | Required (per spec: "df must be pre-filtered") | Pre-filter applied in `dataset_builder.load_data`; not re-applied in signal generator | ✅ (via caller) |
| Raw level columns in feature output | Needed for signal generator: `camarilla_s4/s3/h3/h4`, `asia_high/low`, etc. | `compute_pivot_features` outputs **only distances** (`h4_dist`, `asia_high_dist`, …) + `h*_above`. Raw levels reconstructed by `_level_series` using `close − dist × atr` — works algebraically | ⚠️ (fragile but correct when `atr_14` is present) |
| `prev_day_close` in feature df | Needed | `compute_prev_day_week` computes it but `compute_pivot_features` does not include it in the returned frame; relies on Camarilla fallback | ⚠️ |
| `atr_14` column | Required by signal generator | Not emitted by `compute_pivot_features`; `dataset_builder` injects it separately (`session_pivot_input["atr_14"] = atr_series…`) | ✅ via glue |

**Issues found:**
1. **Asia session window is wrong.** `day_df.between_time('20:00', '02:00')` on a per-calendar-day slice gets **current day 00:00–02:00 + current day 20:00–23:59**. The evening bars belong to the NEXT trading day's Asia session. For a Tuesday NY session the code returns Tuesday evening (future data) instead of Monday evening (which is what Asia preceding Tuesday's open means). This is **both wrong and a lookahead leak** for the evening portion. The 00:00–02:00 portion is correct.
2. **`prev_day_close` is not in `compute_pivot_features` output.** The `(H3+S3)/2` fallback is algebraically exact but silently returns NaN if Camarilla levels aren't joined into the signal's input df. Easy to miss.
3. **Raw level columns are absent from the feature output.** The signal generator reconstructs via `close − dist × atr`, which works when `atr_14` is present. Fragile: add a test that asserts reconstruction == stored level if you ever store both.
4. **Touch proximity semantics** — `low ≤ level + proximity` allows any low below the level to qualify. Arguably fine (deep wick = stronger rejection if close recovers), but spec phrase "reaches within" is ambiguous. Document the chosen interpretation.
5. **`session_pivot_break_signal` spec is resolved.** The continuation H4/S4 break signal is now documented in `Strategies/Session Level Pivots Break.md`.

**Recommendation:** critical for #1, moderate for the rest. The Asia fix is a one-liner change to the session compute — pull Asia from the prior-calendar-day evening through the current-day 02:00 slice, not from current-day between_time. Example:

```python
prior_day = day - pd.Timedelta(days=1)
prior_day_df = df[df.index.normalize() == prior_day]
asia_bars = pd.concat([
    prior_day_df.between_time('20:00', '23:59'),
    day_df.between_time('00:00', '02:00'),
])
```

Also strip the `current-day 20:00–23:59` contamination from the Asia level output. Currently the Asia high/low for Tuesday includes Tuesday's evening (which at the moment of, say, Tuesday 10:00 is in the future) — this is a lookahead feature that will inflate Asia-based distance features during training on recent days.

**Which is more correct?** Only one implementation of this strategy exists (strategyLabbrain). StrategyLab has none.

---

# Cross-cutting observations

1. **strategyLabbrain signal generators return `pd.Series` of direction only.** No stops, no targets, no exit triggers. That pushes responsibility onto the ML labeler / backtester to derive them — and for IFVG (swing-of-sweep stop) and Session Pivots (level-based stops), there is no deterministic way to recover them after the fact. Spec stops become educated guesses.
2. **StrategyLab signal classes return full trade dicts with stop/target.** More complete, but in three places (TTM momentum, Volume Adaptive concept, ConnorsRSI2 target, ORB filters) they diverge from the written spec.
3. **Session filter is enforced at `dataset_builder.load_data` in strategyLabbrain**, so every generator that runs through the ML pipeline gets 09:30–15:00 data. For StrategyLab, the session filter is not consistently enforced in the strategy code itself — callers have to pre-filter.
4. **No IFVG in StrategyLab at all.** If you want an apples-to-apples cross-check between the two stacks on IFVG, it doesn't exist.

---

# Prioritized fix list

## Critical — wrong signal logic or lookahead

1. **StrategyLab `TTMSqueeze`: wrong momentum formula.** `mom = close − sma(close, 12)` is the exact mistake the spec flags. Replace with `linreg(close − (midpoint + SMA)/2, 12)` using the strategyLabbrain `linreg` implementation.
2. **IFVG base: `_detect_sweep` is a rolling-window proxy, not a structural-level sweep.** Rewrite to check sweeps of prev-day H/L, prev-week H/L, and (if available in the df) 1H / 4H / session H/L columns.
3. **IFVG Open Variant: time gate on the sweep is missing; "sweep" is a non-check.** Rewrite per spec: sweep of a structural level within 09:30–09:35, IFVG close at or after the sweep, before roughly 09:40.
4. **Session Level Pivots: Asia window is wrong and contains lookahead.** Rebuild Asia from prior-day 20:00–23:59 + current-day 00:00–02:00. Strip current-day evening from the Asia level.
5. **IFVG HTF df is loaded `session_only=True`.** Drops overnight/pre-market HTF FVGs. Remove `session_only` for the HTF load.
6. **IFVG: no invalidation check before inversion.** Add: when scanning candidates, if any bar between `formed_at` and current bar closed through the FVG in the opposite direction, skip that FVG.
7. **StrategyLab `VolumeAdaptiveORB` is a different strategy than the spec says.** Either rename (e.g., `VolumeDecayORB`) or rewrite to match the fixed-OR + breakout-volume-filter spec.

## Moderate — missing filter, spec deviation, missing output

8. **StrategyLab `InitialBalanceORB`: missing `ib_range ≥ ATR(14)` filter.** Add the guard.
9. **StrategyLab `InitialBalanceORB`: missing 11:00 upper bound on signal window.** Add `if idx.time() > dt.time(11,0): break`.
10. **StrategyLab `VolatilityFilteredORB`: percentile warmup too permissive (20 vs 100) and gated at OR-end not per-bar.** Align to spec.
11. **StrategyLab `ConnorsRSI2`: `target = max(SMA5, entry + 1.0×ATR)` deviates from spec.** Remove the clamp; target should be `entry + target_atr_mult × ATR`.
12. **IFVG HTF `min_gap` is entry-TF value (5/7 pts) on HTF bars.** Scale by HTF bar minutes (e.g., 2× for 15min, 4× for 1h).
13. **IFVG, Session Pivots, ConnorsRSI2, TTMSqueeze: strategyLabbrain returns direction only.** Emit stop / target / sweep-bar metadata so labeling is spec-faithful.
14. **RESOLVED: ORB IB target origin.** Both ORB IB docs now use the broken-level extension target.
15. **StrategyLab `VolumeAdaptiveORB` has non-spec `min_range_atr=0.3`.** Remove or move to spec.
16. **StrategyLab `WickRejectionORB` requires directional body** by default. Spec says optional. Gate by parameter.
17. **All StrategyLab ORB variants: missing 11:00 upper signal-window bound.** Add consistently.

## Minor — cosmetic / fragility

18. **`compute_pivot_features` does not expose raw level columns or `prev_day_close`.** The signal generator reconstructs them algebraically, which works but is brittle. Add a pass-through option.
19. **`_resolve_ifvg_htf` only tries `5min` for 1min entries.** Spec allows 5/15/1h/4h. Broaden the fallback list.
20. **RESOLVED: `session_pivot_break_signal` spec.** Documented in `Strategies/Session Level Pivots Break.md`.
21. **ConnorsRSI2 session filter** is not in-function in either codebase; rely on caller pre-filtering. Add an explicit filter for safety.
22. **RESOLVED: Session Pivots touch semantics.** The spec documents the chosen wick-tag interpretation.

---

*End of audit.*

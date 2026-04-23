# Fix Prompts — Grouped by File/Subsystem

Source of truth: `ml/STRATEGY_CONSISTENCY_AUDIT.md` (22 issues, numbered).

## Why this grouping (not another)

Three reasonable ways to group these bugs:

1. **By severity (critical → moderate → minor).** Sounds good, feels worst-first. Wrong in practice — you'd edit the same file 2–3 times, re-run tests 2–3 times, and risk silent regressions each time.
2. **By strategy (IFVG, ORB, TTM, etc.).** Cleaner, but splits bugs that live in the same module (e.g., IFVG generator vs. dataset builder).
3. **By file/subsystem.** Each group = one self-contained PR. One round of edits, one test run, one review. This is the right answer.

I'm giving you **6 prompts**, ordered so each one leaves the repo in a working state. Run them in order. Don't skip ahead — Group 2 assumes Group 1's outputs exist.

Order of execution:

1. **Group 1 — IFVG correctness block** (critical signal logic)
2. **Group 2 — Session Level Pivots correctness block** (critical lookahead)
3. **Group 3 — TTM Squeeze momentum fix** (critical, tiny file)
4. **Group 4 — StrategyLab ORB variants cleanup** (critical + moderate, single file)
5. **Group 5 — ConnorsRSI2 alignment** (moderate + minor, cross-repo)
6. **Group 6 — Specs & docs cleanup** (kill the contradictions, add missing spec text)

---

## Group 1 — IFVG correctness block

**Issues addressed:** Audit #2, #3, #5, #6, #12, #13 (IFVG portion), #19
**Files touched:** `Implementation/ifvg_generator.py`, `ml/dataset_builder.py`

**Prompt to paste:**

> You are fixing IFVG generation end-to-end. Do all of the following in one pass, then stop and report.
>
> 1. In `Implementation/ifvg_generator.py`, rewrite `_detect_sweep` so it checks sweeps of real structural levels: prev-day high/low, prev-week high/low, and — if present as columns on the input df — 1H high/low, 4H high/low, and current-day session high/low. Remove the rolling-N-bar proxy. If a required structural column is missing, log a warning once and skip that level (don't silently pass).
> 2. In the same file, add an invalidation check before inversion: when scanning a candidate FVG formed at `t_form`, if any bar in `(t_form, t_current]` closed through the FVG in the opposite direction of the eventual inversion, skip that FVG.
> 3. In `ifvg_open_signals`, enforce a time gate: sweep must occur in 09:30–09:35 ET, IFVG close at or after the sweep bar, entry before 09:40 ET. No entry outside that window.
> 4. In `_check_htf_confluence`, scale `min_gap` by the HTF bar size relative to the entry TF: 15min → 2×, 1h → ~4×, 4h → larger. Expose the multiplier table at module top.
> 5. In `ml/dataset_builder.py`'s `_resolve_ifvg_htf`, remove `session_only=True` on the HTF load. Overnight/pre-market HTF FVGs must survive. Also broaden the HTF fallback list for 1min entries to `['5min','15min','1h','4h']`.
> 6. Make `ifvg_signals` and `ifvg_open_signals` return a DataFrame (or dict of series) with: `direction`, `stop_px`, `target_px`, `sweep_bar_ts`, `htf_confluence_tf`. Keep backward compatibility by also exposing the current direction-only output.
>
> After the edits, print a diff summary of each function changed and list which structural columns are now required on the input df.

**Acceptance criteria:**

- `_detect_sweep` references at least `prev_day_high/low` and `prev_week_high/low` by name.
- `ifvg_open_signals` has an explicit 09:30–09:40 ET gate.
- `_check_htf_confluence` has a `HTF_MIN_GAP_MULT` table.
- `_resolve_ifvg_htf` no longer passes `session_only=True` for HTF.
- Return schema includes stop/target/sweep metadata.

---

## Group 2 — Session Level Pivots correctness block

**Issues addressed:** Audit #4, #13 (pivots portion), #18, #22
**Files touched:** `Implementation/camarilla_pivot_generator.py`, `ml/signal_generators.py`

**Prompt to paste:**

> Fix Session Level Pivots. All of the following in one pass.
>
> 1. In `Implementation/camarilla_pivot_generator.py`, rebuild the Asia session computation. Current `SESSION_TIMES['asia'] = ('20:00', '02:00')` is applied to a per-day slice, which either returns empty or includes the current day's evening bars — that's a lookahead bug. Replace with: for each trading day D, Asia = prior-day 20:00–23:59 UNION current-day 00:00–02:00 in America/New_York tz. Make sure London, NY AM, NY PM continue to be computed on the current day only.
> 2. In `compute_pivot_features`, add an option `expose_raw=True` (default False for back-compat) that attaches the raw level columns (`camarilla_h3/h4/s3/s4`, `session_high/low` per session, `prev_day_close`) to the output. The current signal generator reconstructs these algebraically — brittle.
> 3. In `ml/signal_generators.py`, update `session_pivot_signal` to: (a) consume the raw levels from `expose_raw=True` instead of reconstructing, (b) emit `stop_px`, `target_px`, and `level_hit` (which level triggered) alongside direction.
> 4. Add a module-level docstring to `session_pivot_break_signal` either documenting its spec or marking it experimental — do not leave it silent.
> 5. In `Strategies/Session Level Pivots.md`, add one paragraph clarifying the touch semantics: is `low ≤ level + proximity` a wick-tag only, or does deep penetration still count? Pick one, document it, and make the code match.
>
> After the edits, print a diff summary and a before/after example of the Asia window for a sample trading day.

**Acceptance criteria:**

- Asia window no longer uses the `20:00–02:00` string pattern on a single-day slice.
- `compute_pivot_features` has `expose_raw` option.
- `session_pivot_signal` returns stop/target/level metadata.
- `session_pivot_break_signal` has a docstring.
- Session Level Pivots spec has an explicit touch-semantics paragraph.

---

## Group 3 — TTM Squeeze momentum fix

**Issues addressed:** Audit #1, #13 (TTM portion)
**Files touched:** `StrategyLab/strategies/all_strategies.py`, optionally `Implementation/ttm_squeeze_generator.py`

**Prompt to paste:**

> Fix TTMSqueeze in StrategyLab.
>
> 1. Open `StrategyLab/strategies/all_strategies.py`, class `TTMSqueeze`. The current `mom = close - sma(close, self.mom_period)` is exactly the formula the spec warns against. Replace with the linreg-based momentum the spec requires: `mom = linreg(close - (midpoint(high,low,period) + sma(close,period)) / 2, period)` where `midpoint = (highest_high + lowest_low) / 2`. Import or re-implement `linreg` consistently with `Implementation/ttm_squeeze_generator.py`.
> 2. If `Implementation/ttm_squeeze_generator.ttm_squeeze` already returns a direction-only series, extend it to optionally return stop/target metadata so labeling can be faithful to the spec.
> 3. Add a unit test or inline assertion that on a synthetic sine wave, the new momentum flips sign in phase with the linreg-of-delta — not in phase with `close - sma`.
>
> Report the diff and the unit test output.

**Acceptance criteria:**

- `StrategyLab TTMSqueeze.mom` computed via `linreg` of delta, not `close − sma(close)`.
- `linreg` helper matches the strategyLabbrain implementation.
- Synthetic test confirms correct sign.

---

## Group 4 — StrategyLab ORB variants cleanup

**Issues addressed:** Audit #7, #8, #9, #10, #15, #16, #17
**Files touched:** `StrategyLab/strategies/all_strategies.py`

**Prompt to paste:**

> Clean up the ORB family in `StrategyLab/strategies/all_strategies.py`. All in one pass.
>
> 1. `VolumeAdaptiveORB` currently implements a volume-decay dynamic opening range — that is a different strategy than the spec in `strategyLabbrain/Strategies/ORB Volume Adaptive.md`, which calls for a fixed 30-min OR + a breakout-volume-vs-OR-avg filter. Either (a) rewrite to match the spec, or (b) rename the current implementation to `VolumeDecayORB`, keep it as a separate registered strategy, and add a new `VolumeAdaptiveORB` that matches the spec. Pick (a) unless you confirm (b) is what the user actually wants by leaving a TODO at the top of the file.
> 2. `VolumeAdaptiveORB` has a non-spec `min_range_atr=0.3` gate. Remove it, or move it into the spec and cite the source in a comment.
> 3. `InitialBalanceORB`: add the `ib_range ≥ ATR(14)` filter gating entries. Add an 11:00 ET upper bound on signal window (`if idx.time() > time(11,0): break`).
> 4. `VolatilityFilteredORB`: change the percentile warmup from 20 to 100 bars, and gate per-bar (not only at OR-end), consistent with the spec.
> 5. `WickRejectionORB`: the directional-body requirement should be opt-in, not default. Add a `require_directional_body` param defaulting to False, and gate the current behavior behind it.
> 6. Add a consistent 11:00 ET upper signal-window bound to every ORB variant that doesn't already have one.
>
> After edits, print a table listing each ORB class, the filters it now enforces, and the signal-window bounds.

**Acceptance criteria:**

- `VolumeAdaptiveORB` matches spec OR is renamed and the spec version added.
- `min_range_atr` removed or documented.
- `InitialBalanceORB` has range filter + 11:00 cap.
- `VolatilityFilteredORB` warmup ≥ 100, per-bar gate.
- `WickRejectionORB` directional body is optional.
- Every ORB variant enforces 11:00 ET cap.

---

## Group 5 — ConnorsRSI2 alignment

**Issues addressed:** Audit #11, #13 (CRSI2 portion), #21
**Files touched:** `StrategyLab/strategies/all_strategies.py`, `ml/signal_generators.py`

**Prompt to paste:**

> Align both ConnorsRSI2 implementations to the spec.
>
> 1. In `StrategyLab/strategies/all_strategies.py`, class `ConnorsRSI2`: the current `target = max(SMA5, entry + 1.0 × ATR)` deviates from the spec. Remove the clamp. Target = `entry + target_atr_mult × ATR` for longs, `entry − target_atr_mult × ATR` for shorts, where `target_atr_mult` is a parameter defaulting to the value in `strategyLabbrain/Strategies/ConnorsRSI2.md`.
> 2. In `ml/signal_generators.py`, `connors_rsi2`: extend the return to include `stop_px` and `target_px` alongside direction. Keep the direction-only return as a back-compat fallback.
> 3. Add an explicit in-function session filter `09:30–15:00 ET` to both implementations so correctness does not depend on the caller pre-filtering. Make it overridable by an argument.
>
> Print the diff for both files.

**Acceptance criteria:**

- StrategyLab CRSI2 target has no `max(...)` clamp.
- strategyLabbrain CRSI2 returns stop/target metadata.
- Both implementations filter to 09:30–15:00 ET internally.

---

## Group 6 — Specs & docs cleanup

**Issues addressed:** Audit #14, #20 (spec portion), #22 (spec portion)
**Files touched:** `strategyLabbrain/Strategies/*.md`, `StrategyLab/docs/INITIAL_BALANCE_ORB_STRATEGY.md`

**Prompt to paste:**

> Kill the spec contradictions.
>
> 1. `strategyLabbrain/Strategies/ORB IB - Initial Balance.md` and `StrategyLab/docs/INITIAL_BALANCE_ORB_STRATEGY.md` disagree on ORB IB target origin. Pick the research-backed origin and delete the other phrasing from the losing doc. Leave a one-line note in the losing doc pointing to the winning doc so future-you doesn't re-introduce the contradiction.
> 2. For `session_pivot_break_signal`: if Group 2 did not already add a spec, add `Strategies/Session Level Pivots Break.md` describing the signal (or mark it experimental in code and in an EXPERIMENTAL.md index file).
> 3. Touch semantics paragraph for Session Level Pivots — if not already added in Group 2, add it here.
>
> Print the resulting list of spec files and a 1-line summary of each.

**Acceptance criteria:**

- No active ORB IB spec contains conflicting entry-origin and IB-level target origins.
- `session_pivot_break_signal` either has a spec or is explicitly marked experimental.

---

## After all six groups

Run these to catch regressions:

- Re-run the audit against the modified code. Diff against the original audit. Every "Critical" row should be resolved.
- Run a smoke backtest on one month of MNQ data per strategy to confirm: no empty signal series (would indicate the new filters killed everything), no pre-09:30 or post-15:00 signals, no signals outside the strategy-specific window (e.g., no ORB signals after 11:00).
- Diff the signal count per strategy against a pre-fix baseline. A drop of more than 50% on any strategy means a filter is probably too strict — investigate before accepting.

That last step is the one everyone skips. Don't skip it.

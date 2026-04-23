# NinjaScript Execution Rules

**Purpose:** Defines exactly how the live NinjaScript strategy resolves conflicts between the 4 CNN models and 9 underlying strategies at execution time. These rules must be implemented precisely — ambiguity is a logic fault.

---

## 1. Contract Limit

- **Max open contracts across ALL models combined: 50 MNQ**
- This is a hard ceiling. Never exceed it regardless of confidence.
- TopStep 50K allows 5 NQ mini = 50 MNQ micro. Point value = $2/point.
- `total_open_contracts` is tracked as a running integer across the entire strategy instance.

---

## 2. Position Sizing

Contracts for any new trade are computed by:

```
function ComputeContracts(stop_pts, confidence):
    target_risk = 500.0                     # dollars
    base = floor(target_risk / (stop_pts * 2.0))
    base = max(base, 1)

    if confidence >= 0.80:   scale = 1.00
    elif confidence >= 0.70: scale = 0.80
    elif confidence >= 0.65: scale = 0.60
    else:                    scale = 0.40

    sized = max(1, floor(base * scale))
    contracts = min(sized, 50)              # hard ceiling
    
    # Additional ceiling: do not push total open over 50
    contracts = min(contracts, 50 - total_open_contracts)
    return contracts
```

**Example table (stop = 10 pts, $2/pt):**

| Confidence | Base | Scale | Contracts |
|---|---|---|---|
| ≥ 0.80 | 25 | 1.00 | 25 |
| ≥ 0.70 | 25 | 0.80 | 20 |
| ≥ 0.65 | 25 | 0.60 | 15 |
| < 0.65 | 25 | 0.40 | 10 |

If `contracts` would push total above 50, reduce to the available headroom. If headroom = 0, skip the trade entirely.

---

## 3. Conflict Resolution — Two Models Signal Same Bar

### 3A. Same direction (both Long or both Short)

- Sum the contracts each would take independently.
- Cap the total at whichever is smaller: `50 - total_open_contracts` or the combined size.
- Enter once with the capped contract count.
- Use the higher of the two stop distances (more conservative ATR stop) for risk calculations.
- Use the lower of the two targets (take profit at the first model's target).

**Rationale:** Two independent models agreeing is additive confidence but still a single position — we do not open two separate entries.

### 3B. Opposite direction (one Long, one Short)

Compute each model's confidence independently. Then:

```
delta = abs(long_confidence - short_confidence)

if delta >= 0.05:
    # Clear winner — execute the higher-confidence direction only
    execute(higher_confidence_model)
else:
    # Too close to call — skip both
    skip all signals this bar
```

**No exception to this rule.** If delta < 0.05, no trade fires even if both confidences are above the entry threshold.

---

## 4. Conflict Resolution — Existing Position + New Opposite Signal

**Scenario:** A model currently holds an open trade. A different model fires a signal in the opposite direction on the same or a later bar.

**Rule:**

```
if existing_trade_is_open:
    if new_signal.direction == opposite:
        # Only override if existing trade is deep in drawdown AND new confidence is strong
        if current_pnl_pts < -0.50 * atr_14 AND new_signal.confidence > 0.70:
            close existing position
            open new position (new direction, computed contracts)
        else:
            skip new signal — let existing trade run
    elif new_signal.direction == same:
        # Pyramid only if under contract ceiling
        additional = ComputeContracts(new_signal.stop_pts, new_signal.confidence)
        additional = min(additional, 50 - total_open_contracts)
        if additional > 0:
            add_to_position(additional)
```

**Key constraint:** Never open a short while a long is profitable (and vice versa). The existing trade has priority; the new signal is skipped unless the drawdown threshold is met.

---

## 5. Same Model, New Signal While In Trade

Each of the 4 CNN models manages at most **one active trade** at a time.

- If a model is already in a trade, it does not generate new entry signals until its current position is flat.
- The model's output is polled every bar, but the entry gate checks `model_position_flat` before acting.
- Scaling into the same model's existing trade (pyramid) is allowed only via Rule 4 above.

---

## 6. Session End — Hard Flat Rule

**At 14:55 ET, all positions are closed unconditionally.**

```
OnBarUpdate():
    if Time >= 14:55:00 ET:
        if total_open_contracts > 0:
            CloseAllPositions()   // market order
            total_open_contracts = 0
        return  // no new entries permitted
```

- No new entries after 14:55 ET.
- No exceptions for high confidence or approaching targets.
- This prevents carrying positions into the 15:00 settlement and avoids TopStep's end-of-day drawdown trap.

---

## 7. Daily Loss Limit — Drawdown Guard

TopStep 50K drawdown rules:
- **Trailing max drawdown:** $2,000 trailing on unrealized highs
- **Daily loss limit:** $1,000

NinjaScript must track both independently:

```
// Called on every tick
OnMarketData():
    current_dd = (trailing_high_equity - current_equity)
    if current_dd >= 1800:          // warn at $1,800 (90% of $2,000)
        ReduceAllPositionsBy50Pct()
    if current_dd >= 2000:
        CloseAllPositions()
        block_new_entries_today = true

    daily_loss = (start_of_day_equity - current_equity)
    if daily_loss >= 900:           // warn at $900
        ReduceAllPositionsBy50Pct()
    if daily_loss >= 1000:
        CloseAllPositions()
        block_new_entries_today = true
```

`trailing_high_equity` resets to the new equity high on each tick when in profit. It does **not** reset at midnight — it is truly trailing across the combine duration.

---

## 8. Confidence Threshold Gate

A model's output is only acted upon if:

```
predicted_class != NoTrade (class 2)
AND confidence >= 0.60                // minimum threshold
AND NOT block_new_entries_today
AND Time < 14:55 ET
AND total_open_contracts < 50
AND ComputeContracts(stop_pts, confidence) >= 1
```

All conditions must be true. A single failing condition = no trade.

---

## 9. Stop and Target Calculation

These values are set at entry and never moved (no trailing stop during the trade):

```
atr = atr_14[0]          // 14-bar ATR at signal bar close
stop_pts = 1.5 * atr
target_pts = 2.5 * atr   // 1.67 R:R

long_stop   = entry - stop_pts
long_target = entry + target_pts
short_stop  = entry + stop_pts
short_target = entry - target_pts
```

Session boundary override: if `entry_time + (max_bars * bar_seconds)` would exceed 15:00, the target bar count is capped so the exit is at or before 15:00. The stop is still honored at its calculated price.

---

## 10. Common Logic Faults to Avoid

These are the mistakes AI agents commonly make when implementing this — explicitly prohibited:

| Fault | Why It's Wrong |
|---|---|
| Hard-coded `contracts = 5` or `contracts = 1` | Ignores confidence scaling and ATR stop |
| Checking per-model contract count instead of total | Allows 4 × 50 = 200 MNQ open |
| Trailing the stop during the trade | Invalidates triple-barrier label logic used in training |
| Moving the target to break-even after halfway | Not in the training objective — creates train/live mismatch |
| Allowing new entries after 14:55 | Violates session end rule |
| Resetting trailing_high_equity daily | Trailing DD is combine-duration, not daily |
| Using daily P&L instead of unrealized high for DD | Different calculation — use unrealized peak |
| Skipping the delta ≥ 0.05 check on conflicts | Enters both sides, immediately flat with 2× slippage |
| Opening multiple positions per model | Each model is single-position at a time |
| Using `mark-to-market` close price for target | Target is ATR-based at entry, not floating |

---

## 11. Execution Priority Order (per bar)

When multiple signals arrive on the same bar:

1. Apply session end gate (14:55) — if triggered, flat and return
2. Apply daily loss / drawdown gate — if triggered, flat and return  
3. Check existing positions for same-model signals (pyramid or skip)
4. Collect all NEW signals from models not currently in a trade
5. Group by direction (Long pool, Short pool)
6. If both pools non-empty → apply Rule 3B (delta test)
7. If one pool only → apply Rule 3A (aggregate contracts, enter)
8. Check total contract ceiling — reduce if needed
9. Submit entry order

This ordering is deterministic. No randomness. Given the same bar data and model outputs, the same trade decision is always reached.

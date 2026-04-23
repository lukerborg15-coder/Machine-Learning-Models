# Prop Firm Rules — TopStep 50K Evaluation

**Firm:** TopstepTrader
**Account Type:** Combine Evaluation (50K)
**Instrument:** MNQ (Micro Nasdaq)
**NinjaTrader Compatible:** Yes — TopStep supports NinjaTrader via Rithmic
**Future expansion:** `MES` is supported by the firm but is out of scope for current project docs

> ⚠️ **Verify current rules at topstep.com before building the risk simulator.** Prop firm rules update periodically. The rules below reflect the standard TopStep 50K Combine structure — confirm they are current before live deployment.

---

## TopStep 50K Combine Rules

| Rule | Value | Notes |
|---|---|---|
| Account Size | $50,000 | |
| Profit Target | $3,000 | Must reach this to pass Combine |
| Maximum Drawdown (end of day) | $2,000 | Trails from highest EOD balance reached — NOT from account start |
| Daily Loss Limit | $1,000 | Resets each calendar day |
| Max Contracts | **50 MNQ** | Equivalent to 5 NQ mini contracts. MNQ = 1/10 of NQ. |
| Consistency Rule | Yes | No single day may account for >40% of total profits |
| Time Limit | None | No expiration on standard Combine |

---

## Maximum Drawdown — Exact Mechanics

**The $2,000 maximum drawdown trails from the highest EOD balance reached.**

```
max_eod_balance = max(account_balance_at_each_day_close_so_far)
drawdown_floor  = max_eod_balance - 2000

If current_balance < drawdown_floor → account blown
```

**Example:**
- Start: $50,000 → floor = $48,000
- After Day 1 closes at $51,500 → floor = $49,500
- After Day 2 closes at $52,800 → floor = $50,800
- Day 3: account drops intraday to $50,700 → floor is $50,800 → BLOWN

**Critical trap:** Every profitable day permanently raises the floor. A strong week followed by one bad day can blow the account even while still overall profitable. The ML risk simulator must model this exactly.

---

## Daily Loss Limit — Exact Mechanics

**DLL = $1,000 per day.** Measured from start-of-day balance.

```
start_of_day_balance = account balance when session opens (09:30 ET)
daily_loss = start_of_day_balance - current_balance

If daily_loss >= 1000 → stop trading for the day immediately
```

Resets at the start of each new trading day. If open positions approach the DLL, the risk simulator closes them and blocks new entries for the remainder of that session.

---

## Consistency Rule

**No single day may account for more than 40% of total profits at time of pass.**

Example: total profit = $3,200 (passing). No single day can be more than $1,280 (40% × $3,200).

**Impact on ML strategy:**
- Do not let the model run large size on a single strong day — a big win may violate the consistency rule even if the account otherwise passed
- Target: consistent $300–$500 profit days rather than rare $2,000+ days
- The funded_sim.py must flag consistency violations during evaluation

```python
def check_consistency(daily_pnls):
    total_profit = sum(p for p in daily_pnls if p > 0)
    if total_profit <= 0:
        return True
    max_day = max(daily_pnls)
    return max_day / total_profit <= 0.40
```

---

## Position Sizing — 50 MNQ Max, Confidence-Based

**Max contracts:** 50 MNQ (TopStep 50K confirmed — equivalent to 5 NQ mini)
**MNQ point value:** $2/point
**Total P&L per point at 50 contracts:** 50 × $2 = **$100/point**

> ⚠️ **50 MNQ is the hard ceiling, NOT the per-trade size.** Running 50 contracts with a 10-point stop = $1,000 risk = the entire daily loss limit in one trade. Per-trade sizing is calculated dynamically using the formula below.

### Position Sizing Formula

Target risk per trade: **$500** (allows approximately 2 full losses before DLL, with buffer for slippage)

```python
def position_size(stop_pts: float, confidence: float, max_contracts: int = 50) -> int:
    """Compute MNQ contracts for a single trade.

    Steps:
    1. Compute base contracts to risk $500 at this stop distance
    2. Scale down by confidence (lower confidence = smaller size)
    3. Hard cap at max_contracts (50 MNQ, TopStep limit)
    4. Minimum 1 contract if confidence >= threshold

    Args:
        stop_pts: Stop distance in points (must be > 0)
        confidence: Model output confidence (0.0 to 1.0)
        max_contracts: Hard ceiling (50 MNQ for TopStep 50K)

    Returns:
        int: Number of MNQ contracts to trade (1 to 50)
    """
    point_value = 2  # MNQ
    target_risk = 500  # dollars per trade

    # Base size for $500 risk at this stop
    base = int(target_risk / (stop_pts * point_value))
    base = max(1, base)

    # Confidence scale
    if confidence >= 0.80:
        scale = 1.00
    elif confidence >= 0.70:
        scale = 0.80
    elif confidence >= 0.65:
        scale = 0.60
    else:  # 0.60–0.65
        scale = 0.40

    sized = max(1, int(base * scale))
    return min(sized, max_contracts)
```

### Stop Distance Reference

At target $500 risk per trade:

| Stop Distance (pts) | Base contracts | Risk at base | After 0.60 conf (×0.40) | After 0.80 conf (×1.00) |
|---|---|---|---|---|
| 5 pts | 50 | $500 | 20 MNQ / $200 | 50 MNQ / $500 |
| 10 pts | 25 | $500 | 10 MNQ / $200 | 25 MNQ / $500 |
| 15 pts | 16 | $480 | 6 MNQ / $180 | 16 MNQ / $480 |
| 20 pts | 12 | $480 | 4 MNQ / $160 | 12 MNQ / $480 |
| 30 pts | 8 | $480 | 3 MNQ / $180 | 8 MNQ / $480 |
| 40 pts | 6 | $480 | 2 MNQ / $160 | 6 MNQ / $480 |
| 50 pts | 5 | $500 | 2 MNQ / $200 | 5 MNQ / $500 |

**DLL math at $500 target risk:** $1,000 DLL ÷ $500 target risk = 2 losing trades before the day ends. Conservative by design — the trailing DD floor is the bigger threat than the DLL.

---

## Risk Management Class for ML Simulator

```python
class TopStepRiskManager:
    """
    CANONICAL IMPLEMENTATION — copy exactly into ml/topstep_risk.py.
    Do not create a separate version. This is the single source of truth.

    Key rules enforced:
    - Trailing DD from highest EOD balance (NOT from account start)
    - DLL from day_start_balance (NOT from account start)
    - Consistency rule: no single day > 40% of total profits
    - max_contracts = 50 MNQ (TopStep 50K confirmed)
    - position_size() uses dynamic sizing, NOT fixed contract count
    """

    def __init__(
        self,
        account_size: float = 50_000,
        profit_target: float = 3_000,
        max_trailing_dd: float = 2_000,
        daily_loss_limit: float = 1_000,
        max_contracts: int = 50,
        point_value: float = 2.0,
        commission_per_rt: float = 1.40,
    ):
        self.account = account_size
        self.initial_account = account_size
        self.profit_target = profit_target
        self.max_trailing_dd = max_trailing_dd
        self.daily_loss_limit = daily_loss_limit
        self.max_contracts = max_contracts     # 50 MNQ — TopStep 50K limit
        self.point_value = point_value         # $2/point for MNQ
        self.commission_per_rt = commission_per_rt  # $1.40 per round trip per contract

        self.max_equity_eod = account_size     # highest EOD balance seen — DD trails from here
        self.day_start_balance = account_size
        self.daily_pnls: list[float] = []
        self.active = True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(self, stop_pts: float, confidence: float) -> int:
        """Dynamic position size. See Prop Firm Rules.md for derivation.

        Target $500 risk per trade, scaled by confidence, capped at 50 MNQ.
        stop_pts must be > 0.
        """
        if stop_pts <= 0:
            raise ValueError("stop_pts must be positive")
        target_risk = 500.0
        base = int(target_risk / (stop_pts * self.point_value))
        base = max(1, base)
        if confidence >= 0.80:
            scale = 1.00
        elif confidence >= 0.70:
            scale = 0.80
        elif confidence >= 0.65:
            scale = 0.60
        else:
            scale = 0.40
        sized = max(1, int(base * scale))
        return min(sized, self.max_contracts)

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def simulate_trade(
        self,
        entry: float,
        stop: float,
        target: float,
        exit_price: float,
        contracts: int,
    ) -> float:
        """Simulate a completed trade. Returns net P&L in dollars.

        Args:
            entry: Entry price
            stop: Stop price (used only to infer direction)
            target: Target price (used only to infer direction)
            exit_price: Actual exit price (stop, target, or time exit)
            contracts: Number of MNQ contracts traded

        Direction is inferred from stop vs entry:
            stop < entry → long trade (profit when exit > entry)
            stop > entry → short trade (profit when exit < entry)
        """
        direction = 1 if stop < entry else -1
        raw_pnl = (exit_price - entry) * direction * contracts * self.point_value
        cost = self.commission_per_rt * contracts
        return raw_pnl - cost

    # ------------------------------------------------------------------
    # Session / day management
    # ------------------------------------------------------------------

    def update_eod(self, eod_balance: float, day_pnl: float) -> None:
        """Call at end of each trading day.

        Updates trailing DD floor from highest EOD balance seen.
        CRITICAL: floor trails from max_equity_eod, not from account start.
        If eod_balance < (max_equity_eod - max_trailing_dd) → account blown.
        """
        self.daily_pnls.append(day_pnl)
        self.max_equity_eod = max(self.max_equity_eod, eod_balance)
        self.day_start_balance = eod_balance
        self.account = eod_balance
        if eod_balance < (self.max_equity_eod - self.max_trailing_dd):
            self.active = False

    def check_intraday(self, current_balance: float) -> bool:
        """Returns True if trading is allowed, False if DLL hit.

        DLL is measured from day_start_balance, not from account start.
        CRITICAL: use self.daily_loss_limit, not hardcoded 1000.
        """
        daily_loss = self.day_start_balance - current_balance
        return daily_loss < self.daily_loss_limit

    # ------------------------------------------------------------------
    # Pass / consistency checks
    # ------------------------------------------------------------------

    def is_passed(self) -> bool:
        """True if Combine passed: profit >= target AND consistency OK AND account active."""
        if not self.active:
            return False
        total_profit = self.account - self.initial_account
        if total_profit < self.profit_target:
            return False
        return self.check_consistency()

    def check_consistency(self) -> bool:
        """True if no single day > 40% of total positive profit."""
        total_profit = sum(p for p in self.daily_pnls if p > 0)
        if total_profit <= 0:
            return False
        max_day = max(self.daily_pnls) if self.daily_pnls else 0
        return (max_day / total_profit) <= 0.40
```

---

## Common Logic Errors — Guard Against These

1. **Trailing DD from account start (wrong):** `current - 50000 > 2000` is WRONG. Must be `max_equity_eod - current > 2000`.
2. **DLL from account start (wrong):** Must use `day_start_balance`, not `initial_account`.
3. **Fixed 5 contracts:** The old codebase used `self.contracts = 5`. This is now `max_contracts = 50` with `position_size()` computing dynamic sizing. Any reference to `self.contracts` as a fixed value is a bug.
4. **direction from target > entry:** This assumes long. Use `stop < entry` to infer long (stop below entry = long position). Target could be set incorrectly by an agent.
5. **Consistency rule with no profitable days:** If `total_profit <= 0`, consistency returns False — account cannot pass with no profits. Keep this behavior.
6. **Commission per contract vs per trade:** `commission_per_rt` is per contract, multiplied by contracts. Do not apply once per trade regardless of size.

---

## Notes

- TopStep is NinjaTrader-compatible via Rithmic — no custom API needed
- The trailing DD mechanic is the primary account killer — the ML model must be evaluated against this, not just Sharpe/Calmar
- The consistency rule is unique to TopStep (Apex does not have it). It materially changes what "good" performance looks like: a model with one home-run day and nine small losses may have positive Sharpe but fail the consistency rule
- Always report in the evaluation: Combine pass rate, average Combine duration, EOD DD trigger frequency, consistency violation count

"""TopStep risk manager."""

from __future__ import annotations


class TopStepRiskManager:
    """
    CANONICAL IMPLEMENTATION - Agent 4B copies this exactly.
    Do not create a separate version. This is the single source of truth.
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
        self.max_contracts = max_contracts
        self.point_value = point_value
        self.commission_per_rt = commission_per_rt
        self.max_equity_eod = account_size
        self.day_start_balance = account_size
        self.daily_pnls: list[float] = []
        self.active = True

    def position_size(self, stop_pts: float, confidence: float) -> int:
        """Compute dynamic MNQ contracts for one trade."""
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

    def simulate_trade(self, entry, stop, target, exit_price, contracts):
        """Simulate a completed trade, return P&L in dollars"""
        direction = 1 if stop < entry else -1
        raw_pnl = (exit_price - entry) * direction * contracts * self.point_value
        cost = self.commission_per_rt * contracts
        return raw_pnl - cost

    def update_eod(self, eod_balance, day_pnl):
        self.daily_pnls.append(day_pnl)
        # Update trailing floor
        self.max_equity_eod = max(self.max_equity_eod, eod_balance)
        self.day_start_balance = eod_balance
        self.account = eod_balance
        # Check trailing DD breach - uses self.max_trailing_dd, NOT hardcoded
        if eod_balance < (self.max_equity_eod - self.max_trailing_dd):
            self.active = False

    def check_intraday(self, current_balance):
        """Returns True if trading is allowed, False if DLL hit"""
        daily_loss = self.day_start_balance - current_balance
        if daily_loss >= self.daily_loss_limit:
            return False
        return True

    def is_passed(self):
        if not self.active:
            return False
        total_profit = self.account - self.initial_account
        if total_profit < self.profit_target:
            return False
        return self.check_consistency()

    def check_consistency(self):
        total_profit = sum(p for p in self.daily_pnls if p > 0)
        if total_profit <= 0:
            return False
        max_day = max(self.daily_pnls) if self.daily_pnls else 0
        return max_day / total_profit <= 0.40


__all__ = ["TopStepRiskManager"]

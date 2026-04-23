"""Express Funded Account simulation helpers for Agent 3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import json
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.evaluate import (
    DEFAULT_COMMISSION_PER_RT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_STOP_ATR_MULT,
    DEFAULT_TARGET_R,
    OBJECTIVE_META_LABEL,
    OBJECTIVE_THREE_CLASS,
    _resolve_atr,
    _resolve_signal_column,
)
from ml.topstep_risk import TopStepRiskManager


PayoutPath = Literal["standard", "consistency"]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class ExpressFundedRules:
    """Topstep 50K Express Funded assumptions used for payout-readiness ranking."""

    starting_balance: float = 0.0
    max_loss_limit: float = 2_000.0
    daily_loss_limit: float = 1_000.0
    max_contracts: int = 50
    point_value: float = 2.0
    commission_per_rt: float = DEFAULT_COMMISSION_PER_RT
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT
    target_r: float = DEFAULT_TARGET_R
    standard_winning_day_threshold: float = 150.0
    standard_winning_days_required: int = 5
    consistency_traded_days_required: int = 3
    consistency_target: float = 0.40
    standard_payout_cap: float = 5_000.0
    consistency_payout_cap: float = 6_000.0
    minimum_payout_request: float = 125.0
    payout_fraction: float = 0.50
    payout_buffer: float = 1_000.0
    profit_split: float = 0.90
    require_full_window_each_payout: bool = True
    apply_daily_loss_limit: bool = True


@dataclass(frozen=True)
class DeploymentDecision:
    approved: bool
    reason: str
    per_fold_metrics: list[dict[str, Any]]


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _risk_manager_from_rules(rules: ExpressFundedRules) -> TopStepRiskManager:
    return TopStepRiskManager(
        account_size=rules.starting_balance,
        max_trailing_dd=rules.max_loss_limit,
        daily_loss_limit=rules.daily_loss_limit,
        max_contracts=rules.max_contracts,
        point_value=rules.point_value,
        commission_per_rt=rules.commission_per_rt,
    )


def _confidence_for_sizing(row: pd.Series, default_confidence: float) -> float:
    if "confidence" not in row.index or pd.isna(row["confidence"]):
        return default_confidence
    confidence = _finite_float(row["confidence"])
    return confidence if np.isfinite(confidence) else default_confidence


def evaluate_deployment_gate(per_fold_metrics: list[dict[str, Any]] | pd.DataFrame) -> DeploymentDecision:
    """Apply the Agent 3C 5-fold deployment gate (legacy path)."""
    if isinstance(per_fold_metrics, pd.DataFrame):
        records = per_fold_metrics.loc[per_fold_metrics["fold"].astype(str) != "summary"].to_dict(orient="records")
    else:
        records = [dict(record) for record in per_fold_metrics if str(record.get("fold", "")) != "summary"]

    if len(records) != 5:
        return DeploymentDecision(False, f"requires 5 folds, found {len(records)}", records)

    sharpes = np.asarray([_finite_float(record.get("test_sharpe", float("nan"))) for record in records], dtype=float)
    pass_rates = np.asarray([_finite_float(record.get("combine_pass_rate", 0.0)) for record in records], dtype=float)

    if np.isnan(sharpes).any() or not bool((sharpes > 0.0).all()):
        failing = [
            str(record.get("fold", idx + 1))
            for idx, record in enumerate(records)
            if not np.isfinite(sharpes[idx]) or sharpes[idx] <= 0.0
        ]
        return DeploymentDecision(False, f"failed: non-positive test Sharpe in {', '.join(failing)}", records)

    median_sharpe = float(np.median(sharpes))
    if median_sharpe < 0.5:
        return DeploymentDecision(False, f"failed: median test Sharpe {median_sharpe:.3f} < 0.500", records)

    pass_rate_folds = int((pass_rates >= 0.5).sum())
    if pass_rate_folds < 4:
        return DeploymentDecision(False, f"failed: only {pass_rate_folds}/5 folds have Combine pass-rate >= 50%", records)

    return DeploymentDecision(
        True,
        f"passed: all folds Sharpe > 0, median Sharpe {median_sharpe:.3f}, {pass_rate_folds}/5 folds pass-rate >= 50%",
        records,
    )


def evaluate_bootstrap_deployment_gate(
    strategy_results: Mapping[str, Any],
    *,
    p05_sharpe_floor: float = 0.0,
    per_fold_sharpe_floor: float = -0.3,
    p50_profit_factor_floor: float = 1.2,
    p05_pass_rate_floor: float = 0.30,
    require_p50_sharpe_above: float | None = 0.5,
) -> DeploymentDecision:
    """Agent 3E deployment gate on top of bootstrap confidence intervals.

    Required conditions (all must hold):

    1. Aggregated p05 Sharpe > ``p05_sharpe_floor`` (default 0).
    2. Every individual fold's point Sharpe > ``per_fold_sharpe_floor`` (default -0.3).
    3. Aggregated p50 profit factor >= ``p50_profit_factor_floor`` (default 1.2).
    4. Aggregated p05 Combine pass-rate >= ``p05_pass_rate_floor`` (default 0.30).

    If ``require_p50_sharpe_above`` is not None, an additional multiple-testing
    filter is applied: aggregated p50 Sharpe must be >= that value (default 0.5).
    Set to ``None`` to disable the multiple-testing filter.

    ``strategy_results`` must contain ``"aggregated"`` (dict of bootstrap metric
    dicts with at least ``sharpe``, ``profit_factor``, ``pass_rate``) and
    ``"per_fold"`` (list of dicts each carrying a ``sharpe`` bootstrap dict with
    at least a ``point`` key).
    """
    aggregated = dict(strategy_results.get("aggregated", {}))
    per_fold = [dict(fold) for fold in strategy_results.get("per_fold", [])]

    def _fetch(metric_dict: Mapping[str, Any], metric: str, key: str) -> float:
        entry = metric_dict.get(metric, {}) or {}
        return _finite_float(entry.get(key, float("nan")))

    agg_sharpe_p05 = _fetch(aggregated, "sharpe", "p05")
    if not np.isfinite(agg_sharpe_p05) or agg_sharpe_p05 <= p05_sharpe_floor:
        return DeploymentDecision(
            False,
            f"failed: aggregated p05 Sharpe = {agg_sharpe_p05:.3f} <= {p05_sharpe_floor:.2f}",
            per_fold,
        )

    fold_points: list[tuple[str, float]] = []
    for idx, fold in enumerate(per_fold):
        point_sharpe = _finite_float((fold.get("sharpe") or {}).get("point", float("nan")))
        fold_points.append((str(fold.get("fold", f"fold_{idx + 1}")), point_sharpe))
    bad_folds = [
        name for name, value in fold_points if not np.isfinite(value) or value <= per_fold_sharpe_floor
    ]
    if bad_folds:
        details = ", ".join(f"{name}={value:.3f}" for name, value in fold_points if name in bad_folds)
        return DeploymentDecision(
            False,
            f"failed: per-fold point Sharpe <= {per_fold_sharpe_floor:.2f} in {details}",
            per_fold,
        )

    agg_pf_p50 = _fetch(aggregated, "profit_factor", "p50")
    if not np.isfinite(agg_pf_p50) or agg_pf_p50 < p50_profit_factor_floor:
        return DeploymentDecision(
            False,
            f"failed: aggregated p50 profit factor = {agg_pf_p50:.3f} < {p50_profit_factor_floor:.2f}",
            per_fold,
        )

    agg_pass_p05 = _fetch(aggregated, "pass_rate", "p05")
    if not np.isfinite(agg_pass_p05) or agg_pass_p05 < p05_pass_rate_floor:
        return DeploymentDecision(
            False,
            f"failed: aggregated p05 pass rate = {agg_pass_p05:.3f} < {p05_pass_rate_floor:.2f}",
            per_fold,
        )

    if require_p50_sharpe_above is not None:
        agg_sharpe_p50 = _fetch(aggregated, "sharpe", "p50")
        if not np.isfinite(agg_sharpe_p50) or agg_sharpe_p50 < require_p50_sharpe_above:
            return DeploymentDecision(
                False,
                f"failed: aggregated p50 Sharpe = {agg_sharpe_p50:.3f} < {require_p50_sharpe_above:.2f} (multiple-testing filter)",
                per_fold,
            )

    return DeploymentDecision(
        True,
        (
            f"passed: p05 Sharpe {agg_sharpe_p05:.3f} > {p05_sharpe_floor:.2f}, "
            f"p50 PF {agg_pf_p50:.3f} >= {p50_profit_factor_floor:.2f}, "
            f"p05 pass-rate {agg_pass_p05:.3f} >= {p05_pass_rate_floor:.2f}, "
            f"all folds point Sharpe > {per_fold_sharpe_floor:.2f}"
        ),
        per_fold,
    )


def _initial_mll_floor(rules: ExpressFundedRules) -> float:
    return rules.starting_balance - rules.max_loss_limit


def _updated_mll_floor(max_eod_balance: float, first_payout_taken: bool, rules: ExpressFundedRules) -> float:
    if first_payout_taken:
        return 0.0
    return max(_initial_mll_floor(rules), min(0.0, max_eod_balance - rules.max_loss_limit))


def _check_standard_eligibility(window_day_pnls: list[float], rules: ExpressFundedRules) -> bool:
    winning_days = sum(pnl >= rules.standard_winning_day_threshold for pnl in window_day_pnls)
    return winning_days >= rules.standard_winning_days_required and sum(window_day_pnls) > 0


def _check_consistency_eligibility(
    window_day_pnls: list[float],
    window_trade_counts: list[int],
    rules: ExpressFundedRules,
) -> bool:
    traded_days = sum(count > 0 for count in window_trade_counts)
    total_net_profit = sum(window_day_pnls)
    largest_winning_day = max([pnl for pnl in window_day_pnls if pnl > 0], default=0.0)
    if traded_days < rules.consistency_traded_days_required or total_net_profit <= 0 or largest_winning_day <= 0:
        return False
    return largest_winning_day / total_net_profit <= rules.consistency_target


def _is_payout_eligible(
    payout_path: PayoutPath,
    window_day_pnls: list[float],
    window_trade_counts: list[int],
    payout_count: int,
    rules: ExpressFundedRules,
) -> bool:
    if payout_count > 0 and not rules.require_full_window_each_payout:
        return sum(window_day_pnls) > 0
    if payout_path == "standard":
        return _check_standard_eligibility(window_day_pnls, rules)
    if payout_path == "consistency":
        return _check_consistency_eligibility(window_day_pnls, window_trade_counts, rules)
    raise ValueError(f"Unsupported payout path: {payout_path}")


def _safe_payout_amount(account_balance: float, payout_path: PayoutPath, rules: ExpressFundedRules) -> float:
    payout_cap = rules.standard_payout_cap if payout_path == "standard" else rules.consistency_payout_cap
    allowed_amount = min(max(account_balance, 0.0) * rules.payout_fraction, payout_cap)
    safe_amount = max(account_balance - rules.payout_buffer, 0.0)
    payout = max(min(allowed_amount, safe_amount), 0.0)
    if payout < rules.minimum_payout_request:
        return 0.0
    return payout


def _account_floor_threshold_price(
    entry: float,
    account_before_trade: float,
    account_floor: float,
    direction: int,
    contracts: int,
    rules: ExpressFundedRules,
) -> float | None:
    dollars_per_point = contracts * rules.point_value
    if dollars_per_point <= 0:
        return None

    commission = rules.commission_per_rt * contracts
    threshold_net_pnl = account_floor - account_before_trade
    price = entry + direction * ((threshold_net_pnl + commission) / dollars_per_point)
    if direction > 0 and price > entry:
        return None
    if direction < 0 and price < entry:
        return None
    return float(price)


def _funded_exit_price_and_index(
    frame: pd.DataFrame,
    entry_pos: int,
    direction: int,
    entry: float,
    stop: float,
    target: float,
    account_before_trade: float,
    day_start_balance: float,
    mll_floor: float,
    contracts: int,
    rules: ExpressFundedRules,
) -> tuple[float, int, str]:
    """Find a conservative funded exit, prioritizing intrabar DLL/MLL liquidation risk."""
    risk_thresholds: list[tuple[float, str]] = []
    mll_threshold = _account_floor_threshold_price(
        entry=entry,
        account_before_trade=account_before_trade,
        account_floor=mll_floor,
        direction=direction,
        contracts=contracts,
        rules=rules,
    )
    if mll_threshold is not None:
        risk_thresholds.append((mll_threshold, "max_loss_limit"))

    if rules.apply_daily_loss_limit:
        dll_threshold = _account_floor_threshold_price(
            entry=entry,
            account_before_trade=account_before_trade,
            account_floor=day_start_balance - rules.daily_loss_limit,
            direction=direction,
            contracts=contracts,
            rules=rules,
        )
        if dll_threshold is not None:
            risk_thresholds.append((dll_threshold, "daily_loss_limit"))

    entry_day = frame.index[entry_pos].normalize()
    last_same_day_pos = entry_pos

    for cursor in range(entry_pos + 1, len(frame)):
        if frame.index[cursor].normalize() != entry_day:
            break

        last_same_day_pos = cursor
        row = frame.iloc[cursor]
        high = float(row["high"])
        low = float(row["low"])

        risk_hits = [
            (price, reason)
            for price, reason in risk_thresholds
            if (low <= price if direction > 0 else high >= price)
        ]
        if risk_hits:
            # Intrabar path is unknown, so liquidation risk wins over any same-bar target fill.
            exit_price, reason = min(risk_hits, key=lambda item: item[0]) if direction > 0 else max(
                risk_hits, key=lambda item: item[0]
            )
            return exit_price, cursor, reason

        if direction > 0:
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit:
                return stop, cursor, "stop"
            if target_hit:
                return target, cursor, "target"
        else:
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit:
                return stop, cursor, "stop"
            if target_hit:
                return target, cursor, "target"

    exit_row = frame.iloc[last_same_day_pos]
    return float(exit_row["close"]), last_same_day_pos, "session_close"


def simulate_express_funded(
    frame: pd.DataFrame,
    strategy_name: str,
    payout_path: PayoutPath = "standard",
    signal_column: str | None = None,
    start_after: pd.Timestamp | str | None = None,
    rules: ExpressFundedRules | None = None,
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Simulate Express Funded trading from model predictions until failure or data exhaustion."""
    resolved_rules = rules or ExpressFundedRules()
    trading_frame = frame.copy().sort_index()
    if not isinstance(trading_frame.index, pd.DatetimeIndex):
        raise ValueError("Express Funded simulation requires a DatetimeIndex")

    if start_after is not None:
        start_ts = pd.Timestamp(start_after)
        if start_ts.tzinfo is None and trading_frame.index.tz is not None:
            start_ts = start_ts.tz_localize(trading_frame.index.tz)
        elif start_ts.tzinfo is not None and trading_frame.index.tz is not None:
            start_ts = start_ts.tz_convert(trading_frame.index.tz)
        trading_frame = trading_frame.loc[trading_frame.index.normalize() > start_ts.normalize()]

    resolved_signal_column = _resolve_signal_column(strategy_name, signal_column)
    required_columns = {"open", "high", "low", "close", "prediction", resolved_signal_column}
    missing = required_columns.difference(trading_frame.columns)
    if missing:
        raise ValueError(f"Missing required funded simulation columns: {sorted(missing)}")

    atr = _resolve_atr(trading_frame)
    if atr is None:
        raise ValueError("Express Funded simulation requires either ATR columns or OHLC prices")
    trading_frame["atr_eval"] = atr
    normalized_objective = objective.strip().lower()
    if normalized_objective not in {OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL}:
        raise ValueError(f"Unsupported funded simulation objective: {objective}")

    risk_manager = _risk_manager_from_rules(resolved_rules)
    account = resolved_rules.starting_balance
    max_eod_balance = resolved_rules.starting_balance
    mll_floor = _initial_mll_floor(resolved_rules)
    first_payout_taken = False
    active = True
    failure_reason = ""

    trade_records: list[dict[str, Any]] = []
    ledger_records: list[dict[str, Any]] = []
    payout_records: list[dict[str, Any]] = []
    window_day_pnls: list[float] = []
    window_trade_counts: list[int] = []

    if trading_frame.empty:
        return {
            "payout_path": payout_path,
            "status": "insufficient_post_pass_data",
            "active": True,
            "failure_reason": "insufficient_post_pass_data",
            "ending_balance": account,
            "mll_floor": mll_floor,
            "gross_payouts": 0.0,
            "trader_payouts": 0.0,
            "payout_count": 0,
            "trade_count": 0,
            "ledger": pd.DataFrame(),
            "payouts": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "rules": asdict(resolved_rules),
        }

    for day, day_frame in trading_frame.groupby(trading_frame.index.normalize(), sort=True):
        if not active:
            break

        day_start_balance = account
        session_contracts = max(int(resolved_rules.max_contracts), 1)
        day_pnl = 0.0
        day_trade_count = 0
        dll_breached = False
        day_indices = list(day_frame.index)
        cursor = 0

        while cursor < len(day_frame) - 1 and active and not dll_breached:
            row = day_frame.iloc[cursor]
            setup_signal = int(row[resolved_signal_column])
            prediction = int(row["prediction"])
            atr_value = float(row["atr_eval"]) if pd.notna(row["atr_eval"]) else float("nan")
            if setup_signal == 0 or prediction not in (0, 1) or not np.isfinite(atr_value) or atr_value <= 0:
                cursor += 1
                continue
            if normalized_objective == OBJECTIVE_META_LABEL:
                if prediction != 1:
                    cursor += 1
                    continue
                if "confidence" in row.index:
                    confidence = float(row["confidence"]) if pd.notna(row["confidence"]) else float("nan")
                    if not np.isfinite(confidence) or confidence < confidence_threshold:
                        cursor += 1
                        continue
                direction = 1 if setup_signal > 0 else -1
            else:
                if (setup_signal > 0 and prediction != 0) or (setup_signal < 0 and prediction != 1):
                    cursor += 1
                    continue
                direction = 1 if prediction == 0 else -1

            entry = float(row["close"])
            stop_distance = atr_value * resolved_rules.stop_atr_mult
            sizing_confidence = _confidence_for_sizing(row, confidence_threshold)
            try:
                trade_contracts = risk_manager.position_size(stop_distance, sizing_confidence)
            except ValueError:
                cursor += 1
                continue
            stop = entry - stop_distance if direction > 0 else entry + stop_distance
            target = entry + (stop_distance * resolved_rules.target_r) if direction > 0 else entry - (
                stop_distance * resolved_rules.target_r
            )
            account_before_trade = account
            exit_price, exit_pos, exit_reason = _funded_exit_price_and_index(
                day_frame,
                entry_pos=cursor,
                direction=direction,
                entry=entry,
                stop=stop,
                target=target,
                account_before_trade=account_before_trade,
                day_start_balance=day_start_balance,
                mll_floor=mll_floor,
                contracts=trade_contracts,
                rules=resolved_rules,
            )
            pnl = risk_manager.simulate_trade(
                entry=entry,
                stop=stop,
                target=target,
                exit_price=exit_price,
                contracts=trade_contracts,
            )

            account += pnl
            day_pnl += pnl
            day_trade_count += 1

            if exit_reason == "max_loss_limit" or account <= mll_floor + 1e-9:
                active = False
                failure_reason = "max_loss_limit"
            if (
                exit_reason == "daily_loss_limit"
                or (
                    resolved_rules.apply_daily_loss_limit
                    and day_start_balance - account >= resolved_rules.daily_loss_limit - 1e-9
                )
            ):
                dll_breached = True

            trade_records.append(
                {
                    "entry_time": day_indices[cursor],
                    "exit_time": day_indices[exit_pos],
                    "payout_path": payout_path,
                    "setup_signal": setup_signal,
                    "prediction": prediction,
                    "confidence": float(row["confidence"]) if "confidence" in row.index and pd.notna(row["confidence"]) else np.nan,
                    "direction": direction,
                    "entry": entry,
                    "target": target,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "contracts": trade_contracts,
                    "pnl": pnl,
                    "account_after_trade": account,
                    "mll_floor": mll_floor,
                }
            )
            cursor = max(exit_pos + 1, cursor + 1)

        max_eod_balance = max(max_eod_balance, account)
        mll_floor = _updated_mll_floor(max_eod_balance, first_payout_taken, resolved_rules)
        if account <= mll_floor + 1e-9 and active:
            active = False
            failure_reason = "max_loss_limit"

        window_day_pnls.append(day_pnl)
        window_trade_counts.append(day_trade_count)
        payout_eligible = active and _is_payout_eligible(
            payout_path=payout_path,
            window_day_pnls=window_day_pnls,
            window_trade_counts=window_trade_counts,
            payout_count=len(payout_records),
            rules=resolved_rules,
        )

        payout_amount = 0.0
        if payout_eligible:
            payout_amount = _safe_payout_amount(account, payout_path, resolved_rules)
            if payout_amount > 0:
                account -= payout_amount
                first_payout_taken = True
                mll_floor = 0.0
                payout_records.append(
                    {
                        "date": pd.Timestamp(day),
                        "payout_path": payout_path,
                        "gross_payout": payout_amount,
                        "trader_payout": payout_amount * resolved_rules.profit_split,
                        "account_after_payout": account,
                        "mll_floor": mll_floor,
                    }
                )
                window_day_pnls = []
                window_trade_counts = []

        ledger_records.append(
            {
                "date": pd.Timestamp(day),
                "payout_path": payout_path,
                "day_pnl": day_pnl,
                "trade_count": day_trade_count,
                "session_contracts": session_contracts,
                "dll_breached": dll_breached,
                "account_eod": account,
                "mll_floor": mll_floor,
                "payout_eligible": payout_eligible,
                "payout_amount": payout_amount,
                "active": active,
                "failure_reason": failure_reason,
            }
        )

    payouts = pd.DataFrame(payout_records)
    ledger = pd.DataFrame(ledger_records)
    trades = pd.DataFrame(trade_records)
    gross_payouts = float(payouts["gross_payout"].sum()) if not payouts.empty else 0.0
    trader_payouts = float(payouts["trader_payout"].sum()) if not payouts.empty else 0.0
    if active and not failure_reason:
        failure_reason = "data_exhausted"

    return {
        "payout_path": payout_path,
        "status": "active" if active else "failed",
        "active": active,
        "failure_reason": failure_reason,
        "ending_balance": float(account),
        "mll_floor": float(mll_floor),
        "gross_payouts": gross_payouts,
        "trader_payouts": trader_payouts,
        "payout_count": int(len(payouts)),
        "trade_count": int(len(trades)),
        "ledger": ledger,
        "payouts": payouts,
        "trades": trades,
        "rules": asdict(resolved_rules),
    }


def simulate_both_express_paths(
    frame: pd.DataFrame,
    strategy_name: str,
    signal_column: str | None = None,
    start_after: pd.Timestamp | str | None = None,
    rules: ExpressFundedRules | None = None,
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Run Standard and Consistency Express Funded paths for the same post-Combine stream."""
    return {
        payout_path: simulate_express_funded(
            frame=frame,
            strategy_name=strategy_name,
            payout_path=payout_path,
            signal_column=signal_column,
            start_after=start_after,
            rules=rules,
            objective=objective,
            confidence_threshold=confidence_threshold,
        )
        for payout_path in ("standard", "consistency")
    }


def payout_adjusted_survival_score(result: dict[str, Any]) -> float:
    """Simple payout-first ranking score for Agent 3 reports."""
    trader_payouts = float(result.get("trader_payouts", 0.0))
    ending_balance = float(result.get("ending_balance", 0.0))
    survival_bonus = 500.0 if result.get("active", False) else 0.0
    trade_count = int(result.get("trade_count", 0))
    sparse_penalty = 500.0 if trade_count < 5 else 0.0
    return trader_payouts + max(ending_balance, 0.0) * 0.10 + survival_bonus - sparse_penalty


def simulate_funded_after_combine(
    frame: pd.DataFrame,
    strategy_name: str,
    combine_result: dict[str, Any],
    signal_column: str | None = None,
    rules: ExpressFundedRules | None = None,
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run both Express Funded paths only if the preceding Combine simulation passed."""
    pass_day = combine_result.get("pass_day")
    if pd.isna(pass_day) or pass_day is None:
        return {
            "combine_passed": False,
            "status": "combine_not_passed",
            "best_path": None,
            "best_score": float("-inf"),
            "paths": {},
        }

    paths = simulate_both_express_paths(
        frame=frame,
        strategy_name=strategy_name,
        signal_column=signal_column,
        start_after=pass_day,
        rules=rules,
        objective=objective,
        confidence_threshold=confidence_threshold,
    )
    best_path = max(paths, key=lambda name: payout_adjusted_survival_score(paths[name]))
    return {
        "combine_passed": True,
        "status": "funded_simulated",
        "best_path": best_path,
        "best_score": payout_adjusted_survival_score(paths[best_path]),
        "paths": paths,
    }


def _format_metric(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _format_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _fold_metric_records(strategy_name: str, artifact_dir: Path | str = ARTIFACT_DIR) -> pd.DataFrame:
    eval_path = Path(artifact_dir) / f"eval_{strategy_name}.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation artifact: {eval_path}")
    frame = pd.read_csv(eval_path)
    if "fold" not in frame.columns:
        raise ValueError(f"Evaluation artifact missing fold column: {eval_path}")
    return frame


def build_deployment_decisions(
    strategy_names: list[str] | None = None,
    artifact_dir: Path | str = ARTIFACT_DIR,
) -> pd.DataFrame:
    from ml.train import build_training_jobs

    rows: list[dict[str, Any]] = []
    for job in build_training_jobs(strategy_names):
        eval_frame = _fold_metric_records(job.strategy_name, artifact_dir=artifact_dir)
        per_fold = eval_frame.loc[eval_frame["fold"].astype(str) != "summary"].copy()
        decision = evaluate_deployment_gate(per_fold)
        sharpes = pd.to_numeric(per_fold.get("test_sharpe", pd.Series(dtype=float)), errors="coerce")
        pass_rates = pd.to_numeric(per_fold.get("combine_pass_rate", pd.Series(dtype=float)), errors="coerce")
        trade_counts = pd.to_numeric(per_fold.get("trade_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
        low_sample_folds = per_fold.loc[trade_counts < 30, "fold"].astype(str).tolist()
        rows.append(
            {
                "strategy_name": job.strategy_name,
                "deployment_candidate": decision.approved,
                "deployment_reason": decision.reason,
                "median_test_sharpe": float(sharpes.median()) if not sharpes.empty else float("nan"),
                "min_test_sharpe": float(sharpes.min()) if not sharpes.empty else float("nan"),
                "max_test_sharpe": float(sharpes.max()) if not sharpes.empty else float("nan"),
                "combine_pass_folds": int((pass_rates >= 0.5).sum()),
                "low_sample_folds": ", ".join(low_sample_folds),
            }
        )
    decisions = pd.DataFrame(rows).sort_values(
        ["deployment_candidate", "median_test_sharpe", "combine_pass_folds"],
        ascending=False,
    )
    output_path = Path(artifact_dir) / "agent3c_deployment_decisions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_path, index=False)
    return decisions


def _load_bootstrap_artifact(
    strategy_name: str,
    artifact_dir: Path | str = ARTIFACT_DIR,
) -> dict[str, Any] | None:
    """Read ``eval_{strategy}_bootstrap.json`` produced by ``evaluate.compute_bootstrap_cis``."""
    path = Path(artifact_dir) / f"eval_{strategy_name}_bootstrap.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_bootstrap_deployment_decisions(
    strategy_names: list[str] | None = None,
    artifact_dir: Path | str = ARTIFACT_DIR,
    require_p50_sharpe_above: float | None = 0.5,
) -> pd.DataFrame:
    """Agent 3E: apply the 4-condition bootstrap gate to every strategy."""
    from ml.train import build_training_jobs

    rows: list[dict[str, Any]] = []
    for job in build_training_jobs(strategy_names):
        payload = _load_bootstrap_artifact(job.strategy_name, artifact_dir=artifact_dir)
        if payload is None:
            rows.append(
                {
                    "strategy_name": job.strategy_name,
                    "deployment_candidate": False,
                    "deployment_reason": "missing bootstrap artifact (run `python ml/evaluate.py --bootstrap`)",
                    "agg_sharpe_p05": float("nan"),
                    "agg_sharpe_p50": float("nan"),
                    "agg_pf_p50": float("nan"),
                    "agg_pass_p05": float("nan"),
                    "worst_fold_sharpe": float("nan"),
                    "n_trades": 0,
                }
            )
            continue
        decision = evaluate_bootstrap_deployment_gate(
            payload, require_p50_sharpe_above=require_p50_sharpe_above
        )
        agg = payload.get("aggregated", {}) or {}
        per_fold = payload.get("per_fold", []) or []
        sharpe_points = [
            _finite_float((fold.get("sharpe") or {}).get("point", float("nan"))) for fold in per_fold
        ]
        worst_fold_sharpe = float(min(sharpe_points)) if sharpe_points else float("nan")
        rows.append(
            {
                "strategy_name": job.strategy_name,
                "deployment_candidate": decision.approved,
                "deployment_reason": decision.reason,
                "agg_sharpe_p05": _finite_float((agg.get("sharpe") or {}).get("p05", float("nan"))),
                "agg_sharpe_p50": _finite_float((agg.get("sharpe") or {}).get("p50", float("nan"))),
                "agg_pf_p50": _finite_float((agg.get("profit_factor") or {}).get("p50", float("nan"))),
                "agg_pass_p05": _finite_float((agg.get("pass_rate") or {}).get("p05", float("nan"))),
                "worst_fold_sharpe": worst_fold_sharpe,
                "n_trades": int(agg.get("n_trades", 0)),
            }
        )

    decisions = pd.DataFrame(rows)
    if not decisions.empty:
        decisions = decisions.sort_values(
            ["deployment_candidate", "agg_sharpe_p05", "agg_sharpe_p50"], ascending=False
        )
    output_path = Path(artifact_dir) / "agent3e_deployment_decisions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_path, index=False)
    return decisions


def write_agent3c_final_report(
    decisions: pd.DataFrame,
    artifact_dir: Path | str = ARTIFACT_DIR,
) -> Path:
    artifact_path = Path(artifact_dir)
    output_path = artifact_path / "FINAL_EVAL_REPORT.md"
    approved_count = int(decisions["deployment_candidate"].sum()) if not decisions.empty else 0
    lines = [
        "# Final Evaluation Report - Agent 3C",
        "",
        f"**Deployment candidates:** {approved_count}",
        "",
        "## Deployment Gate",
        "",
        "| Strategy | Deploy | Median Test Sharpe | Min Fold Sharpe | Max Fold Sharpe | Combine Folds >=50% | Low Sample Folds | Reason |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in decisions.to_dict(orient="records"):
        lines.append(
            "| {strategy_name} | {deployment_candidate} | {median_test_sharpe} | {min_test_sharpe} | {max_test_sharpe} | {combine_pass_folds}/5 | {low_sample_folds} | {deployment_reason} |".format(
                strategy_name=row["strategy_name"],
                deployment_candidate=bool(row["deployment_candidate"]),
                median_test_sharpe=_format_metric(row["median_test_sharpe"]),
                min_test_sharpe=_format_metric(row["min_test_sharpe"]),
                max_test_sharpe=_format_metric(row["max_test_sharpe"]),
                combine_pass_folds=int(row["combine_pass_folds"]),
                low_sample_folds=_format_text(row.get("low_sample_folds", "")),
                deployment_reason=row["deployment_reason"],
            )
        )

    for row in decisions.to_dict(orient="records"):
        strategy_name = row["strategy_name"]
        eval_frame = _fold_metric_records(strategy_name, artifact_dir=artifact_path)
        per_fold = eval_frame.loc[eval_frame["fold"].astype(str) != "summary"].copy()
        lines.extend(
            [
                "",
                f"## {strategy_name}",
                "",
                "| Fold | Test Sharpe | Combine Pass Rate | Trade Count | Sample Flag | Runtime Seconds |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for fold_row in per_fold.to_dict(orient="records"):
            trade_count = int(float(fold_row.get("trade_count", 0))) if pd.notna(fold_row.get("trade_count", 0)) else 0
            sample_flag = _format_text(fold_row.get("sample_flag", ""))
            if trade_count < 30 and "low sample" not in sample_flag.lower():
                sample_flag = "low sample"
            lines.append(
                "| {fold} | {test_sharpe} | {combine_pass_rate} | {trade_count} | {sample_flag} | {runtime_seconds} |".format(
                    fold=fold_row.get("fold", ""),
                    test_sharpe=_format_metric(fold_row.get("test_sharpe")),
                    combine_pass_rate=_format_metric(fold_row.get("combine_pass_rate")),
                    trade_count=trade_count,
                    sample_flag=sample_flag,
                    runtime_seconds=_format_metric(fold_row.get("runtime_seconds"), digits=2),
                )
            )
        lines.append(
            "| summary | median {median} | min {min_sharpe} / max {max_sharpe} |  | aggregated summary |  |".format(
                median=_format_metric(row["median_test_sharpe"]),
                min_sharpe=_format_metric(row["min_test_sharpe"]),
                max_sharpe=_format_metric(row["max_test_sharpe"]),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Apply deployment gates to evaluation artifacts.")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy name to evaluate; repeatable.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--objective",
        choices=(OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL),
        default=OBJECTIVE_META_LABEL,
    )
    parser.add_argument(
        "--gate-version",
        choices=("3c", "3d", "3e"),
        default="3e",
        help="Which deployment gate to apply (default 3e bootstrap CI gate).",
    )
    parser.add_argument(
        "--legacy-gate",
        action="store_true",
        help="Shortcut for --gate-version=3c (legacy per-fold-Sharpe gate).",
    )
    parser.add_argument(
        "--no-multiple-testing-filter",
        action="store_true",
        help="Disable the 3E recommended p50 Sharpe >= 0.5 multiple-testing filter.",
    )
    args = parser.parse_args(argv)

    gate_version = "3c" if args.legacy_gate else args.gate_version
    if gate_version in ("3c", "3d"):
        decisions = build_deployment_decisions(
            strategy_names=args.strategies, artifact_dir=args.artifact_dir
        )
        if args.objective == OBJECTIVE_META_LABEL:
            from ml.evaluate import write_meta_label_final_report

            report_path = write_meta_label_final_report(
                strategy_names=args.strategies, artifact_dir=args.artifact_dir
            )
        else:
            report_path = write_agent3c_final_report(decisions, artifact_dir=args.artifact_dir)
        print(decisions.to_string(index=False))
        print(f"report_path={report_path}")
        return 0

    # gate_version == "3e"
    require_p50 = None if args.no_multiple_testing_filter else 0.5
    decisions = build_bootstrap_deployment_decisions(
        strategy_names=args.strategies,
        artifact_dir=args.artifact_dir,
        require_p50_sharpe_above=require_p50,
    )
    print(decisions.to_string(index=False))
    print(f"artifact_dir={args.artifact_dir}")
    return 0


__all__ = [
    "DeploymentDecision",
    "ExpressFundedRules",
    "PayoutPath",
    "build_bootstrap_deployment_decisions",
    "build_deployment_decisions",
    "evaluate_bootstrap_deployment_gate",
    "evaluate_deployment_gate",
    "payout_adjusted_survival_score",
    "simulate_both_express_paths",
    "simulate_express_funded",
    "simulate_funded_after_combine",
    "write_agent3c_final_report",
]


if __name__ == "__main__":
    raise SystemExit(main())

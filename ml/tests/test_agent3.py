"""Agent 3 funded-sim, HPO, and export contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ml.export_onnx import build_model_config, load_checkpoint, load_scaler_payload, verify_onnx_matches_pytorch
from ml.funded_sim import (
    ExpressFundedRules,
    payout_adjusted_survival_score,
    simulate_express_funded,
    simulate_funded_after_combine,
)
from ml.topstep_risk import TopStepRiskManager
from ml import hyperparam_search
from ml.hyperparam_search import (
    hpo_trial_count_for_runtime,
    build_hpo_manifest,
    run_all_strategy_hpo,
    run_strategy_hpo,
    sample_search_configs,
)


def _winning_days_frame(day_count: int, atr: float = 20.0, confidence: float = 0.70) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    index: list[pd.Timestamp] = []
    target_price = 100.0 + (atr * 1.5)
    for day_idx in range(day_count):
        day = pd.Timestamp("2025-01-02", tz="America/New_York") + pd.Timedelta(days=day_idx)
        entry_ts = day + pd.Timedelta(hours=9, minutes=30)
        exit_ts = day + pd.Timedelta(hours=9, minutes=35)
        index.extend([entry_ts, exit_ts])
        rows.extend(
            [
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "atr": atr,
                    "orb_wick_signal": 1,
                    "prediction": 0,
                    "confidence": confidence,
                },
                {
                    "open": target_price,
                    "high": target_price,
                    "low": target_price,
                    "close": target_price,
                    "atr": atr,
                    "orb_wick_signal": 0,
                    "prediction": 2,
                    "confidence": 0.0,
                },
            ]
    )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="datetime"))


def _risk_manager_for_rules(rules: ExpressFundedRules) -> TopStepRiskManager:
    return TopStepRiskManager(
        account_size=rules.starting_balance,
        max_trailing_dd=rules.max_loss_limit,
        daily_loss_limit=rules.daily_loss_limit,
        max_contracts=rules.max_contracts,
        point_value=rules.point_value,
        commission_per_rt=rules.commission_per_rt,
    )


def _account_floor_threshold_price(
    entry: float,
    account_floor: float,
    direction: int,
    contracts: int,
    rules: ExpressFundedRules,
) -> float:
    dollars_per_point = contracts * rules.point_value
    commission = rules.commission_per_rt * contracts
    threshold_net_pnl = account_floor - rules.starting_balance
    return entry + direction * ((threshold_net_pnl + commission) / dollars_per_point)


def _daily_loss_only_extreme(
    entry: float,
    direction: int,
    atr: float,
    confidence: float,
    rules: ExpressFundedRules,
) -> tuple[float, int]:
    contracts = _risk_manager_for_rules(rules).position_size(atr * rules.stop_atr_mult, confidence)
    daily_loss_price = _account_floor_threshold_price(
        entry,
        rules.starting_balance - rules.daily_loss_limit,
        direction,
        contracts,
        rules,
    )
    max_loss_price = _account_floor_threshold_price(
        entry,
        rules.starting_balance - rules.max_loss_limit,
        direction,
        contracts,
        rules,
    )
    return (daily_loss_price + max_loss_price) / 2.0, contracts


def test_express_funded_standard_path_requests_max_safe_payout_and_locks_mll() -> None:
    result = simulate_express_funded(
        _winning_days_frame(5, atr=50.0),
        strategy_name="orb_wick",
        payout_path="standard",
        signal_column="orb_wick_signal",
    )

    assert result["payout_count"] == 1
    assert result["gross_payouts"] == pytest.approx(486.0)
    assert result["trader_payouts"] == pytest.approx(437.4)
    assert result["mll_floor"] == 0.0
    assert result["ending_balance"] == pytest.approx(1_000.0)
    assert set(result["trades"]["contracts"]) == {2}


def test_express_funded_consistency_path_can_request_payout() -> None:
    result = simulate_express_funded(
        _winning_days_frame(3, atr=150.0),
        strategy_name="orb_wick",
        payout_path="consistency",
        signal_column="orb_wick_signal",
    )

    assert result["payout_count"] == 1
    assert result["gross_payouts"] > 0.0
    assert not result["payouts"].empty
    assert result["payouts"]["payout_path"].iloc[0] == "consistency"


def test_express_funded_payout_window_resets_after_payout() -> None:
    result = simulate_express_funded(
        _winning_days_frame(10, atr=50.0),
        strategy_name="orb_wick",
        payout_path="standard",
        signal_column="orb_wick_signal",
    )

    assert result["payout_count"] == 2
    assert len(result["payouts"]) == 2
    assert result["payouts"]["gross_payout"].gt(0).all()


def test_express_funded_consistency_uses_total_net_profit_not_positive_day_sum() -> None:
    frame = _winning_days_frame(4, atr=100.0)
    frame.loc[frame.index[3], ["open", "high", "low", "close"]] = -125.0

    result = simulate_express_funded(
        frame,
        strategy_name="orb_wick",
        payout_path="consistency",
        signal_column="orb_wick_signal",
    )

    assert result["payout_count"] == 0
    assert not result["ledger"]["payout_eligible"].any()


def test_express_funded_enforces_minimum_payout_request() -> None:
    frame = _winning_days_frame(5, atr=35.0, confidence=0.60)

    result = simulate_express_funded(
        frame,
        strategy_name="orb_wick",
        payout_path="standard",
        signal_column="orb_wick_signal",
    )

    assert result["payout_count"] == 0
    assert result["gross_payouts"] == 0.0


def test_express_funded_uses_50_contract_session_ceiling_and_dynamic_trade_size() -> None:
    result = simulate_express_funded(
        _winning_days_frame(4, atr=100.0),
        strategy_name="orb_wick",
        payout_path="consistency",
        signal_column="orb_wick_signal",
        rules=ExpressFundedRules(payout_buffer=10_000.0),
    )

    assert result["ledger"]["session_contracts"].tolist() == [50, 50, 50, 50]
    assert set(result["trades"]["contracts"]) == {1}


def test_express_funded_daily_loss_and_max_loss_failure() -> None:
    index = pd.DatetimeIndex(
        ["2025-01-02 09:30", "2025-01-02 09:35"],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 1300.0],
            "high": [101.0, 1300.0],
            "low": [99.0, 1300.0],
            "close": [100.0, 1300.0],
            "atr": [400.0, 400.0],
            "orb_wick_signal": [-1, 0],
            "prediction": [1, 2],
        },
        index=index,
    )

    result = simulate_express_funded(frame, strategy_name="orb_wick", signal_column="orb_wick_signal")

    assert result["status"] == "failed"
    assert result["failure_reason"] == "max_loss_limit"
    assert result["ledger"]["dll_breached"].iloc[0]


def test_express_funded_intratrade_daily_loss_liquidates_long_before_same_bar_target() -> None:
    rules = ExpressFundedRules()
    confidence = 0.60
    loss_low, contracts = _daily_loss_only_extreme(
        entry=100.0,
        direction=1,
        atr=10.0,
        confidence=confidence,
        rules=rules,
    )
    index = pd.DatetimeIndex(
        ["2025-01-02 09:30", "2025-01-02 09:35"],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 200.0],
            "low": [99.0, loss_low],
            "close": [100.0, 200.0],
            "atr": [10.0, 10.0],
            "orb_wick_signal": [1, 0],
            "prediction": [0, 2],
            "confidence": [confidence, 0.0],
        },
        index=index,
    )

    result = simulate_express_funded(
        frame,
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
        rules=rules,
    )

    assert result["active"]
    assert result["ledger"]["dll_breached"].iloc[0]
    assert result["trades"]["exit_reason"].iloc[0] == "daily_loss_limit"
    assert result["trades"]["contracts"].iloc[0] == contracts
    assert result["ending_balance"] == pytest.approx(-1_000.0)


def test_express_funded_intratrade_max_loss_liquidates_short_before_same_bar_target() -> None:
    index = pd.DatetimeIndex(
        ["2025-01-02 09:30", "2025-01-02 09:35"],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 700.0],
            "low": [99.0, 0.0],
            "close": [100.0, 0.0],
            "atr": [10.0, 10.0],
            "orb_wick_signal": [-1, 0],
            "prediction": [1, 2],
        },
        index=index,
    )

    result = simulate_express_funded(frame, strategy_name="orb_wick", signal_column="orb_wick_signal")

    assert result["status"] == "failed"
    assert result["failure_reason"] == "max_loss_limit"
    assert result["trades"]["exit_reason"].iloc[0] == "max_loss_limit"
    assert result["ending_balance"] == pytest.approx(-2_000.0)


def test_express_funded_intratrade_daily_loss_liquidates_short_before_same_bar_target() -> None:
    rules = ExpressFundedRules()
    confidence = 0.60
    loss_high, contracts = _daily_loss_only_extreme(
        entry=100.0,
        direction=-1,
        atr=10.0,
        confidence=confidence,
        rules=rules,
    )
    index = pd.DatetimeIndex(
        ["2025-01-02 09:30", "2025-01-02 09:35"],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, loss_high],
            "low": [99.0, 0.0],
            "close": [100.0, 0.0],
            "atr": [10.0, 10.0],
            "orb_wick_signal": [-1, 0],
            "prediction": [1, 2],
            "confidence": [confidence, 0.0],
        },
        index=index,
    )

    result = simulate_express_funded(
        frame,
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
        rules=rules,
    )

    assert result["active"]
    assert result["ledger"]["dll_breached"].iloc[0]
    assert result["trades"]["exit_reason"].iloc[0] == "daily_loss_limit"
    assert result["trades"]["contracts"].iloc[0] == contracts
    assert result["ending_balance"] == pytest.approx(-1_000.0)


def test_express_funded_intratrade_max_loss_liquidates_long_before_same_bar_target() -> None:
    index = pd.DatetimeIndex(
        ["2025-01-02 09:30", "2025-01-02 09:35"],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 200.0],
            "low": [99.0, -700.0],
            "close": [100.0, 200.0],
            "atr": [10.0, 10.0],
            "orb_wick_signal": [1, 0],
            "prediction": [0, 2],
        },
        index=index,
    )

    result = simulate_express_funded(frame, strategy_name="orb_wick", signal_column="orb_wick_signal")

    assert result["status"] == "failed"
    assert result["failure_reason"] == "max_loss_limit"
    assert result["trades"]["exit_reason"].iloc[0] == "max_loss_limit"
    assert result["ending_balance"] == pytest.approx(-2_000.0)


def test_express_funded_reports_insufficient_post_pass_data() -> None:
    result = simulate_express_funded(
        _winning_days_frame(1),
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
        start_after="2026-01-01",
    )

    assert result["status"] == "insufficient_post_pass_data"
    assert result["failure_reason"] == "insufficient_post_pass_data"


def test_payout_adjusted_survival_score_rewards_payouts_and_survival() -> None:
    funded = simulate_express_funded(
        _winning_days_frame(5),
        strategy_name="orb_wick",
        payout_path="standard",
        signal_column="orb_wick_signal",
    )
    failed = {"trader_payouts": 0.0, "ending_balance": -2_100.0, "active": False, "trade_count": 1}

    assert payout_adjusted_survival_score(funded) > payout_adjusted_survival_score(failed)


def test_funded_simulation_starts_only_after_combine_pass_day() -> None:
    frame = _winning_days_frame(8)
    no_pass = simulate_funded_after_combine(
        frame,
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
        combine_result={"pass_day": None},
    )
    after_pass = simulate_funded_after_combine(
        frame,
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
        combine_result={"pass_day": frame.index[1].normalize()},
    )

    assert no_pass["status"] == "combine_not_passed"
    assert after_pass["status"] == "funded_simulated"
    assert after_pass["best_path"] in {"standard", "consistency"}


def test_hpo_manifest_is_validation_only_and_reproducible() -> None:
    first = build_hpo_manifest(["ttm"], n_trials=3, seed=7)
    second = build_hpo_manifest(["ttm"], n_trials=3, seed=7)

    assert first == second
    assert first["selection_metric"] == "validation_sharpe"
    assert not first["test_touched_during_search"]
    assert len(first["trial_configs"]) == 3
    assert sample_search_configs(n_trials=3, seed=7) == first["trial_configs"]


def test_hpo_trial_count_scales_from_timing_benchmark() -> None:
    assert hpo_trial_count_for_runtime(60.0) == 20
    assert hpo_trial_count_for_runtime(900.0) == 10
    assert hpo_trial_count_for_runtime(2_000.0) == 5


def test_agent3_hpo_stops_when_estimate_exceeds_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Job:
        def __init__(self, strategy_name: str) -> None:
            self.strategy_name = strategy_name
            self.timeframe = "5min"

    benchmark = pd.DataFrame({"runtime_seconds": [2_000.0]})
    monkeypatch.setattr(hyperparam_search, "build_training_jobs", lambda _names: [_Job("ttm"), _Job("connors")])
    monkeypatch.setattr(hyperparam_search, "benchmark_hpo_runtime", lambda *_args, **_kwargs: benchmark)

    result = run_all_strategy_hpo(["ttm", "connors"], output_dir=None, max_estimated_seconds=1_000.0)

    assert result["status"] == "aborted_estimate_over_limit"
    assert result["n_trials"] == 5
    assert result["estimated_seconds"] == 20_000.0


def test_hpo_real_path_uses_validation_split_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Job:
        strategy_name = "ttm"
        timeframe = "5min"
        signal_column = "model_2_signal"
        signal_cols = ("ttm_signal",)

    idx = pd.date_range("2025-01-02", periods=3, tz="America/New_York")
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "ttm_signal": [1, 1, 1], "label": [0, 1, 2]}, index=idx)
    train_frame = frame.iloc[[0]]
    val_frame = frame.iloc[[1]]
    poison_test_frame = frame.iloc[[2]]
    touched = {"test": False}

    def fake_build_temporal_splits(
        _frame: pd.DataFrame,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {"fold_2": {"train": train_frame, "val": val_frame, "test": poison_test_frame}}

    def fake_train_one_fold(**kwargs: object) -> dict[str, object]:
        assert kwargs["train_frame"] is train_frame
        assert kwargs["val_frame"] is val_frame
        return {"model": object(), "scaler": object()}

    def fake_evaluate_on_validation(**kwargs: object) -> dict[str, float | int]:
        assert kwargs["val_frame"] is val_frame
        touched["test"] = kwargs["val_frame"] is poison_test_frame
        return {
            "test_sharpe": 1.0,
            "test_f1": 0.5,
            "test_profit_factor": 2.0,
            "trade_count": 7,
        }

    monkeypatch.setattr(hyperparam_search, "build_training_jobs", lambda _names: [_Job()])
    monkeypatch.setattr(hyperparam_search, "_load_strategy_frame", lambda _job, objective="three_class": frame)
    monkeypatch.setattr(hyperparam_search, "build_temporal_splits", fake_build_temporal_splits)
    monkeypatch.setattr(hyperparam_search, "feature_columns_from_frame", lambda _frame: ["f1"])
    monkeypatch.setattr(hyperparam_search, "load_data", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(hyperparam_search, "_train_one_fold", fake_train_one_fold)
    monkeypatch.setattr(hyperparam_search, "_evaluate_on_validation", fake_evaluate_on_validation)

    result = run_strategy_hpo("ttm", n_trials=1, fold_name="fold_2", dry_run=False, output_dir=None)

    assert not touched["test"]
    assert result["selection_metric"].iloc[0] == "validation_sharpe"
    assert not bool(result["test_touched_during_search"].iloc[0])


def test_export_model_config_and_scaler_contract_exclude_targets() -> None:
    checkpoint = load_checkpoint("ttm")
    scaler = load_scaler_payload("ttm")
    config = build_model_config("ttm", checkpoint, deployment_candidate=True)

    assert config["n_features"] == 35
    assert config["seq_len"] == 30
    assert config["opset_version"] == 12
    assert config["deployment_candidate"]
    assert "future_return" not in checkpoint["feature_columns"]
    assert "label" not in checkpoint["feature_columns"]
    assert len(scaler["mean_"]) == 35
    assert len(scaler["scale_"]) == 35


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime is installed under Python 3.13")
def test_onnx_runtime_available_for_agent3_export_verification() -> None:
    assert importlib.util.find_spec("onnx") is not None
    assert importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime is installed under Python 3.13")
def test_research_ttm_onnx_matches_pytorch_if_exported() -> None:
    onnx_path = Path("ml/artifacts/model_ttm.onnx")
    if not onnx_path.exists():
        pytest.skip("research TTM ONNX has not been exported in this environment")

    verification = verify_onnx_matches_pytorch("ttm")
    assert verification["max_diff"] < 1e-5

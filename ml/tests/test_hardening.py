"""Agent 2.5 hardening tests for leakage, risk, evaluation, and artifacts."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from ml.dataset_builder import compute_time_features, load_data
from ml.evaluate import simulate_trading
from ml.signal_generators import (
    connors_rsi2,
    ifvg_combined,
    orb_initial_balance,
    orb_volatility_filtered,
    orb_volume_adaptive,
    orb_wick_rejection,
    ttm_squeeze,
)
from ml.topstep_risk import TopStepRiskManager
from ml.train import ARTIFACT_DIR, build_training_jobs


ROOT_DIR = Path(__file__).resolve().parents[2]
NEWS_DATES_PATH = ROOT_DIR / "ml" / "data" / "news_dates.csv"


@lru_cache(maxsize=None)
def _session_data(timeframe: str) -> pd.DataFrame:
    return load_data("mnq", timeframe, session_only=True)


def _ifvg_base(df: pd.DataFrame) -> pd.Series:
    base, _ = ifvg_combined(df, timeframe_minutes=5)
    return base


def _ifvg_open(df: pd.DataFrame) -> pd.Series:
    _, open_variant = ifvg_combined(df, timeframe_minutes=5)
    return open_variant


SIGNAL_GENERATORS: tuple[tuple[str, Callable[[pd.DataFrame], pd.Series]], ...] = (
    ("orb_vol", orb_volatility_filtered),
    ("orb_wick", orb_wick_rejection),
    ("orb_ib", orb_initial_balance),
    ("orb_va", orb_volume_adaptive),
    ("connors", connors_rsi2),
    ("ifvg", _ifvg_base),
    ("ifvg_open", _ifvg_open),
    ("ttm", ttm_squeeze),
)


@pytest.mark.parametrize(("strategy_name", "generator"), SIGNAL_GENERATORS)
def test_signal_generators_do_not_change_past_when_future_row_mutates(
    strategy_name: str,
    generator: Callable[[pd.DataFrame], pd.Series],
) -> None:
    df = _session_data("5min").iloc[:900].copy()
    baseline = generator(df)

    mutate_ts = df.index[650]
    mutated = df.copy()
    mutated.at[mutate_ts, "open"] = mutated.at[mutate_ts, "open"] * 1.07
    mutated.at[mutate_ts, "high"] = mutated.at[mutate_ts, "high"] * 1.10
    mutated.at[mutate_ts, "low"] = mutated.at[mutate_ts, "low"] * 0.90
    mutated.at[mutate_ts, "close"] = mutated.at[mutate_ts, "close"] * 1.05
    mutated.at[mutate_ts, "volume"] = mutated.at[mutate_ts, "volume"] * 3

    updated = generator(mutated)
    earlier = baseline.index < mutate_ts

    pd.testing.assert_series_equal(
        baseline.loc[earlier],
        updated.loc[earlier],
        check_names=False,
        obj=f"{strategy_name} signals before mutated row",
    )


@pytest.mark.parametrize(("strategy_name", "generator"), SIGNAL_GENERATORS)
def test_signal_generators_match_when_future_tail_is_truncated(
    strategy_name: str,
    generator: Callable[[pd.DataFrame], pd.Series],
) -> None:
    df = _session_data("5min").iloc[:900].copy()
    truncated = df.iloc[:-40].copy()

    full_signals = generator(df)
    truncated_signals = generator(truncated)

    pd.testing.assert_series_equal(
        full_signals.loc[truncated_signals.index],
        truncated_signals,
        check_names=False,
        obj=f"{strategy_name} full-vs-truncated signals",
    )


def test_topstep_daily_loss_limit_boundaries() -> None:
    risk = TopStepRiskManager()

    assert risk.check_intraday(49_001.0)
    assert not risk.check_intraday(49_000.0)


def test_topstep_trade_math_uses_dynamic_contracts_and_commissions() -> None:
    risk = TopStepRiskManager()

    assert risk.max_contracts == 50
    assert risk.point_value == 2
    assert risk.position_size(stop_pts=5.0, confidence=0.99) == 50

    long_pnl = risk.simulate_trade(entry=100.0, stop=90.0, target=110.0, exit_price=110.0, contracts=5)
    short_pnl = risk.simulate_trade(entry=100.0, stop=110.0, target=90.0, exit_price=90.0, contracts=5)

    assert long_pnl == pytest.approx(93.0)
    assert short_pnl == pytest.approx(93.0)


def _synthetic_eval_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:30",
            "2025-01-02 09:35",
            "2025-01-03 09:30",
            "2025-01-03 09:35",
        ],
        tz="America/New_York",
        name="datetime",
    )
    return pd.DataFrame(
        {
            "open": [100.0, 115.0, 100.0, 115.0],
            "high": [101.0, 116.0, 101.0, 116.0],
            "low": [99.0, 114.0, 99.0, 114.0],
            "close": [100.0, 115.0, 100.0, 115.0],
            "atr": [10.0, 10.0, 10.0, 10.0],
            "orb_wick_signal": [1, 0, -1, 0],
            "prediction": [0, 2, 1, 2],
            "label": [0, 2, 1, 2],
        },
        index=index,
    )


def test_simulate_trading_known_long_win_and_short_loss() -> None:
    result = simulate_trading(
        _synthetic_eval_frame(),
        strategy_name="orb_wick",
        signal_column="orb_wick_signal",
    )
    trades = result["trades"]

    assert result["trade_count"] == 2
    assert result["win_rate"] == pytest.approx(0.5)
    assert trades["pnl"].tolist() == pytest.approx([171.6, -188.4])
    assert result["ending_account"] == pytest.approx(49_983.2)
    assert not result["combine_passed"]


def test_simulate_trading_stops_after_daily_loss_limit_breach() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:30",
            "2025-01-02 09:35",
            "2025-01-02 09:40",
            "2025-01-02 09:45",
        ],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 700.0, 100.0, 700.0],
            "high": [101.0, 700.0, 101.0, 700.0],
            "low": [99.0, 700.0, 99.0, 700.0],
            "close": [100.0, 700.0, 100.0, 700.0],
            "atr": [400.0, 400.0, 400.0, 400.0],
            "orb_wick_signal": [-1, 0, -1, 0],
            "prediction": [1, 2, 1, 2],
            "label": [1, 2, 1, 2],
        },
        index=index,
    )

    result = simulate_trading(frame, strategy_name="orb_wick", signal_column="orb_wick_signal")

    assert result["trade_count"] == 1
    assert result["trades"]["pnl"].iloc[0] == pytest.approx(-1201.4)
    assert result["active"]


def test_simulate_trading_reports_consistency_failure() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-02 09:30",
            "2025-01-02 09:35",
            "2025-01-03 09:30",
            "2025-01-03 09:35",
            "2025-01-06 09:30",
            "2025-01-06 09:35",
        ],
        tz="America/New_York",
        name="datetime",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 2650.0, 100.0, 175.0, 100.0, 175.0],
            "high": [101.0, 2650.0, 101.0, 175.0, 101.0, 175.0],
            "low": [99.0, 2650.0, 99.0, 175.0, 99.0, 175.0],
            "close": [100.0, 2650.0, 100.0, 175.0, 100.0, 175.0],
            "atr": [1700.0, 1700.0, 50.0, 50.0, 50.0, 50.0],
            "orb_wick_signal": [1, 0, 1, 0, 1, 0],
            "prediction": [0, 2, 0, 2, 0, 2],
            "label": [0, 2, 0, 2, 0, 2],
        },
        index=index,
    )

    result = simulate_trading(frame, strategy_name="orb_wick", signal_column="orb_wick_signal")

    assert result["trade_count"] == 3
    assert result["ending_account"] > 55_000.0
    assert not result["consistency_ok"]
    assert not result["combine_passed"]


def test_agent2_artifact_contract_is_complete() -> None:
    required_eval_columns = {
        "fold",
        "best_epoch",
        "val_loss",
        "val_f1",
        "train_windows",
        "val_windows",
        "test_windows",
        "test_f1",
        "test_roc_auc",
        "test_accuracy",
        "test_sharpe",
        "test_profit_factor",
        "test_win_rate",
        "test_avg_r",
        "test_max_drawdown",
        "combine_pass_rate",
        "avg_days_to_pass",
        "trade_count",
        "confusion_matrix",
    }

    for job in build_training_jobs():
        scaler_path = Path(job.scaler_path)
        model_path = Path(job.model_path)
        eval_path = Path(job.eval_path)

        if not scaler_path.exists() or not model_path.exists() or not eval_path.exists():
            pytest.skip("Grouped Agent 4B artifacts are produced only after the human-run training phase.")

        assert scaler_path.exists(), f"missing scaler: {scaler_path}"
        assert model_path.exists(), f"missing checkpoint: {model_path}"
        assert eval_path.exists(), f"missing eval CSV: {eval_path}"
        assert scaler_path.stat().st_size > 0
        assert model_path.stat().st_size > 0
        assert eval_path.stat().st_size > 0

        with scaler_path.open("rb") as handle:
            scaler_payload = pickle.load(handle)
        assert len(scaler_payload["feature_columns"]) == 35
        assert len(scaler_payload["mean_"]) == 35
        assert len(scaler_payload["scale_"]) == 35

        eval_frame = pd.read_csv(eval_path)
        assert required_eval_columns.issubset(eval_frame.columns)
        assert {"fold_1", "fold_2", "summary"}.issubset(set(eval_frame["fold"].astype(str)))


def test_news_dates_calendar_is_real_and_covers_all_event_types() -> None:
    news_dates = pd.read_csv(NEWS_DATES_PATH)
    parsed_dates = pd.to_datetime(news_dates["date"], errors="raise")

    assert not news_dates["date"].isna().any()
    assert set(news_dates["event_type"]) == {"FOMC", "NFP", "CPI"}
    assert parsed_dates.min().year == 2021
    assert parsed_dates.max() >= pd.Timestamp("2026-12-01")

    for year in range(2021, 2027):
        year_events = news_dates.loc[parsed_dates.dt.year == year, "event_type"]
        assert {"FOMC", "NFP", "CPI"}.issubset(set(year_events))

    expected_dates = {
        ("2025-03-19", "FOMC"),
        ("2025-12-16", "NFP"),
        ("2025-12-18", "CPI"),
        ("2026-03-18", "FOMC"),
    }
    actual_pairs = set(zip(news_dates["date"], news_dates["event_type"], strict=False))
    assert expected_dates.issubset(actual_pairs)


def test_is_news_day_flags_curated_calendar_dates() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-03-19 09:30",
            "2025-03-20 09:30",
            "2026-03-11 09:30",
        ],
        tz="America/New_York",
        name="datetime",
    )
    features = compute_time_features(pd.DataFrame(index=index))

    assert features.loc[pd.Timestamp("2025-03-19 09:30", tz="America/New_York"), "is_news_day"] == 1
    assert features.loc[pd.Timestamp("2025-03-20 09:30", tz="America/New_York"), "is_news_day"] == 0
    assert features.loc[pd.Timestamp("2026-03-11 09:30", tz="America/New_York"), "is_news_day"] == 1

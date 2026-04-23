"""Agent 3E tests — stationary block bootstrap + 4-condition deployment gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.bootstrap import (  # noqa: E402
    PF_CAP,
    bootstrap_pass_rate,
    bootstrap_trade_metrics,
    stationary_block_bootstrap,
    _sharpe,
)
from ml.funded_sim import evaluate_bootstrap_deployment_gate  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Point estimate reproducibility
# ---------------------------------------------------------------------------
def test_stationary_bootstrap_point_estimate_matches_input() -> None:
    trades = np.array([0.5, -0.3, 0.2, -0.1, 0.7], dtype=float)
    result = stationary_block_bootstrap(
        trades, np.mean, n_resamples=1, expected_block_len=1.0, random_state=42
    )
    assert result["point"] == pytest.approx(float(np.mean(trades)))
    assert result["n_resamples"] == 1
    assert result["n_trades"] == 5


# ---------------------------------------------------------------------------
# 2. Block bootstrap preserves autocorrelation → wider CIs than iid
# ---------------------------------------------------------------------------
def _iid_bootstrap_sharpe(trades: np.ndarray, n_resamples: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(trades)
    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = trades[rng.integers(0, n, size=n)]
        out[i] = _sharpe(sample)
    return out


def test_bootstrap_preserves_autocorrelation() -> None:
    # Highly autocorrelated P&L: long positive streak followed by long negative streak.
    # Block bootstrap should produce wider Sharpe CIs than iid bootstrap because it
    # keeps the regime blocks intact.
    trades = np.concatenate([np.full(40, 0.5), np.full(40, -0.4)])
    block = stationary_block_bootstrap(
        trades, _sharpe, n_resamples=500, expected_block_len=10.0, random_state=1234
    )
    iid_samples = _iid_bootstrap_sharpe(trades, n_resamples=500, random_state=1234)
    block_width = block["p95"] - block["p05"]
    iid_width = float(np.percentile(iid_samples, 95) - np.percentile(iid_samples, 5))
    assert block_width > iid_width, (
        f"block CI width {block_width:.3f} should exceed iid CI width {iid_width:.3f} "
        "on autocorrelated data"
    )


# ---------------------------------------------------------------------------
# 3. Percentiles in correct order
# ---------------------------------------------------------------------------
def test_bootstrap_p05_le_p50_le_p95() -> None:
    rng = np.random.default_rng(7)
    trades = rng.normal(loc=0.1, scale=1.0, size=120)
    metrics = bootstrap_trade_metrics(
        trades, n_resamples=500, expected_block_len=5.0, random_state=7
    )
    for name, stats in metrics.items():
        assert stats["p05"] <= stats["p50"] + 1e-9, f"{name} p05 > p50"
        assert stats["p50"] <= stats["p95"] + 1e-9, f"{name} p50 > p95"


# ---------------------------------------------------------------------------
# 4. Reproducibility: same random_state → identical output
# ---------------------------------------------------------------------------
def test_bootstrap_reproducible() -> None:
    trades = np.array([0.3, -0.2, 0.5, -0.4, 0.7, -0.1, 0.2, -0.3, 0.6, -0.5], dtype=float)
    a = stationary_block_bootstrap(
        trades, _sharpe, n_resamples=250, expected_block_len=3.0, random_state=99
    )
    b = stationary_block_bootstrap(
        trades, _sharpe, n_resamples=250, expected_block_len=3.0, random_state=99
    )
    for key in ("point", "mean", "std", "p05", "p50", "p95"):
        assert a[key] == b[key], f"mismatch on {key}: {a[key]} vs {b[key]}"


# ---------------------------------------------------------------------------
# 5. All winning trades: no NaN / no inf, profit_factor capped
# ---------------------------------------------------------------------------
def test_bootstrap_handles_all_winning_trades() -> None:
    trades = np.full(50, 0.4, dtype=float)
    metrics = bootstrap_trade_metrics(
        trades, n_resamples=200, expected_block_len=4.0, random_state=5
    )
    pf = metrics["profit_factor"]
    wr = metrics["win_rate"]
    assert np.isfinite(pf["p05"]) and np.isfinite(pf["p50"]) and np.isfinite(pf["p95"])
    assert pf["point"] <= PF_CAP + 1e-9
    assert pf["p95"] <= PF_CAP + 1e-9
    assert wr["point"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. All losing trades
# ---------------------------------------------------------------------------
def test_bootstrap_handles_all_losing_trades() -> None:
    trades = np.full(50, -0.3, dtype=float)
    metrics = bootstrap_trade_metrics(
        trades, n_resamples=200, expected_block_len=4.0, random_state=5
    )
    pf = metrics["profit_factor"]
    wr = metrics["win_rate"]
    sharpe = metrics["sharpe"]
    assert pf["point"] == 0.0
    assert wr["point"] == 0.0
    # Constant returns have zero std → Sharpe == 0 by the degenerate-safe branch.
    assert sharpe["point"] == 0.0
    for stats in metrics.values():
        for key in ("point", "mean", "p05", "p50", "p95"):
            assert np.isfinite(stats[key]) or key == "std"


# ---------------------------------------------------------------------------
# 7. Single trade
# ---------------------------------------------------------------------------
def test_bootstrap_handles_single_trade() -> None:
    trades = np.array([0.25], dtype=float)
    result = stationary_block_bootstrap(
        trades, np.mean, n_resamples=50, expected_block_len=2.0, random_state=0
    )
    assert result["point"] == pytest.approx(0.25)
    assert result["n_trades"] == 1
    assert result["unreliable"] is True
    assert result["p05"] == pytest.approx(0.25)
    assert result["p95"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Helper for deployment-gate tests: build a bootstrap-result dict.
# ---------------------------------------------------------------------------
def _make_strategy_results(
    *,
    agg_sharpe_p05: float,
    agg_sharpe_p50: float,
    agg_pf_p50: float,
    agg_pass_p05: float,
    fold_sharpe_points: list[float],
) -> dict:
    return {
        "aggregated": {
            "sharpe": {"p05": agg_sharpe_p05, "p50": agg_sharpe_p50, "p95": agg_sharpe_p50 + 0.1, "point": agg_sharpe_p50},
            "profit_factor": {"p05": agg_pf_p50 - 0.2, "p50": agg_pf_p50, "p95": agg_pf_p50 + 0.2, "point": agg_pf_p50},
            "pass_rate": {"p05": agg_pass_p05, "p50": agg_pass_p05 + 0.1, "p95": agg_pass_p05 + 0.2, "point": agg_pass_p05},
            "n_trades": 250,
        },
        "per_fold": [
            {
                "fold": f"fold_{idx + 1}",
                "sharpe": {"point": point, "p05": point - 0.3, "p50": point, "p95": point + 0.3},
            }
            for idx, point in enumerate(fold_sharpe_points)
        ],
    }


# ---------------------------------------------------------------------------
# 8. Gate rejects negative p05 Sharpe even if point is positive
# ---------------------------------------------------------------------------
def test_deployment_gate_rejects_negative_p05_sharpe() -> None:
    results = _make_strategy_results(
        agg_sharpe_p05=-0.05,
        agg_sharpe_p50=1.2,
        agg_pf_p50=1.5,
        agg_pass_p05=0.60,
        fold_sharpe_points=[0.8, 0.9, 1.0, 1.1, 1.2],
    )
    decision = evaluate_bootstrap_deployment_gate(results)
    assert decision.approved is False
    assert "p05 Sharpe" in decision.reason


# ---------------------------------------------------------------------------
# 9. Gate rejects when one fold's point Sharpe is <= -0.3
# ---------------------------------------------------------------------------
def test_deployment_gate_rejects_single_bad_fold() -> None:
    results = _make_strategy_results(
        agg_sharpe_p05=0.2,
        agg_sharpe_p50=0.9,
        agg_pf_p50=1.5,
        agg_pass_p05=0.50,
        fold_sharpe_points=[0.8, 0.9, 1.0, 1.1, -0.5],  # one terrible fold
    )
    decision = evaluate_bootstrap_deployment_gate(results)
    assert decision.approved is False
    assert "per-fold point Sharpe" in decision.reason
    assert "fold_5" in decision.reason


# ---------------------------------------------------------------------------
# 10. Gate accepts when all four conditions + multiple-testing filter met
# ---------------------------------------------------------------------------
def test_deployment_gate_accepts_all_conditions_met() -> None:
    results = _make_strategy_results(
        agg_sharpe_p05=0.25,
        agg_sharpe_p50=0.75,  # >= 0.5 multiple-testing filter
        agg_pf_p50=1.35,  # >= 1.2
        agg_pass_p05=0.40,  # >= 0.30
        fold_sharpe_points=[0.3, 0.5, 0.8, 0.9, 1.1],  # all > -0.3
    )
    decision = evaluate_bootstrap_deployment_gate(results)
    assert decision.approved is True, f"expected approval, got {decision.reason}"
    assert "passed" in decision.reason

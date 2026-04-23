"""Stationary block bootstrap for trading metrics (Agent 3E).

Implements Politis & Romano (1994) stationary block bootstrap for autocorrelated
trade series plus a path-dependent Combine pass-rate bootstrap that resamples
day sequences and re-runs ``TopStepRiskManager``.

Caps profit factor and Calmar at 10.0 to avoid infinite values from degenerate
resamples (zero-loss or zero-drawdown streams). Flags small-sample results
(``n_trades < 30`` or ``n_days < 30``) as ``unreliable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_BLOCK_LEN = 5.0
DEFAULT_RANDOM_STATE = 42
RELIABILITY_THRESHOLD = 30
PF_CAP = 10.0
CALMAR_CAP = 10.0


# ---------------------------------------------------------------------------
# Core stationary block bootstrap
# ---------------------------------------------------------------------------
def _resample_stationary_block(
    series: np.ndarray,
    rng: np.random.Generator,
    expected_block_len: float,
) -> np.ndarray:
    """Return one stationary-block resample of the same length as ``series``.

    Block lengths are drawn from ``Geometric(1/expected_block_len)``. Block
    starts wrap around the array so every element has equal resample probability.
    """
    n = len(series)
    if n == 0:
        return series.copy()
    p = 1.0 / max(float(expected_block_len), 1e-9)
    sample = np.empty(n, dtype=series.dtype)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        block_len = max(int(rng.geometric(p)), 1)
        take = min(block_len, n - pos)
        # Copy ``take`` consecutive (circular) elements starting at ``start``.
        if start + take <= n:
            sample[pos : pos + take] = series[start : start + take]
        else:
            first = n - start
            sample[pos : pos + first] = series[start:n]
            sample[pos + first : pos + take] = series[: take - first]
        pos += take
    return sample


def stationary_block_bootstrap(
    trades: Sequence[float] | np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    expected_block_len: float = DEFAULT_BLOCK_LEN,
    random_state: int | None = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Stationary block bootstrap of ``metric_fn`` on ``trades``.

    Returns a dict with ``point``, ``mean``, ``std``, ``p05``, ``p50``, ``p95``,
    ``n_resamples``, ``n_trades`` and an ``unreliable`` flag (True when
    ``n_trades < 30``).
    """
    arr = np.asarray(trades, dtype=float)
    n = len(arr)
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if expected_block_len <= 0:
        raise ValueError("expected_block_len must be positive")

    if n == 0:
        return {
            "point": float("nan"),
            "mean": float("nan"),
            "std": 0.0,
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "n_resamples": int(n_resamples),
            "n_trades": 0,
            "unreliable": True,
        }

    rng = np.random.default_rng(random_state)
    point_estimate = float(metric_fn(arr))
    resampled = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        sample = _resample_stationary_block(arr, rng, expected_block_len)
        resampled[i] = float(metric_fn(sample))

    # Use nan-aware percentile/mean/std so a handful of degenerate resamples
    # (PF cap hit etc.) do not propagate and kill every summary stat.
    return {
        "point": point_estimate,
        "mean": float(np.nanmean(resampled)) if np.isfinite(resampled).any() else float("nan"),
        "std": float(np.nanstd(resampled)) if np.isfinite(resampled).any() else 0.0,
        "p05": float(np.nanpercentile(resampled, 5)) if np.isfinite(resampled).any() else float("nan"),
        "p50": float(np.nanpercentile(resampled, 50)) if np.isfinite(resampled).any() else float("nan"),
        "p95": float(np.nanpercentile(resampled, 95)) if np.isfinite(resampled).any() else float("nan"),
        "n_resamples": int(n_resamples),
        "n_trades": int(n),
        "unreliable": bool(n < RELIABILITY_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# Metric helpers (all capped / degenerate-safe)
# ---------------------------------------------------------------------------
def _sharpe(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    # Exact check for constant arrays: floating-point std() of a constant
    # series can be ~1e-17 (not 0) because its mean is not exactly the
    # element value, which blows Sharpe up to ~1e16. Peak-to-peak is exact.
    if float(np.ptp(trades)) == 0.0:
        return 0.0
    std = float(trades.std())
    if std == 0.0:
        return 0.0
    return float(np.sqrt(252.0) * trades.mean() / std)


def _calmar(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    equity = np.cumsum(trades)
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0
    total = float(trades.sum())
    if max_dd == 0.0:
        return CALMAR_CAP if total > 0 else 0.0
    calmar = total / max_dd
    if calmar > CALMAR_CAP:
        return CALMAR_CAP
    if calmar < -CALMAR_CAP:
        return -CALMAR_CAP
    return float(calmar)


def _profit_factor(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    wins = float(trades[trades > 0].sum())
    losses = float(-trades[trades < 0].sum())
    if losses == 0.0:
        return PF_CAP if wins > 0 else 0.0
    return float(min(wins / losses, PF_CAP))


def _win_rate(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    return float((trades > 0).mean())


def _avg_r(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    return float(trades.mean())


# ---------------------------------------------------------------------------
# Public: bootstrap a suite of trade metrics
# ---------------------------------------------------------------------------
def bootstrap_trade_metrics(
    trade_pnl: Sequence[float] | np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    expected_block_len: float = DEFAULT_BLOCK_LEN,
    random_state: int | None = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, Any]]:
    """Bootstrap Sharpe / Calmar / profit-factor / win-rate / avg-r on a trade stream."""
    kwargs = dict(
        n_resamples=n_resamples,
        expected_block_len=expected_block_len,
        random_state=random_state,
    )
    return {
        "sharpe": stationary_block_bootstrap(trade_pnl, _sharpe, **kwargs),
        "calmar": stationary_block_bootstrap(trade_pnl, _calmar, **kwargs),
        "profit_factor": stationary_block_bootstrap(trade_pnl, _profit_factor, **kwargs),
        "win_rate": stationary_block_bootstrap(trade_pnl, _win_rate, **kwargs),
        "avg_r": stationary_block_bootstrap(trade_pnl, _avg_r, **kwargs),
    }


# ---------------------------------------------------------------------------
# Pass-rate bootstrap: resample day sequences and replay TopStepRiskManager
# ---------------------------------------------------------------------------
def _run_topstep_sequence(day_pnls: np.ndarray, manager_factory: Callable[[], Any]) -> float:
    manager = manager_factory()
    for day_pnl in day_pnls:
        if not manager.active:
            break
        new_balance = manager.account + float(day_pnl)
        manager.update_eod(new_balance, float(day_pnl))
        if manager.is_passed():
            return 1.0
    return 1.0 if manager.is_passed() else 0.0


def bootstrap_pass_rate(
    daily_pnls: Sequence[float] | np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    expected_block_len: float = DEFAULT_BLOCK_LEN,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    manager_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap the TopStep Combine pass-rate by resampling day sequences.

    Each resample is a re-ordered block of consecutive days. For every resample
    we instantiate a fresh ``TopStepRiskManager`` and replay the daily P&L
    sequence, returning 1.0 if ``is_passed()`` ever fires, else 0.0.
    """
    if manager_factory is None:
        from ml.topstep_risk import TopStepRiskManager as _Mgr

        manager_factory = _Mgr

    arr = np.asarray(daily_pnls, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "point": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "n_resamples": int(n_resamples),
            "n_days": 0,
            "unreliable": True,
        }

    point = _run_topstep_sequence(arr, manager_factory)
    rng = np.random.default_rng(random_state)
    resampled = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        sample = _resample_stationary_block(arr, rng, expected_block_len)
        resampled[i] = _run_topstep_sequence(sample, manager_factory)

    return {
        "point": float(point),
        "mean": float(resampled.mean()),
        "std": float(resampled.std()),
        "p05": float(np.percentile(resampled, 5)),
        "p50": float(np.percentile(resampled, 50)),
        "p95": float(np.percentile(resampled, 95)),
        "n_resamples": int(n_resamples),
        "n_days": int(n),
        "unreliable": bool(n < RELIABILITY_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# Helper: load per-fold trade + daily-pnl CSVs written by train.py
# ---------------------------------------------------------------------------
def load_per_fold_series(
    strategy_name: str,
    artifact_dir: str | Path,
    fold_names: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{fold_name: {"trade_pnl": arr, "daily_pnl": arr}}``.

    Looks for ``fold_trades_{strategy}_{fold}.csv`` and
    ``fold_daily_pnls_{strategy}_{fold}.csv`` produced by ``train._train_one_fold``.
    Returns an empty dict if no per-fold artifacts are present, which lets the
    caller fall back to aggregated-stream bootstrap.
    """
    import pandas as pd

    root = Path(artifact_dir)
    default_folds = fold_names or [f"fold_{i}" for i in range(1, 6)]
    out: dict[str, dict[str, np.ndarray]] = {}
    for fold in default_folds:
        trade_path = root / f"fold_trades_{strategy_name}_{fold}.csv"
        daily_path = root / f"fold_daily_pnls_{strategy_name}_{fold}.csv"
        if not trade_path.exists() or not daily_path.exists():
            continue
        trades = pd.read_csv(trade_path)
        daily = pd.read_csv(daily_path)
        trade_pnl = pd.to_numeric(trades.get("pnl", trades.get("r_multiple")), errors="coerce").dropna().to_numpy()
        daily_pnl = pd.to_numeric(daily.get("pnl", daily.get("day_pnl")), errors="coerce").dropna().to_numpy()
        out[fold] = {"trade_pnl": trade_pnl, "daily_pnl": daily_pnl}
    return out


__all__ = [
    "PF_CAP",
    "CALMAR_CAP",
    "RELIABILITY_THRESHOLD",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_BLOCK_LEN",
    "DEFAULT_RANDOM_STATE",
    "stationary_block_bootstrap",
    "bootstrap_trade_metrics",
    "bootstrap_pass_rate",
    "load_per_fold_series",
]

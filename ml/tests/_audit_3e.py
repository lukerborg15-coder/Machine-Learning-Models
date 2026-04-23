"""Agent 3E audit — 10 adversarial checks. Run as a script, not via pytest.

Usage:
    cd C:\\Users\\Luker\\strategyLabbrain
    python ml/tests/_audit_3e.py

Exits 0 if all checks PASS, 1 if any FAIL. Writes the full check log to
``ml/artifacts/agent3e_audit_log.txt`` for later inclusion in AGENT3E_AUDIT.md.

Assumes the full 3E artifact chain has already been generated:
    1. `python ml/train.py --retrain-from-hpo --objective meta_label`
       produces `fold_trades_{strategy}_{fold}.csv` +
       `fold_daily_pnls_{strategy}_{fold}.csv`.
    2. `python ml/evaluate.py --bootstrap` produces
       `eval_{strategy}_bootstrap.json` per strategy.
    3. `python ml/funded_sim.py --gate-version=3e` produces
       `agent3e_deployment_decisions.csv`.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.bootstrap import (  # noqa: E402
    CALMAR_CAP,
    PF_CAP,
    RELIABILITY_THRESHOLD,
    _calmar,
    _profit_factor,
    _sharpe,
    bootstrap_pass_rate,
    bootstrap_trade_metrics,
    stationary_block_bootstrap,
)
from ml.funded_sim import (  # noqa: E402
    evaluate_bootstrap_deployment_gate,
)

ARTIFACT_DIR = ROOT_DIR / "ml" / "artifacts"

RESULTS: list[tuple[int, str, bool, str]] = []
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


def record(idx: int, name: str, passed: bool, evidence: str) -> None:
    RESULTS.append((idx, name, passed, evidence))
    tag = "PASS" if passed else "FAIL"
    log(f"[CHECK {idx}] {tag} - {name}")
    log(f"         {evidence}")


# ---------------------------------------------------------------------------
# Check 1 - Block bootstrap widens CIs relative to iid bootstrap on
# autocorrelated data (core correctness of Politis & Romano).
# ---------------------------------------------------------------------------
def check_autocorrelation_preservation() -> None:
    try:
        # 80 trades: 40 positive streak then 40 negative streak.
        trades = np.concatenate([np.full(40, 0.5), np.full(40, -0.4)])
        block = stationary_block_bootstrap(
            trades, _sharpe, n_resamples=1000, expected_block_len=10.0, random_state=2024
        )
        rng = np.random.default_rng(2024)
        iid = np.empty(1000, dtype=float)
        for i in range(1000):
            sample = trades[rng.integers(0, len(trades), size=len(trades))]
            iid[i] = _sharpe(sample)
        block_w = block["p95"] - block["p05"]
        iid_w = float(np.percentile(iid, 95) - np.percentile(iid, 5))
        passed = block_w > iid_w
        record(
            1,
            "Block bootstrap widens CIs vs iid on autocorrelated series",
            passed,
            f"block p95-p05 = {block_w:.3f}, iid p95-p05 = {iid_w:.3f}",
        )
    except Exception as exc:
        record(1, "Autocorrelation preservation", False, f"exception: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Check 2 - Determinism: same random_state -> identical output on real data.
# ---------------------------------------------------------------------------
def check_reproducibility() -> None:
    try:
        rng = np.random.default_rng(0)
        trades = rng.normal(loc=0.05, scale=1.0, size=200)
        a = stationary_block_bootstrap(
            trades, _sharpe, n_resamples=500, expected_block_len=5.0, random_state=123
        )
        b = stationary_block_bootstrap(
            trades, _sharpe, n_resamples=500, expected_block_len=5.0, random_state=123
        )
        keys = ("point", "mean", "std", "p05", "p50", "p95")
        mismatches = [k for k in keys if a[k] != b[k]]
        record(
            2,
            "Reproducibility: random_state=123 repeat is bit-identical",
            not mismatches,
            f"mismatched keys: {mismatches}" if mismatches else "all 6 summary keys identical",
        )
    except Exception as exc:
        record(2, "Reproducibility", False, f"exception: {exc}")


# ---------------------------------------------------------------------------
# Check 3 - Percentile ordering p05 <= p50 <= p95 across ALL bootstrap
# artifacts on disk for every strategy / fold / metric.
# ---------------------------------------------------------------------------
def check_percentile_ordering_on_disk() -> None:
    try:
        paths = sorted(ARTIFACT_DIR.glob("eval_*_bootstrap.json"))
        if not paths:
            record(
                3,
                "Percentile ordering on disk",
                False,
                "no eval_*_bootstrap.json files found - run `python ml/evaluate.py --bootstrap` first",
            )
            return
        violations: list[str] = []
        metric_keys = ("sharpe", "calmar", "profit_factor", "win_rate", "avg_r")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            strategy = payload.get("strategy_name", path.stem)
            for fold in payload.get("per_fold", []):
                fold_name = fold.get("fold", "?")
                for metric in metric_keys:
                    stats = fold.get(metric)
                    if not isinstance(stats, dict):
                        continue
                    p05, p50, p95 = stats.get("p05"), stats.get("p50"), stats.get("p95")
                    if p05 is None or p50 is None or p95 is None:
                        continue
                    if not (p05 <= p50 + 1e-9 <= p95 + 2e-9):
                        violations.append(
                            f"{strategy}/{fold_name}/{metric}: p05={p05} p50={p50} p95={p95}"
                        )
            agg = payload.get("aggregated", {})
            for metric in metric_keys:
                stats = agg.get(metric)
                if not isinstance(stats, dict):
                    continue
                p05, p50, p95 = stats.get("p05"), stats.get("p50"), stats.get("p95")
                if p05 is None or p50 is None or p95 is None:
                    continue
                if not (p05 <= p50 + 1e-9 <= p95 + 2e-9):
                    violations.append(
                        f"{strategy}/aggregated/{metric}: p05={p05} p50={p50} p95={p95}"
                    )
        record(
            3,
            "Percentile ordering p05 <= p50 <= p95 on every disk artifact",
            not violations,
            f"{len(paths)} files scanned, {len(violations)} violations"
            + (f": {violations[:5]}" if violations else ""),
        )
    except Exception as exc:
        record(3, "Percentile ordering on disk", False, f"exception: {exc}")


# ---------------------------------------------------------------------------
# Check 4 - Pass-rate bootstrap resamples DAY SEQUENCES and replays the
# TopStep manager. Ball-park: bootstrap_pass_rate must return exactly 0.0
# or 1.0 (per-sample outcomes), never a fractional single-sample value.
# ---------------------------------------------------------------------------
def check_pass_rate_uses_manager_replay() -> None:
    try:
        # A profitable sequence: +$150/day for 25 days should reach the
        # $3000 profit target. Bootstrap outcomes must be binary.
        daily = np.full(25, 150.0, dtype=float)
        result = bootstrap_pass_rate(
            daily, n_resamples=50, expected_block_len=3.0, random_state=9
        )
        # point should be 1.0 (we definitely hit $3750 > $3000 target)
        mean = result["mean"]
        p05, p95 = result["p05"], result["p95"]
        point_ok = result["point"] in (0.0, 1.0)
        mean_in_range = 0.0 <= mean <= 1.0
        p_in_range = 0.0 <= p05 <= 1.0 and 0.0 <= p95 <= 1.0
        # Also: a clearly losing sequence (-$200/day for 30 days) should produce
        # point = 0.0 because the account will hit DLL or MLL.
        losing = np.full(30, -200.0, dtype=float)
        bad = bootstrap_pass_rate(
            losing, n_resamples=50, expected_block_len=3.0, random_state=9
        )
        losing_point_zero = bad["point"] == 0.0 and bad["mean"] == 0.0
        passed = point_ok and mean_in_range and p_in_range and losing_point_zero
        record(
            4,
            "Pass-rate bootstrap replays TopStepRiskManager on day sequences",
            passed,
            (
                f"winning seq: point={result['point']} mean={mean:.3f} p05={p05:.3f} p95={p95:.3f}; "
                f"losing seq: point={bad['point']} mean={bad['mean']:.3f}"
            ),
        )
    except Exception as exc:
        record(4, "Pass-rate replay", False, f"exception: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Check 5 - Degenerate inputs never produce infinities: all-win, all-lose,
# and zero-volatility streams must keep PF and Calmar capped and Sharpe
# finite.
# ---------------------------------------------------------------------------
def check_no_infinities() -> None:
    try:
        scenarios = {
            "all_winning": np.full(40, 0.4, dtype=float),
            "all_losing": np.full(40, -0.3, dtype=float),
            "zero_pnl": np.zeros(40, dtype=float),
        }
        problems: list[str] = []
        for name, trades in scenarios.items():
            metrics = bootstrap_trade_metrics(
                trades, n_resamples=200, expected_block_len=4.0, random_state=11
            )
            for metric_name, stats in metrics.items():
                for key in ("point", "mean", "p05", "p50", "p95"):
                    v = stats.get(key)
                    if v is None:
                        continue
                    if not np.isfinite(v):
                        problems.append(f"{name}.{metric_name}.{key}={v}")
                    if metric_name == "profit_factor" and v > PF_CAP + 1e-9:
                        problems.append(
                            f"{name}.profit_factor.{key}={v} > PF_CAP={PF_CAP}"
                        )
                    if metric_name == "calmar" and abs(v) > CALMAR_CAP + 1e-9:
                        problems.append(
                            f"{name}.calmar.{key}={v} > CALMAR_CAP={CALMAR_CAP}"
                        )
        record(
            5,
            "Degenerate inputs stay finite and respect PF/Calmar caps",
            not problems,
            f"{len(problems)} violations" + (f": {problems[:5]}" if problems else ""),
        )
    except Exception as exc:
        record(5, "No infinities", False, f"exception: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Check 6 - Small-sample reliability flag fires when n < 30.
# ---------------------------------------------------------------------------
def check_small_sample_unreliable_flag() -> None:
    try:
        tiny = np.array([0.3, -0.1, 0.5], dtype=float)
        tiny_res = stationary_block_bootstrap(
            tiny, _sharpe, n_resamples=50, expected_block_len=2.0, random_state=1
        )
        big = np.random.default_rng(1).normal(0.1, 1.0, size=50)
        big_res = stationary_block_bootstrap(
            big, _sharpe, n_resamples=50, expected_block_len=2.0, random_state=1
        )
        passed = (
            tiny_res["unreliable"] is True
            and big_res["unreliable"] is False
            and tiny_res["n_trades"] < RELIABILITY_THRESHOLD
            and big_res["n_trades"] >= RELIABILITY_THRESHOLD
        )
        record(
            6,
            f"Unreliable flag fires for n<{RELIABILITY_THRESHOLD}, not for n>={RELIABILITY_THRESHOLD}",
            passed,
            f"n=3 unreliable={tiny_res['unreliable']}, n=50 unreliable={big_res['unreliable']}",
        )
    except Exception as exc:
        record(6, "Small-sample flag", False, f"exception: {exc}")


# ---------------------------------------------------------------------------
# Check 7 - Deployment gate rejects clear failures on synthetic inputs.
# ---------------------------------------------------------------------------
def check_gate_rejection_paths() -> None:
    try:
        def _mk(
            p05: float, p50: float, pf: float, pass_p05: float, fold_sharpes: list[float]
        ) -> dict:
            return {
                "aggregated": {
                    "sharpe": {"p05": p05, "p50": p50, "p95": p50 + 0.1, "point": p50},
                    "profit_factor": {"p05": pf - 0.2, "p50": pf, "p95": pf + 0.2, "point": pf},
                    "pass_rate": {"p05": pass_p05, "p50": pass_p05 + 0.1, "p95": pass_p05 + 0.2, "point": pass_p05},
                    "n_trades": 200,
                },
                "per_fold": [
                    {"fold": f"fold_{i+1}", "sharpe": {"point": s, "p05": s - 0.3, "p50": s, "p95": s + 0.3}}
                    for i, s in enumerate(fold_sharpes)
                ],
            }

        rejects = [
            ("negative p05 Sharpe", _mk(-0.01, 1.0, 1.5, 0.5, [0.5, 0.8, 1.0, 1.1, 1.2]), "p05 Sharpe"),
            ("one bad fold", _mk(0.2, 0.9, 1.5, 0.5, [0.5, 0.8, 1.0, 1.1, -0.4]), "per-fold point Sharpe"),
            ("weak PF", _mk(0.1, 0.8, 1.1, 0.5, [0.2, 0.5, 0.8, 1.0, 1.2]), "p50 profit factor"),
            ("weak pass rate", _mk(0.1, 0.8, 1.5, 0.20, [0.2, 0.5, 0.8, 1.0, 1.2]), "p05 pass rate"),
        ]
        accept = _mk(0.25, 0.75, 1.35, 0.40, [0.4, 0.6, 0.8, 1.0, 1.1])

        failures: list[str] = []
        for label, results, expected_token in rejects:
            d = evaluate_bootstrap_deployment_gate(results)
            if d.approved:
                failures.append(f"{label}: expected reject but approved")
            elif expected_token not in d.reason:
                failures.append(f"{label}: reason missing token '{expected_token}' -> {d.reason}")
        acc = evaluate_bootstrap_deployment_gate(accept)
        if not acc.approved:
            failures.append(f"accept-case rejected: {acc.reason}")
        record(
            7,
            "Deployment gate rejects on each failure axis and accepts clean case",
            not failures,
            "all 4 rejection paths + 1 acceptance path verified" if not failures else f"{failures}",
        )
    except Exception as exc:
        record(7, "Gate rejection paths", False, f"exception: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Check 8 - Multiple-testing filter: p50 Sharpe >= 0.5 is required by default
# but can be disabled via require_p50_sharpe_above=None.
# ---------------------------------------------------------------------------
def check_multiple_testing_filter() -> None:
    try:
        results = {
            "aggregated": {
                "sharpe": {"p05": 0.2, "p50": 0.35, "p95": 0.5, "point": 0.35},
                "profit_factor": {"p05": 1.3, "p50": 1.4, "p95": 1.5, "point": 1.4},
                "pass_rate": {"p05": 0.40, "p50": 0.55, "p95": 0.7, "point": 0.55},
                "n_trades": 180,
            },
            "per_fold": [
                {"fold": f"fold_{i+1}", "sharpe": {"point": s, "p05": s - 0.3, "p50": s, "p95": s + 0.3}}
                for i, s in enumerate([0.2, 0.3, 0.4, 0.35, 0.3])
            ],
        }
        with_filter = evaluate_bootstrap_deployment_gate(results)
        without_filter = evaluate_bootstrap_deployment_gate(
            results, require_p50_sharpe_above=None
        )
        passed = (
            with_filter.approved is False
            and "multiple-testing filter" in with_filter.reason
            and without_filter.approved is True
        )
        record(
            8,
            "Multiple-testing filter rejects p50 Sharpe < 0.5 by default, toggles off with None",
            passed,
            f"with_filter.approved={with_filter.approved} ('{with_filter.reason}'); "
            f"without_filter.approved={without_filter.approved}",
        )
    except Exception as exc:
        record(8, "Multiple-testing filter", False, f"exception: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Check 9 - On-disk aggregation sanity: for every strategy, the sum of
# per-fold n_trades equals the aggregated n_trades. Keeps the 3E artifact
# from silently dropping folds.
# ---------------------------------------------------------------------------
def check_aggregated_counts_match_sum_of_folds() -> None:
    try:
        paths = sorted(ARTIFACT_DIR.glob("eval_*_bootstrap.json"))
        if not paths:
            record(
                9,
                "Aggregated n_trades matches sum of per-fold n_trades",
                False,
                "no eval_*_bootstrap.json files found",
            )
            return
        mismatches: list[str] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            strategy = payload.get("strategy_name", path.stem)
            per_fold_sum = sum(int(f.get("n_trades", 0)) for f in payload.get("per_fold", []))
            agg = int(payload.get("aggregated", {}).get("n_trades", 0))
            if per_fold_sum != agg:
                mismatches.append(f"{strategy}: sum(folds)={per_fold_sum} vs aggregated={agg}")
        record(
            9,
            "Aggregated n_trades matches sum of per-fold n_trades",
            not mismatches,
            f"{len(paths)} strategies, {len(mismatches)} mismatches"
            + (f": {mismatches}" if mismatches else ""),
        )
    except Exception as exc:
        record(9, "Aggregated counts", False, f"exception: {exc}")


# ---------------------------------------------------------------------------
# Check 10 - Full pytest suite remains green.
# ---------------------------------------------------------------------------
def check_full_test_suite() -> None:
    import subprocess

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "ml/tests", "-q", "--tb=short"],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        summary = ""
        for line_ in reversed(proc.stdout.splitlines() + proc.stderr.splitlines()):
            if "passed" in line_ or "failed" in line_ or "error" in line_:
                summary = line_.strip()
                break
        passed = proc.returncode == 0
        record(
            10,
            "Full pytest suite is green",
            passed,
            f"returncode={proc.returncode} | {summary}",
        )
    except Exception as exc:
        record(10, "Full pytest suite", False, f"exception: {exc}")


def main() -> int:
    checks = [
        check_autocorrelation_preservation,
        check_reproducibility,
        check_percentile_ordering_on_disk,
        check_pass_rate_uses_manager_replay,
        check_no_infinities,
        check_small_sample_unreliable_flag,
        check_gate_rejection_paths,
        check_multiple_testing_filter,
        check_aggregated_counts_match_sum_of_folds,
        check_full_test_suite,
    ]
    for fn in checks:
        fn()

    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for idx, name, passed, evidence in RESULTS:
        tag = "PASS" if passed else "FAIL"
        log(f"[{idx:>2}] {tag} - {name}")

    all_pass = all(r[2] for r in RESULTS)
    log("")
    log("VERDICT: " + ("ALL PASS" if all_pass else "ONE OR MORE FAILED"))

    out_path = ARTIFACT_DIR / "agent3e_audit_log.txt"
    out_path.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    log(f"log written: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

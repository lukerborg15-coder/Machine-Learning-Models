"""Agent 3D audit — 10 adversarial checks. Run as a script, not via pytest.

Usage:
    cd C:\\Users\\Luker\\strategyLabbrain
    python ml/tests/_audit_3d.py

Exits 0 if all checks PASS, 1 if any FAIL. Writes the full check log to
``ml/artifacts/agent3d_audit_log.txt`` for later inclusion in AGENT3D_AUDIT.md.
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

from ml.evaluate import simulate_trading  # noqa: E402
from ml.labels import triple_barrier_label  # noqa: E402
from ml.train import compute_binary_pos_weight  # noqa: E402

RESULTS: list[tuple[int, str, bool, str]] = []
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


def record(idx: int, name: str, passed: bool, evidence: str) -> None:
    RESULTS.append((idx, name, passed, evidence))
    tag = "PASS" if passed else "FAIL"
    log(f"[CHECK {idx}] {tag} — {name}")
    log(f"         {evidence}")


def _price_frame(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    start: str = "2025-01-02 09:30",
    freq: str = "5min",
) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close), freq=freq, tz="America/New_York", name="datetime")
    close_series = pd.Series(close, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close_series,
            "high": high if high is not None else (close_series + 0.1).tolist(),
            "low": low if low is not None else (close_series - 0.1).tolist(),
            "close": close_series,
            "volume": np.full(len(close), 1_000.0),
        },
        index=index,
    )


# ------------------------------------------------------------
# Check 1 — synthetic stop-hit-first (long)
# ------------------------------------------------------------
def check_1() -> None:
    # entry @ bar 0 = 100. ATR=1.0, stop_atr_mult=1.5 -> stop = 98.5, target = 101.5.
    # Bar 2 has low=98.0 (<= 98.5) and no target hit -> stop wins at 98.5.
    close = [100.0, 100.0, 98.0, 98.2, 98.5]
    high = [100.0, 100.2, 100.0, 98.5, 99.0]
    low = [100.0, 99.8, 98.0, 98.0, 98.0]
    frame = _price_frame(close, high=high, low=low)
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=10, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    ok = (
        int(row["label"]) == 0
        and row["barrier_hit"] == "stop"
        and abs(float(row["exit_price"]) - 98.5) < 1e-9
        and int(row["exit_bar"]) == 2
    )
    record(
        1,
        "synthetic stop-hit-first (long)",
        ok,
        f"label={row['label']}, barrier_hit={row['barrier_hit']}, exit_price={row['exit_price']}, exit_bar={row['exit_bar']}",
    )


# ------------------------------------------------------------
# Check 2 — synthetic target-hit-first (long)
# ------------------------------------------------------------
def check_2() -> None:
    # entry @ bar 0 = 100. ATR=1.0 -> target = 101.5, stop = 98.5.
    # Bar 2 has high=102.0 (>= 101.5), low=100.2 (no stop) -> target wins at 101.5.
    close = [100.0, 100.5, 101.8, 101.0, 101.0]
    high = [100.0, 100.8, 102.0, 101.2, 101.2]
    low = [100.0, 100.2, 100.2, 100.5, 100.5]
    frame = _price_frame(close, high=high, low=low)
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=10, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    ok = (
        int(row["label"]) == 1
        and row["barrier_hit"] == "target"
        and abs(float(row["exit_price"]) - 101.5) < 1e-9
        and int(row["exit_bar"]) == 2
    )
    record(
        2,
        "synthetic target-hit-first (long)",
        ok,
        f"label={row['label']}, barrier_hit={row['barrier_hit']}, exit_price={row['exit_price']}, exit_bar={row['exit_bar']}",
    )


# ------------------------------------------------------------
# Check 3 — short symmetry
# ------------------------------------------------------------
def check_3() -> None:
    # Short entry @ 100. stop = 101.5, target = 98.5. Price falls, bar 2 low=98.0 -> target wins.
    close = [100.0, 99.5, 98.2, 98.0, 98.0]
    high = [100.0, 100.0, 99.0, 98.2, 98.2]
    low = [100.0, 99.0, 98.0, 97.8, 97.8]
    frame = _price_frame(close, high=high, low=low)
    signal = pd.Series([-1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=10, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    ok = (
        int(row["label"]) == 1
        and row["barrier_hit"] == "target"
        and float(row["r_multiple"]) > 0
        and abs(float(row["exit_price"]) - 98.5) < 1e-9
    )
    record(
        3,
        "short symmetry",
        ok,
        f"label={row['label']}, barrier_hit={row['barrier_hit']}, r_multiple={row['r_multiple']:.4f}, exit_price={row['exit_price']}",
    )


# ------------------------------------------------------------
# Check 4 — same-bar conflict: stop wins (conservative)
# ------------------------------------------------------------
def check_4() -> None:
    # entry @ 100. ATR=1.0 -> stop=98.5, target=101.5.
    # Bar 2 has high=102.0 AND low=98.0 (both barriers touched) -> conservative stop wins.
    close = [100.0, 100.0, 100.0, 100.0, 100.0]
    high = [100.0, 100.5, 102.0, 101.0, 101.0]
    low = [100.0, 99.5, 98.0, 99.0, 99.0]
    frame = _price_frame(close, high=high, low=low)
    signal = pd.Series([1, 0, 0, 0, 0], index=frame.index)
    atr = pd.Series(1.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=10, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    ok = (
        int(row["label"]) == 0
        and row["barrier_hit"] == "stop"
        and abs(float(row["exit_price"]) - 98.5) < 1e-9
    )
    record(
        4,
        "same-bar conflict stop-wins",
        ok,
        f"label={row['label']}, barrier_hit={row['barrier_hit']}, exit_price={row['exit_price']} (labels.py loop checks stop_hit before target_hit at lines 123-134)",
    )


# ------------------------------------------------------------
# Check 5 — session boundary at 15:00 ET
# ------------------------------------------------------------
def check_5() -> None:
    # Signal fires at 14:30 ET on a 5-min timeframe. max_bars=60 would walk to 19:30.
    # Expected: vertical barrier truncated at or before 15:00 ET.
    close = [100.0] * 12  # 14:30, 14:35, ..., 15:25
    high = [100.1] * 12
    low = [99.9] * 12
    frame = _price_frame(close, high=high, low=low, start="2025-01-02 14:30", freq="5min")
    signal = pd.Series([1] + [0] * 11, index=frame.index)
    atr = pd.Series(10.0, index=frame.index)  # big ATR so neither stop nor target hit

    labels = triple_barrier_label(frame, signal, atr, max_bars=60, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    exit_time = row["exit_time"]
    # Session end is 15:00. The last in-session bar is the one at exactly 15:00 (timestamp > session_end is false for equality).
    # So exit_time should be at 15:00 ET.
    ok = (
        isinstance(exit_time, pd.Timestamp)
        and exit_time.hour == 15
        and exit_time.minute == 0
        and row["barrier_hit"] == "vertical"
    )
    record(
        5,
        "session boundary at 15:00 ET",
        ok,
        f"exit_time={exit_time}, barrier_hit={row['barrier_hit']}",
    )


# ------------------------------------------------------------
# Check 6 — label/trade-sim consistency on 20 real signal bars
# ------------------------------------------------------------
def check_6() -> None:
    # The features parquet stores normalized OHLC (open_norm, ..., close_norm).
    # Raw OHLC must be loaded via dataset_builder.load_data("mnq", "5min") and
    # joined with the feature parquet's signal + atr columns.
    from ml.dataset_builder import load_data  # local import to avoid import cost at module load

    parquet_path = Path(ROOT_DIR) / "ml" / "data" / "features_mnq_5min.parquet"
    if not parquet_path.exists():
        record(6, "label/trade-sim consistency on 20 real signal bars", False, f"missing {parquet_path}")
        return

    feat = pd.read_parquet(parquet_path)
    if not isinstance(feat.index, pd.DatetimeIndex):
        if "datetime" in feat.columns:
            feat["datetime"] = pd.to_datetime(feat["datetime"])
            feat = feat.set_index("datetime")
    feat = feat.sort_index()

    sig_col = "ifvg_signal"
    if sig_col not in feat.columns:
        record(6, "label/trade-sim consistency on 20 real signal bars", False, f"missing column {sig_col}")
        return
    if "atr_14" not in feat.columns:
        record(6, "label/trade-sim consistency on 20 real signal bars", False, "missing atr_14 in features parquet")
        return

    raw = load_data("mnq", "5min", session_only=True)
    # Join: raw provides open/high/low/close; features provide signal + atr_14.
    merged = raw.loc[:, ["open", "high", "low", "close"]].join(
        feat.loc[:, [sig_col, "atr_14"]], how="inner"
    ).sort_index()
    merged = merged.rename(columns={"atr_14": "atr"})

    active = merged.index[(merged[sig_col] != 0) & merged[sig_col].notna()]
    if len(active) < 20:
        record(6, "label/trade-sim consistency on 20 real signal bars", False, f"only {len(active)} active signals")
        return
    sample_ts = list(active[:20])

    labels = triple_barrier_label(
        merged.loc[:, ["open", "high", "low", "close"]],
        signal_series=merged[sig_col],
        atr_series=merged["atr"],
        stop_atr_mult=1.5,
        target_r_mult=1.0,
        max_bars=60,
        transaction_cost_pts=0.07,
    )

    mismatches: list[str] = []
    matches = 0
    comparable = 0
    for ts in sample_ts:
        label_row = labels.loc[ts]
        if pd.isna(label_row["label"]):
            continue

        # Build a single-signal slice for simulate_trading.
        start_idx = merged.index.get_loc(ts)
        end_idx = min(start_idx + 60, len(merged) - 1)
        slice_frame = merged.iloc[start_idx : end_idx + 1].copy()
        sigs = pd.Series(0, index=slice_frame.index, dtype=int)
        sigs.iloc[0] = int(np.sign(merged[sig_col].loc[ts]))
        slice_frame[sig_col] = sigs
        # simulate_trading three_class mapping: 0=long, 1=short, 2=no_trade.
        preds = pd.Series(2, index=slice_frame.index, dtype=int)
        preds.iloc[0] = 0 if sigs.iloc[0] > 0 else 1
        slice_frame["prediction"] = preds

        try:
            sim = simulate_trading(
                slice_frame,
                strategy_name="ifvg",
                signal_column=sig_col,
                objective="three_class",
            )
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{ts}: simulate_trading error {exc}")
            continue
        if sim["trades"].empty:
            # Trade sim refused (e.g. DD block / session cutoff) — skip.
            continue

        trade = sim["trades"].iloc[0]
        sim_pnl = float(trade["pnl"])
        label = int(label_row["label"])
        comparable += 1
        if label == 1 and sim_pnl > 0:
            matches += 1
        elif label == 0 and sim_pnl <= 0:
            matches += 1
        else:
            mismatches.append(
                f"{ts}: label={label} sim_pnl={sim_pnl:.2f} direction={int(trade['direction'])}"
            )

    if comparable == 0:
        record(6, "label/trade-sim consistency on 20 real signal bars", False, "no comparable pairs produced")
        return

    # Tolerate up to 4/20 mismatches (simulate_trading has DD/commission/session
    # constraints the label generator does not).
    tolerance = 4
    passed = len(mismatches) <= tolerance
    record(
        6,
        "label/trade-sim consistency on 20 real signal bars",
        passed,
        f"comparable={comparable}, matches={matches}, mismatches={len(mismatches)} (tolerance={tolerance}). First mismatches: {mismatches[:3]}",
    )


# ------------------------------------------------------------
# Check 7 — vertical-barrier transaction cost
# ------------------------------------------------------------
def check_7() -> None:
    # Flat price for 60 bars. ATR=10 so stop/target never hit. Vertical barrier:
    # pnl_pts = 0 - 0.07 = -0.07 -> label = 0.
    close = [100.0] * 60
    high = [100.0] * 60
    low = [100.0] * 60
    frame = _price_frame(close, high=high, low=low, start="2025-01-02 09:30")
    # Drop past session to stay within 09:30-15:00 ET = 67 bars max. Trim to 60.
    # Note _price_frame uses 5min; 60 bars from 09:30 is 14:25 — still in session.
    signal = pd.Series([1] + [0] * 59, index=frame.index)
    atr = pd.Series(10.0, index=frame.index)

    labels = triple_barrier_label(frame, signal, atr, max_bars=60, transaction_cost_pts=0.07)
    row = labels.iloc[0]
    ok = (
        int(row["label"]) == 0
        and row["barrier_hit"] == "vertical"
        and float(row["r_multiple"]) < 0
    )
    record(
        7,
        "vertical-barrier transaction cost",
        ok,
        f"label={row['label']}, barrier_hit={row['barrier_hit']}, r_multiple={row['r_multiple']:.6f} (labels.py:138-140 subtracts transaction_cost_pts before sign check)",
    )


# ------------------------------------------------------------
# Check 8 — class weight train-split only
# ------------------------------------------------------------
def check_8() -> None:
    fold_path = Path(ROOT_DIR) / "ml" / "artifacts" / "eval_ifvg_fold_1.csv"
    if not fold_path.exists():
        record(8, "class weight train-split only", False, f"missing {fold_path}")
        return
    df = pd.read_csv(fold_path)
    row = df.iloc[0]
    c0 = int(row["class_count_0"])
    c1 = int(row["class_count_1"])
    stored_pw = float(row["pos_weight"])
    train_windows = int(row["train_windows"])
    val_windows = int(row["val_windows"])

    # 1. class counts sum to train_windows (not val_windows)
    counts_match_train = (c0 + c1 == train_windows)
    counts_match_val = (c0 + c1 == val_windows)

    # 2. stored pos_weight equals compute_binary_pos_weight on synthetic labels with counts c0, c1.
    reconstructed = compute_binary_pos_weight(np.array([0] * c0 + [1] * c1, dtype=int))
    weight_matches = abs(reconstructed - stored_pw) < 1e-9

    # 3. formula: pos_weight = n_neg / n_pos (= c0 / c1 here)
    formula_matches = abs(stored_pw - (c0 / c1)) < 1e-9

    ok = counts_match_train and (not counts_match_val) and weight_matches and formula_matches
    record(
        8,
        "class weight train-split only",
        ok,
        f"c0={c0} c1={c1} sum={c0+c1} train_windows={train_windows} val_windows={val_windows} stored_pw={stored_pw} reconstructed={reconstructed:.12f} formula_c0/c1={c0/c1:.12f}",
    )


# ------------------------------------------------------------
# Check 9 — threshold-sweep monotonic sanity
# ------------------------------------------------------------
def check_9() -> None:
    report_path = Path(ROOT_DIR) / "ml" / "artifacts" / "FINAL_EVAL_REPORT.md"
    if not report_path.exists():
        record(9, "threshold-sweep monotonic sanity", False, f"missing {report_path}")
        return
    text = report_path.read_text(encoding="utf-8")
    # Find table rows: start with "| <strategy>" and contain 14 pipes separating 13 columns.
    rows = [ln for ln in text.splitlines() if ln.startswith("|") and "---" not in ln and "Strategy" not in ln and "|" in ln]
    # Report header columns:
    # | Strategy | Old 3c | AUC | Brier | S@0.50 | T@0.50 | S@0.55 | T@0.55 | S@0.60 | T@0.60 | S@0.65 | T@0.65 | S@0.70 | T@0.70 |
    violators: list[str] = []
    checked = 0
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 14:
            continue
        strategy = cells[0]
        # Trades columns are at indices 5, 7, 9, 11, 13 (T@0.50 .. T@0.70)
        try:
            trades = [float(cells[i]) if cells[i] else float("nan") for i in (5, 7, 9, 11, 13)]
        except ValueError:
            continue
        finite = [t for t in trades if np.isfinite(t)]
        if not finite:
            continue
        checked += 1
        for a, b in zip(trades, trades[1:]):
            if np.isfinite(a) and np.isfinite(b) and b > a + 1e-9:
                violators.append(f"{strategy}: {trades}")
                break

    ok = checked > 0 and not violators
    record(
        9,
        "threshold-sweep monotonic sanity",
        ok,
        f"strategies_checked={checked}, violators={violators}",
    )


# ------------------------------------------------------------
# Run all
# ------------------------------------------------------------
def main() -> int:
    checks = [check_1, check_2, check_3, check_4, check_5, check_6, check_7, check_8, check_9]
    for fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            idx = int(fn.__name__.split("_")[-1])
            record(idx, f"{fn.__name__} raised", False, f"{exc}\n{tb}")

    log("")
    log("=" * 60)
    log("Agent 3D audit summary")
    log("=" * 60)
    for idx, name, ok, _ in sorted(RESULTS):
        log(f"  {idx:>2}. {'PASS' if ok else 'FAIL'} — {name}")

    artifact = Path(ROOT_DIR) / "ml" / "artifacts" / "agent3d_audit_log.txt"
    artifact.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    log(f"\nWrote {artifact}")

    results_json = Path(ROOT_DIR) / "ml" / "artifacts" / "agent3d_audit_results.json"
    results_json.write_text(
        json.dumps([{"idx": idx, "name": n, "passed": ok, "evidence": ev} for idx, n, ok, ev in RESULTS], indent=2),
        encoding="utf-8",
    )
    log(f"Wrote {results_json}")

    return 0 if all(ok for _, _, ok, _ in RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

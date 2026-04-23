"""Build old 3-class vs meta-label Sharpe comparison table for AGENT3D_AUDIT.md.

Uses ``ml/artifacts/agent3d_old_three_class_sharpe.csv`` and the per-fold
``eval_{strategy}_fold_*.csv`` artifacts. No retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS = ROOT_DIR / "ml" / "artifacts"

STRATEGIES = ["ifvg", "ifvg_open", "orb_ib", "orb_vol", "orb_wick", "orb_va", "ttm", "connors"]
THRESHOLDS = [("0_50", "0.50"), ("0_55", "0.55"), ("0_60", "0.60"), ("0_65", "0.65"), ("0_70", "0.70")]


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.3f}"


def main() -> int:
    old = pd.read_csv(ARTIFACTS / "agent3d_old_three_class_sharpe.csv")
    old_map = dict(zip(old["strategy_name"], pd.to_numeric(old["old_three_class_test_sharpe_median"], errors="coerce")))

    lines = [
        "| Strategy | Old 3-class Sharpe | Meta @0.50 | Meta @0.55 | Meta @0.60 | Meta @0.65 | Meta @0.70 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for strategy in STRATEGIES:
        old_sharpe = float(old_map.get(strategy, float("nan")))

        sharpes: dict[str, float] = {suffix: float("nan") for suffix, _ in THRESHOLDS}
        fold_values: dict[str, list[float]] = {suffix: [] for suffix, _ in THRESHOLDS}
        for fold in range(1, 6):
            fold_csv = ARTIFACTS / f"eval_{strategy}_fold_{fold}.csv"
            if not fold_csv.exists():
                continue
            df = pd.read_csv(fold_csv)
            if df.empty:
                continue
            row = df.iloc[0]
            for suffix, _ in THRESHOLDS:
                col = f"test_sharpe_thr_{suffix}"
                if col in row and pd.notna(row[col]):
                    try:
                        fold_values[suffix].append(float(row[col]))
                    except (TypeError, ValueError):
                        continue
        for suffix, _ in THRESHOLDS:
            arr = np.asarray([v for v in fold_values[suffix] if np.isfinite(v)], dtype=float)
            sharpes[suffix] = float(np.median(arr)) if arr.size else float("nan")

        lines.append(
            "| {s} | {o} | {a} | {b} | {c} | {d} | {e} |".format(
                s=strategy,
                o=_fmt(old_sharpe),
                a=_fmt(sharpes["0_50"]),
                b=_fmt(sharpes["0_55"]),
                c=_fmt(sharpes["0_60"]),
                d=_fmt(sharpes["0_65"]),
                e=_fmt(sharpes["0_70"]),
            )
        )

    output = ARTIFACTS / "agent3d_audit_sharpe_comparison.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

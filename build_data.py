"""
build_data.py — Convert Databento 24h 1min CSV to continuous front-month
OHLCV CSVs for all required timeframes.

Usage (zip):
    python build_data.py --zip "C:/Users/Luker/Downloads/GLBX-20260322-DXXDYL39QB (1).zip"

Usage (raw CSV — use this if you downloaded the CSV directly from Databento):
    python build_data.py --csv "C:/Users/Luker/Downloads/GLBX-20260421-UK8HFFC6GA/glbx-mdp3-20210319-20260318.ohlcv-1m.csv"

Output (written to data/):
    mnq_1min_databento.csv
    mnq_2min_databento.csv
    mnq_3min_databento.csv
    mnq_5min_databento.csv
    mnq_15min_databento.csv
    mnq_30min_databento.csv
    mnq_1h_databento.csv
"""

import argparse
import zipfile
import io
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"
TIMEFRAMES = {
    "1min":  "1min",
    "2min":  "2min",
    "3min":  "3min",
    "5min":  "5min",
    "15min": "15min",
    "30min": "30min",
    "1h":    "1h",
}
INSTRUMENT = "mnq"

# ── Helpers ───────────────────────────────────────────────────────────────────
_RAW_COLS = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
_RAW_DTYPES = {
    "open": float, "high": float, "low": float,
    "close": float, "volume": float, "symbol": str,
}


def load_raw(zip_path: Path) -> pd.DataFrame:
    """Extract the OHLCV CSV from the zip and load into DataFrame."""
    print(f"Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        # Find the ohlcv csv inside the zip
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv") and "ohlcv" in n)
        print(f"  Reading: {csv_name}  ({zf.getinfo(csv_name).file_size / 1e6:.0f} MB)")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, usecols=_RAW_COLS, dtype=_RAW_DTYPES)
    print(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} unique symbols")
    return df


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load a raw Databento OHLCV CSV directly (no zip required)."""
    print(f"Reading CSV: {csv_path}")
    file_size_mb = csv_path.stat().st_size / 1e6
    print(f"  File size: {file_size_mb:.0f} MB — this may take a minute for large files...")
    df = pd.read_csv(csv_path, usecols=_RAW_COLS, dtype=_RAW_DTYPES)
    print(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} unique symbols")
    return df


def filter_front_month(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Keep only the front-month continuous contract for the given instrument.

    Strategy:
    - Filter to rows whose symbol starts with the instrument prefix (e.g. MNQ)
      and contains NO hyphen (excludes spread contracts like MNQH1-MNQM1).
    - At each 1min timestamp, keep only the contract with the highest volume
      (the actively traded front month).
    """
    prefix = instrument.upper()
    mask = df["symbol"].str.startswith(prefix) & ~df["symbol"].str.contains("-")
    df = df[mask].copy()
    print(f"  After symbol filter: {len(df):,} rows, contracts: {sorted(df['symbol'].unique())[:10]}")

    # Parse timestamps — Databento uses nanosecond UTC strings
    df["datetime"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.drop(columns=["ts_event"])

    # At each timestamp keep the highest-volume contract (front month)
    df = (
        df.sort_values(["datetime", "volume"], ascending=[True, False])
          .drop_duplicates(subset="datetime", keep="first")
          .reset_index(drop=True)
    )
    print(f"  After front-month dedup: {len(df):,} rows")
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1min bars to a higher timeframe using standard OHLCV aggregation."""
    df = df.set_index("datetime").sort_index()
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    # closed/label='left' means the bar labelled 09:30 covers 09:30–09:34 (for 5min)
    resampled = df[list(agg.keys())].resample(rule, closed="left", label="left").agg(agg)
    # Drop bars with no trades (gaps in overnight session)
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled[resampled["volume"] > 0]
    resampled = resampled.reset_index()
    return resampled


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format to the expected output schema:
        datetime,open,high,low,close,volume
    datetime is Eastern time with UTC offset string.
    """
    df = df.copy()
    df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
    # Keep only required columns in order
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows -> {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="Path to Databento zip file")
    group.add_argument("--csv", help="Path to raw Databento OHLCV CSV file (use if you downloaded CSV directly)")
    parser.add_argument("--instrument", default=INSTRUMENT, help="Instrument prefix (default: mnq)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw 1min data (zip or raw csv)
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)
        raw = load_raw_csv(csv_path)
    else:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            print(f"ERROR: zip not found: {zip_path}")
            sys.exit(1)
        raw = load_raw(zip_path)

    # Step 2: Filter to front-month continuous contract
    print(f"\nBuilding front-month continuous contract for {args.instrument.upper()}...")
    df_1min = filter_front_month(raw, args.instrument)

    # Verify we have overnight data
    eastern = df_1min["datetime"].dt.tz_convert("America/New_York")
    hours = eastern.dt.hour.unique()
    overnight_hours = [h for h in hours if h < 9 or h >= 16]
    if overnight_hours:
        print(f"  PASS: Overnight data confirmed - hours present: {sorted(overnight_hours)[:6]}...")
    else:
        print("  WARNING: No overnight hours found - data may be RTH-only")

    # Step 3: Save 1min (already at correct resolution)
    df_out = format_output(df_1min)
    save_csv(df_out, DATA_DIR / f"{args.instrument}_1min_databento.csv")

    # Step 4: Resample to all other timeframes
    resample_map = {
        "2min":  "2min",
        "3min":  "3min",
        "5min":  "5min",
        "15min": "15min",
        "30min": "30min",
        "1h":    "1h",
    }

    print("\nResampling to higher timeframes...")
    for tf_name, rule in resample_map.items():
        resampled = resample_ohlcv(df_1min.copy(), rule)
        df_out = format_output(resampled)
        save_csv(df_out, DATA_DIR / f"{args.instrument}_{tf_name}_databento.csv")

    # Step 5: Sanity check
    print("\n-- Sanity check --")
    df_5min = pd.read_csv(DATA_DIR / f"{args.instrument}_5min_databento.csv", nrows=5)
    print("5min first 5 rows:")
    print(df_5min[["datetime", "open", "close", "volume"]].to_string(index=False))

    df_all = pd.read_csv(DATA_DIR / f"{args.instrument}_5min_databento.csv")
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], utc=True).dt.tz_convert("America/New_York")
    hours_in_file = sorted(df_all["datetime"].dt.hour.unique())
    print(f"\n5min unique hours in file: {hours_in_file}")
    if any(h < 9 for h in hours_in_file):
        print("PASS: Overnight data present in 5min output")
    else:
        print("WARNING: Only RTH hours in 5min output")

    total_rows = sum(
        len(pd.read_csv(DATA_DIR / f"{args.instrument}_{tf}_databento.csv"))
        for tf in ["1min", "2min", "3min", "5min", "15min", "30min", "1h"]
    )
    print(f"\nAll timeframes written. Total rows across all files: {total_rows:,}")
    print("\nDone. Run `python ml/dataset_builder.py --rebuild` to rebuild feature parquets.")


if __name__ == "__main__":
    main()

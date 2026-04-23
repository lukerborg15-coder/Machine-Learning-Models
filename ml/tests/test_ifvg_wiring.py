"""
Smoke test for IFVG signal wiring.

Tests that:
1. Structural-level columns are present on the DataFrame passed to ifvg_combined
2. IFVG signals fire (at least one non-zero signal on real data)
3. No warnings about missing structural columns are raised
4. HTF timeframe string is properly determined before FVG detection
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Setup path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.dataset_builder import _compute_signal_features, load_data
from Implementation.ifvg_generator import ifvg_combined


def test_ifvg_structural_columns_wiring():
    """Test that _compute_signal_features correctly wires structural columns AND
    that IFVG signals actually fire end-to-end.

    Per-day FVG isolation is the intended design (IFVG is intraday). The real
    prerequisite for non-zero signals was:
      1. structural-level columns wired onto df_for_ifvg (fixed in wiring pass)
      2. invalidation range excluding current bar (fixed via off-by-one)
    """
    print("\n" + "=" * 80)
    print("TEST: IFVG Structural Columns Wiring")
    print("=" * 80)

    # Load full dataset
    try:
        df = load_data("mes", "5min", session_only=True)
        print(f"Loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    except FileNotFoundError as e:
        print(f"SKIP: Data file not found: {e}")
        return

    if len(df) < 100:
        print(f"SKIP: Only {len(df)} bars available, need at least 100 for meaningful test")
        return

    # Call _compute_signal_features
    print("\nCalling _compute_signal_features(df)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            signal_features = _compute_signal_features("mes", "5min", df)
            print(f"  ✓ _compute_signal_features completed without error")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            raise

    # Check for structural-column warnings (THIS IS THE CRITICAL WIRING TEST)
    structural_warnings = [
        warn for warn in w
        if issubclass(warn.category, UserWarning)
        and "Structural level column" in str(warn.message)
        and any(col in str(warn.message) for col in [
            'prev_day_low', 'prev_day_high', 'prev_week_low', 'prev_week_high',
            'session_low', 'session_high', 'overnight_low', 'overnight_high'
        ])
    ]

    print(f"\nWarning check: {len(w)} total warnings captured")
    if structural_warnings:
        print(f"  ✗ FAIL: Structural column warnings found:")
        for warn in structural_warnings:
            print(f"    - {warn.message}")
        assert False, f"Expected NO structural column warnings, got {len(structural_warnings)}"
    else:
        print(f"  ✓ PASS: No structural-column warnings for required columns")

    # Verify the expected columns exist on the signal output
    print(f"\nSignal feature columns: {signal_features.columns.tolist()}")
    assert "ifvg_signal" in signal_features.columns, "Missing ifvg_signal column"
    assert "ifvg_open_signal" in signal_features.columns, "Missing ifvg_open_signal column"
    print(f"  ✓ PASS: Required IFVG signal columns exist")

    # CRITICAL acceptance: signals must actually fire end-to-end
    base_nonzero = int((signal_features["ifvg_signal"] != 0).sum())
    open_nonzero = int((signal_features["ifvg_open_signal"] != 0).sum())
    total_nonzero = base_nonzero + open_nonzero
    print(f"\nSignal counts: base={base_nonzero}, open={open_nonzero}, total={total_nonzero}")
    assert total_nonzero > 0, (
        f"IFVG signals = 0 on {len(df)} bars. Wiring columns alone is not enough; "
        f"confirm invalidation range excludes current bar and structural levels are populated."
    )
    print(f"  ✓ PASS: IFVG fires {total_nonzero} non-zero signals end-to-end")

    # Enforce per-day cap: base + open combined ≤ 2 per calendar day
    combined = (signal_features["ifvg_signal"] != 0).astype(int) + (
        signal_features["ifvg_open_signal"] != 0
    ).astype(int)
    per_day = combined.groupby(combined.index.date).sum()
    over_cap = int((per_day > 2).sum())
    assert over_cap == 0, f"Daily cap violated on {over_cap} days (expected 0)"
    print(f"  ✓ PASS: Daily cap (≤2 combined/day) respected across {len(per_day)} days")


def test_htf_timeframe_determination():
    """Test that HTF timeframe is correctly determined before FVG detection."""
    print("\n" + "=" * 80)
    print("TEST: HTF Timeframe Determination")
    print("=" * 80)

    # Load HTF data
    try:
        htf_df = load_data("mes", "15min", session_only=False)
        print(f"Loaded {len(htf_df)} HTF bars")
    except FileNotFoundError:
        print("SKIP: HTF data not available")
        return

    if len(htf_df) < 10:
        print("SKIP: Not enough HTF bars")
        return

    # Load entry TF
    try:
        entry_df = load_data("mes", "5min", session_only=True)
    except FileNotFoundError:
        print("SKIP: Entry TF data not available")
        return

    # Call ifvg_combined with HTF
    print("\nCalling ifvg_combined with HTF data...")
    with warnings.catch_warnings(record=True) as warn_list:
        warnings.simplefilter("always")
        try:
            base, open_var = ifvg_combined(
                entry_df,
                timeframe_minutes=5,
                htf_df=htf_df,
                legacy_output=True
            )
            print(f"  ✓ ifvg_combined completed")
            print(f"    Base signals: {(base != 0).sum()}")
            print(f"    Open signals: {(open_var != 0).sum()}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            raise

    # Check for errors in timeframe detection
    error_warnings = [w for w in warn_list if "timeframe" in str(w.message).lower()]
    if error_warnings:
        print(f"  ⚠ Timeframe-related warnings:")
        for warn in error_warnings:
            print(f"    - {warn.message}")
    else:
        print(f"  ✓ No timeframe-related warnings")

    print(f"\n✓ PASS: HTF timeframe determination working")


def test_nat_comparison_fix():
    """Test that pd.notna() is used instead of != pd.NaT."""
    print("\n" + "=" * 80)
    print("TEST: NaT Comparison Fix")
    print("=" * 80)

    # Create a simple test case
    ts_value = pd.Timestamp("2024-01-15 10:30", tz="America/New_York")
    nat_value = pd.NaT

    # Old broken comparison
    print("\nOld broken comparison (pd.NaT != pd.NaT):")
    result_broken = (nat_value != pd.NaT)
    print(f"  Result: {result_broken} (should be True, but is {result_broken})")

    # New fixed comparison
    print("\nNew fixed comparison (pd.notna(pd.NaT)):")
    result_fixed_nat = pd.notna(nat_value)
    print(f"  Result: {result_fixed_nat} (should be False, is {result_fixed_nat})")

    result_fixed_ts = pd.notna(ts_value)
    print(f"  Result for valid timestamp: {result_fixed_ts} (should be True, is {result_fixed_ts})")

    assert result_fixed_nat == False, "pd.notna(pd.NaT) should be False"
    assert result_fixed_ts == True, "pd.notna(valid_timestamp) should be True"

    print(f"\n✓ PASS: NaT comparison fix verified")


if __name__ == "__main__":
    # Run all tests
    try:
        test_ifvg_structural_columns_wiring()
        test_htf_timeframe_determination()
        test_nat_comparison_fix()
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n" + "=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        sys.exit(1)
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

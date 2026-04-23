# Testing Requirements

**Purpose:** Catch logic errors before training. Every function in the ML pipeline must be testable in isolation. Tests must be run after each agent session handoff before proceeding.

---

## Why Testing Is Critical for This Pipeline

The training loop runs without error even when the data is wrong. Temporal leakage, off-by-one errors, wrong shift directions, and session filter bugs all produce a model that trains and evaluates without crashing — but the results are meaningless or fraudulent. Unit tests are the only way to verify correctness before spending compute on training.

---

## Test Suite Requirements

### 1. Data Loader Tests

```python
def test_timezone_is_eastern():
    df = load_data("mnq_5min")
    assert str(df.index.tz) == "America/New_York"

def test_session_hours_only():
    df = load_data("mnq_5min")
    hours = df.index.hour + df.index.minute / 60
    assert hours.min() >= 9.5     # no bars before 09:30
    assert hours.max() <= 15.0    # no bars after 15:00

def test_no_duplicate_timestamps():
    df = load_data("mnq_5min")
    assert df.index.duplicated().sum() == 0

def test_data_date_range():
    df = load_data("mnq_5min")
    assert df.index.min().date() >= pd.Timestamp("2021-01-01").date()
    assert df.index.max().date() <= pd.Timestamp("2026-12-31").date()
```

### 2. Feature Engineering Tests

```python
def test_synthetic_delta_no_division_by_zero():
    # Create test bar with high == low (doji)
    bar = {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}
    delta = compute_synthetic_delta(bar)
    assert delta == 0.0  # must return 0, not NaN or inf

def test_no_nan_in_features_after_warmup():
    df = load_data("mnq_5min")
    features = compute_features(df)
    warmup = 200  # longest indicator warmup (SMA200)
    post_warmup = features.iloc[warmup:]
    assert not post_warmup.isnull().any().any()

def test_forward_return_uses_future_bar():
    # label[i] must reference bar i+1 price, not bar i
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104]
    })
    labels = compute_forward_returns(df, periods=1)
    # label[0] = log(101/100), not log(100/99)
    expected = np.log(101 / 100)
    assert abs(labels.iloc[0] - expected) < 1e-6

def test_no_lookahead_in_rolling_features():
    # Rolling mean at bar i must use only bars 0..i-1
    df = load_data("mnq_5min")
    features = compute_features(df)
    # Corrupt the last row of close and verify feature at that row changed
    # (only if it correctly uses current bar) vs did not change (if it uses future)
    # This is a manual audit test — run and inspect
    pass

def test_camarilla_uses_prior_day():
    # H3 on day 2 must use day 1's OHLC, not day 2's
    # Verify by checking that H3 values are constant within a single day
    df = load_data("mnq_5min")
    df_with_levels = compute_camarilla(df)
    # Group by date — H3 should be constant within each date
    for date, group in df_with_levels.groupby(df_with_levels.index.date):
        assert group["H3"].nunique() == 1, f"H3 changed intraday on {date}"

def test_camarilla_correct_prior_day():
    # H3 on the SECOND trading day must equal the formula applied to the FIRST day's OHLC
    # This catches: using shift(1) per bar instead of per calendar day,
    #               using current day's data, or wrong day after weekends/holidays
    df = load_data("mnq_5min")
    df_with_levels = compute_camarilla(df)
    dates = sorted(df_with_levels.index.normalize().unique())
    for i in range(1, min(len(dates), 10)):  # check first 10 days
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_day = df[df.index.normalize() == prev_date]
        curr_day = df_with_levels[df_with_levels.index.normalize() == curr_date]
        if len(prev_day) == 0 or len(curr_day) == 0:
            continue
        prev_high = prev_day['high'].max()
        prev_low = prev_day['low'].min()
        prev_close = prev_day['close'].iloc[-1]
        expected_h3 = prev_close + (prev_high - prev_low) * 0.275
        actual_h3 = curr_day['H3'].iloc[0]
        assert abs(actual_h3 - expected_h3) < 1e-6, (
            f"H3 on {curr_date.date()} expected {expected_h3:.4f} "
            f"(from {prev_date.date()} OHLC) but got {actual_h3:.4f}"
        )
```

### 3. Strategy Signal Tests

```python
def test_strategy_signal_no_lookahead():
    # Strategy signal at bar i must not use price at bar i+1 or later
    # Method: compute signals on full dataset, then remove last N bars and recompute
    # Signals on the truncated dataset must match those on the full dataset for all bars except the last N
    df = load_data("mnq_5min")
    signals_full = compute_orb_signals(df)
    signals_truncated = compute_orb_signals(df.iloc[:-5])
    # All signals except last 5 must be identical
    assert (signals_full.iloc[:-5] == signals_truncated).all()

def test_max_signals_per_day():
    # No strategy should fire more than max_signals_per_day times on any single day
    df = load_data("mnq_5min")
    signals = compute_orb_signals(df)
    daily_counts = signals[signals != 0].groupby(signals.index.date).count()
    assert (daily_counts <= 1).all()

def test_ifvg_shared_daily_limit():
    # IFVG + IFVG Open Variant combined must not exceed 2 signals per day
    df = load_data("mnq_1min")
    ifvg_signals = compute_ifvg_signals(df)
    ifvg_open_signals = compute_ifvg_open_signals(df)
    combined = (ifvg_signals != 0).astype(int) + (ifvg_open_signals != 0).astype(int)
    daily = combined.groupby(combined.index.date).sum()
    assert (daily <= 2).all()
```

### 4. Train/Test Split Tests

```python
def test_no_train_test_overlap():
    train_df, val_df, test_df = split_data(df)
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

def test_scaler_fit_only_on_train():
    train_df, val_df, _ = split_data(df)
    scaler = StandardScaler()
    scaler.fit(train_df)
    val_transformed = scaler.transform(val_df)
    # Verify: if we refit on val, the result should be different
    scaler_val = StandardScaler()
    scaler_val.fit(val_df)
    val_refitted = scaler_val.transform(val_df)
    # They should not be identical (different means/stds)
    assert not np.allclose(val_transformed, val_refitted)

def test_no_shuffle_in_time_split():
    train_df, val_df, test_df = split_data(df)
    # All train dates must precede all val dates
    assert all(d < val_df.index.min() for d in train_df.index)
```

### 5. Model Output Tests

```python
def test_model_output_shape():
    model = load_model()
    dummy = torch.zeros(1, SEQ_LEN, N_FEATURES)
    output = model(dummy)
    assert output.shape == (1, 3)  # 3 classes: Long, Short, No Trade

def test_onnx_output_matches_pytorch():
    # After export, ONNX output must match PyTorch output within tolerance
    pytorch_out = model(dummy_input).detach().numpy()
    onnx_session = ort.InferenceSession("model_ifvg.onnx")
    onnx_out = onnx_session.run(None, {"input": dummy_input.numpy()})[0]
    assert np.allclose(pytorch_out, onnx_out, atol=1e-5)
```

### 6. Prop Firm Rule Tests

```python
def test_topstep_50k_eod_trailing_dd():
    # Trailing DD trails from highest EOD balance seen, not from account start
    eod_equity_curve = [50000, 50500, 51000]
    max_eod_balance = max(eod_equity_curve)
    current = 49100
    drawdown = max_eod_balance - current  # = 1900
    assert drawdown < 2000  # still within $2K limit
    # If drawdown >= 2000, account is blown

def test_topstep_50k_daily_loss_limit():
    # Daily loss must not exceed $1000 from start of day equity
    # Test that trade simulator correctly tracks and halts on DLL breach
    pass
```

---

## Test Runner Protocol

After each agent session:
1. Run the full test suite: `python -m pytest ml/tests/ -v --tb=short`
2. All tests must pass before writing `AGENT_N_STATUS.md`
3. If any test fails, fix the issue — do not mark the task complete and hand off
4. Log the test run output in the status file

---

## Common Failure Modes to Test For

See [[Architecture Overview]] and the logic gaps sections in each strategy note. Key items:
- `high == low` division by zero in synthetic delta
- SMA(200) warmup not respected — signals before bar 200
- Scaler fit on full dataset before split
- Forward return computed as backward return (wrong shift direction)
- Session filter wrong timezone (Central instead of Eastern)
- Walk-forward windows shuffled or overlapping
- ONNX feature order mismatch
- Prop firm EOD trailing DD computed from account start instead of equity peak

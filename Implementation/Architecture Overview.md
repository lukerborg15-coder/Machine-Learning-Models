# Architecture Overview

**Project:** Automated Prop Firm Payout Harvesting System
**Current Phase:** ML Training Layer (`ml` folder)
**Not in current scope:** FastAPI execution agent, React app
**Contingency only:** Revisit a Python inference server or webhook-based execution path only if the NinjaScript deployment path proves unworkable

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CURRENT SCOPE                        │
│                                                         │
│  Data (OHLCV CSVs)                                      │
│       ↓                                                 │
│  Feature Engineering (Python)                           │
│  - OHLCV features                                       │
│  - Synthetic delta                                       │
│  - Strategy signal labels (ORBs, IFVG, TTMSqueeze, etc) │
│  - Session level pivot features                         │
│       ↓                                                 │
│  Conv1D CNN — 4 Grouped Models (PyTorch)                │
│  - ALL BARS TRAINING (~98k 5min bars per model)         │
│  - NoTrade label for non-signal bars (class 2)          │
│  - 5 rolling walk-forward folds                         │
│  - Triple-barrier labels (stop/target/time)             │
│  - Purge + embargo at fold boundaries                   │
│       ↓                                                 │
│  ONNX Export (4 files, one per model group)             │
│  model_1.onnx / model_2.onnx / model_3.onnx / model_4.onnx │
│  + feature_order.json + scaler_params.json (per model)  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  FUTURE SCOPE                           │
│                                                         │
│  NinjaScript (NinjaTrader 8)                            │
│  - Loads 4 ONNX models via Microsoft.ML.OnnxRuntime     │
│  - Computes same features in real time                  │
│  - 50 MNQ max, dynamic confidence-based sizing          │
│  - Conflict resolution per NinjaScript Execution Rules  │
│  - 14:55 ET hard flat, daily loss + trailing DD guards  │
│                                                         │
│  No FastAPI required for the primary path               │
│  No external broker API required unless NinjaScript fails │
│  Prop firms are supported if they allow NinjaTrader     │
└─────────────────────────────────────────────────────────┘
```

---

## Data Layer

**Source:** Databento historical OHLCV bar data
**Instrument:** MNQ (Micro Nasdaq futures)
**Available timeframes:** 1min, 2min, 3min, 5min, 15min, 30min, 1H
**Date range:** 2021-03-19 to 2026-03-18
**File format:** CSV with columns `datetime,open,high,low,close,volume`
**Timezone in files:** Eastern time with UTC offset (`2021-03-19 09:30:00-04:00`)
**Future expansion:** `MES` data can be added later, but the current pipeline is specified and evaluated for `MNQ` only

**Loading:**
```python
df = pd.read_csv(path)
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert("America/New_York")
df = df.set_index('datetime')
# Session filter: 09:30–15:00 ET
df = df.between_time('09:30', '15:00')
```

---

## Feature Engineering Layer

Features fed to the CNN (per bar, windowed):

**Price/Volume:**
- open, high, low, close (normalized — see [[Tick Data - Delta Features]])
- volume (log-scaled)
- synthetic_delta = volume × (close - open) / (high - low) [guard: 0 if high==low]
- returns: log(close / close_lag1), log(close / close_lag5)
- ATR(14) normalized

**Strategy Signals (9 signal columns — ALL present in every bar's feature vector):**
- ifvg_signal: 1=long, -1=short, 0=none (IFVG)
- ifvg_open_signal: 1=long, -1=short, 0=none (IFVG Open Variant)
- connors_signal: 1=long, -1=short, 0=none (ConnorsRSI2)
- ttm_signal: 1=long, -1=short, 0=none (TTMSqueeze)
- orb_ib_signal: 1=long, -1=short, 0=none (ORB Initial Balance)
- orb_vol_signal: 1=long, -1=short, 0=none (ORB Volatility Filtered)
- orb_wick_signal: 1=long, -1=short, 0=none (ORB Wick Rejection)
- session_pivot_signal: 1=long, -1=short, 0=none (Session Level Pivots — Camarilla H3/H4/S3/S4 rejection/fade)
- session_pivot_break_signal: 1=long, -1=short, 0=none (Session Level Pivots — Camarilla H4/S4 close-through continuation)

**Session Level Features** (see [[Session Level Pivots]]):
- camarilla_h3/h4/s3/s4 distances (ATR-normalized)
- session high/low distances for Asia, London, Pre-Market, NY AM
- prev_day_high/low distances
- prev_week_high/low distances

**Time Features:**
- time_of_day (minutes since 09:30, normalized to [0,1])
- day_of_week (cyclical encoding via `dow_sin`, `dow_cos`)
- is_news_day (flag for major economic releases)

---

## Model Architecture

**Type:** Conv1D CNN (confirmed by CNN research survey — best for bar-based financial sequences)
**Input:** `(batch, sequence_length, n_features)` — windowed bars
**Output classes:** 3 — Long (0), Short (1), NoTrade (2)
**Architecture:**
```
ConstantPad1d((kernel_size - 1, 0), 0) → Conv1D(..., padding=0) → ReLU → BatchNorm
ConstantPad1d((kernel_size - 1, 0), 0) → Conv1D(..., padding=0) → ReLU → BatchNorm
GlobalAvgPool1D
Dense(128) → Dropout(0.3)
Dense(3)    # 3 classes: Long / Short / NoTrade
```

**Causal padding is mandatory** — prevents temporal leakage in the convolution.

**4 model groups — each trains on ALL session bars:**

| Model | Signal Columns | ONNX File |
|---|---|---|
| model_1 | ifvg_signal, connors_signal | model_1.onnx |
| model_2 | ifvg_open_signal, ttm_signal | model_2.onnx |
| model_3 | orb_ib_signal, session_pivot_signal | model_3.onnx |
| model_3 | orb_vol_signal, session_pivot_signal, session_pivot_break_signal | model_3.onnx |
| model_4 | orb_ib_signal, orb_wick_signal | model_4.onnx |

Each model receives the **full ~98,000 session bars** (5min). Non-signal bars are labeled NoTrade. Training is NOT restricted to signal-only rows. Class weights correct for extreme NoTrade imbalance.

---

## Walk-Forward Validation

**5 rolling folds** replace the original 2 anchored folds. Each fold has a train / val / test split with a purge gap + embargo between train and val to prevent label leakage.

| Fold | Train End | Val End | Test End |
|---|---|---|---|
| fold_1 | 2023-06-30 | 2023-12-31 | 2024-06-30 |
| fold_2 | 2023-12-31 | 2024-06-30 | 2024-12-31 |
| fold_3 | 2024-06-30 | 2024-12-31 | 2025-06-30 |
| fold_4 | 2024-12-31 | 2025-06-30 | 2025-12-31 |
| fold_5 | 2025-06-30 | 2025-12-31 | 2026-03-18 |

Aggregated OOS test: ~24 months. The deployment gate evaluates bootstrap statistics across all 5 folds' test results.

**Critical rule**: Scaler is fit ONLY on the training window of each fold. Validation and test windows are transformed with the training-window scaler. Scaler is never refit on val or test data.

**Purge gap:** Last N bars of train (where N = forward_horizon_bars) are removed to prevent label leakage at the train/val boundary.

---

## ONNX Export — 4 Files (One Per Model Group)

```
model_1.onnx  + feature_order_1.json  + scaler_params_1.json
model_2.onnx  + feature_order_2.json  + scaler_params_2.json
model_3.onnx  + feature_order_3.json  + scaler_params_3.json
model_4.onnx  + feature_order_4.json  + scaler_params_4.json
```

Export per model group:
```python
torch.onnx.export(
    model, dummy_input,
    f"ml/artifacts/model_{model_name}.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=12   # required for NT8 / .NET 4.8 compatibility
)
```

`feature_order_{model_name}.json` and `scaler_params_{model_name}.json` are the contracts with NinjaScript. The feature order and scaler mean/std must be hardcoded in C# exactly as exported — any mismatch silently corrupts inference.

---

## NinjaScript Integration (Future)

See [[NinjaScript Integration]] for full detail.

**Key points:**
- NinjaTrader bars are in Central time — must offset by +1 hour to match Eastern time session filters
- NuGet: `Microsoft.ML.OnnxRuntime` for ONNX inference in C#
- Feature order in NinjaScript must match `feature_order_{model_name}.json` exactly
- All 4 models run simultaneously; conflict resolution is handled per `NinjaScript Execution Rules.md`
- Max 50 MNQ total across all models; dynamic sizing via `ComputeContracts(stop_pts, confidence)`
- Prop firms that support NinjaTrader (TopStep, etc.) work natively — no API needed

---

## Prop Firm Target

**Primary:** TopstepTrader — 50K Combine Evaluation
See [[Prop Firm Rules]] for full TopStep 50K rules including the trailing drawdown and consistency rule.

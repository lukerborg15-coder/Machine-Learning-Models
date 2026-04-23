# Tick Data & Delta Features

**Related:** [[Architecture Overview]], [[ML Operators Guide]]

---

## Current Situation

Real tick data from Databento costs approximately $4,000 for the full 2021–2026 date range (2,500 GB at ~$28/GB for Trades data). This is too expensive for the initial build. We use a **synthetic delta approximation** computed from existing OHLCV data instead.

Real tick data can be added later (smaller date range, or recorded live via NinjaTrader/Rithmic) once the model architecture is validated.

---

## Synthetic Delta

### Formula

```python
def synthetic_delta(open_, high, low, close, volume):
    bar_range = high - low
    if bar_range == 0:
        return 0.0  # doji bar — no directional information
    directional_component = (close - open_) / bar_range
    return volume * directional_component
```

### Interpretation

- Positive value: buying pressure dominated — price closed near the top of the range
- Negative value: selling pressure dominated — price closed near the bottom of the range
- Zero: perfectly neutral candle, or doji (high == low)
- Magnitude: scales with volume — high-volume bars produce stronger signals

### Limitations vs. Real Delta

Real order flow delta measures the literal difference between trades executed at the ask (buyers) vs. trades at the bid (sellers). Synthetic delta is an approximation using price position within the bar:

| Metric | Synthetic Delta | Real Delta |
|---|---|---|
| Data required | OHLCV only | Tick-by-tick trades |
| Accuracy | Approximate | Exact |
| Misses | Iceberg orders, split candles | Nothing |
| Cost | Free (from existing data) | ~$4K for full history |
| Availability for live trading | Computed in real time from bar data | Requires tick feed |

Synthetic delta works well enough as a feature for CNN training when real delta is unavailable. Studies show it correlates significantly with real cumulative delta on liquid futures markets like ES/NQ.

---

## Normalization

Raw synthetic delta values vary enormously across bars (volume × directional component). Before feeding to the CNN:

```python
# Option 1: Normalize by volume (gives pure directional component)
directional_pct = (close - open_) / (high - low)  # range: roughly -1 to +1

# Option 2: Log-scale volume then combine
volume_log = np.log1p(volume)
delta_norm = volume_log * directional_pct

# Option 3: Z-score over rolling window
delta_zscore = (delta - rolling_mean(delta, 20)) / rolling_std(delta, 20)
```

Test all three normalizations as hyperparameters. Start with Option 1 (pure directional component) as the baseline.

---

## Edge Cases and Guards

**Doji bars (high == low):** Division by zero. Fix: check `if (high - low) == 0: return 0.0` before computing.

**Extended hours bars:** Synthetic delta on overnight bars has less meaning — very low volume, wider spreads. Session filter (09:30–15:00) already excludes these from training data.

**Large volume spikes:** On news events, volume can be 10–50× normal. Log-scaling volume reduces this impact. Without log-scaling, a single 10× volume bar dominates the feature range and distorts normalization.

---

## Getting Real Tick Data Later

### Option 1: Smaller Databento Purchase
Buy a shorter date range — e.g., 2024–2026 only (~$500–800) for validation/testing purposes. Use synthetic delta for the full training window (2021–2024) and real delta for recent data.

### Option 2: Record via NinjaTrader/Rithmic Live
Once live on a Rithmic-connected NinjaTrader account, use NinjaTrader's Market Replay or custom data recording to build a tick dataset going forward. Free, but requires waiting for history to accumulate.

### Option 3: Interactive Brokers Historical Data
IBKR provides tick data for futures with an active account. Shorter history than Databento but cheaper. Useful for model validation.

---

## Storing Real Delta (When Available)

Format:
```
datetime,bid_volume,ask_volume,delta,cum_delta,price
2024-01-02 09:30:01.234-05:00,124,89,−35,−35,16423.25
```

Aggregate tick delta to bar delta:
```python
# For each bar, sum all tick deltas within the bar's time range
bar_delta = tick_df.loc[bar_start:bar_end, 'delta'].sum()
```

Then use `bar_delta` directly instead of synthetic delta as the CNN input feature.

---

## CNN Feature Compatibility

Whether using synthetic or real delta, the feature column name is the same in the feature matrix: `delta`. The CNN architecture does not change. Swapping real for synthetic delta requires only updating `dataset_builder.py` — the model, training loop, and NinjaScript code are unchanged.

This is intentional — the feature interface is stable regardless of data source.

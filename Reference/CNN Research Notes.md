# CNN Research Notes

**Source:** "Convolutional Neural Networks for Day Trading and Financial Time Series" (research survey PDF)
**Relevance:** Confirms architecture choices and identifies the most dangerous failure modes for this project

---

## Key Findings

### Architecture: Conv1D is Correct for Bar Data

The survey confirms **Conv1D (1D convolutional networks) are the best architecture for bar-based financial sequences** — better than LSTM/RNN for most financial time series tasks because:
- Conv1D learns local patterns (e.g., 3-bar or 5-bar formations) independent of position in the sequence
- No vanishing gradient problems on long sequences
- Faster to train than LSTM
- Causal padding makes temporal correctness enforceable at the architecture level

**Our choice (Conv1D CNN) is validated.**

### Confirmed: Walk-Forward Validation is Required

Standard random K-fold cross-validation is inappropriate for time series. Walk-forward validation (or expanding window) is the correct method. The survey found that papers using random splits systematically overestimated performance.

**Our train 2021–2024 / val 2025 / test 2026 split is the correct approach.**

### Dual Evaluation Metric Is Required

Papers that only report classification metrics (accuracy, F1) without trading metrics (Sharpe, profit factor) are unreliable. A model can achieve high F1 by predicting the majority class and still lose money in simulation. The survey recommends always reporting both.

**Our evaluation phase (Phase E) requires both classification AND trading metrics.**

---

## Critical Failure Modes (From Survey)

These are the most commonly identified failure modes in CNN financial trading papers that produced fraudulent backtests:

### 1. Temporal Leakage (Most Common — Silent Failure)
**What it is:** Future data contaminating training data. Examples:
- StandardScaler fit on the entire dataset before train/test split
- Rolling features computed using future bars (e.g., `rolling(20)` that looks forward)
- Labels computed incorrectly (using current bar price instead of future bar price)

**Why it's dangerous:** The model appears to work extremely well in backtests, producing near-perfect results. There is no error message. The only way to catch it is with explicit unit tests.

**Our fix:** Documented in [[Testing Requirements]]:
- Scaler always fit on train set only
- All rolling features are trailing (backward-looking only)
- Labels use `shift(-N)` verified to be forward-looking
- Test: truncate dataset by 5 bars, recompute features, verify earlier bars are identical

### 2. Data Snooping / Overfitting to Backtest

**What it is:** Testing many hyperparameter combinations on the validation set and selecting the best one — the final selected config is overfit to the validation set even though it was never "directly" trained on it.

**Our fix:**
- Hyperparameter search uses validation set, but test set is held out entirely until final evaluation
- Report results on the held-out 2026 test set as the final performance number
- If test performance is significantly worse than validation, the model is overfit

### 3. Nonstationarity / Regime Shift

**What it is:** Financial time series are nonstationary — their statistical properties change over time (volatility regimes, market structure changes, correlation breakdowns). A model trained on 2021 low-volatility markets may fail in 2022 high-volatility markets.

**Our mitigation:**
- Walk-forward training covers multiple market regimes (2021–2024 includes COVID recovery, 2022 bear market, 2023–2024 bull market)
- ATR-normalized features help with stationarity (price in ATR units rather than raw points)
- The ML model must be re-trained periodically (at minimum quarterly) with recent data

### 4. Under-Modeled Transaction Costs

**What it is:** Backtests that ignore slippage, commissions, and bid-ask spread. A strategy that looks profitable at $0 cost may be unprofitable at realistic costs.

**Realistic costs for MNQ trading:**
- Commission: ~$0.35–$0.50 per side per contract (Rithmic)
- Slippage: 0.25–1 tick average on MNQ (0.25 points = $0.50 per contract)
- Bid-ask spread: usually 0.25 ticks on MNQ during RTH
- Total round-trip per contract: approximately $1.50–$2.50

**Our fix:** The trading simulator must include per-trade cost:
```python
COMMISSION_PER_SIDE = 0.45   # USD per contract
SLIPPAGE_TICKS = 1           # 1 tick = 0.25 points = $0.50 on MNQ
MNQ_TICK_VALUE = 0.50

cost_per_trade = (COMMISSION_PER_SIDE * 2) + (SLIPPAGE_TICKS * MNQ_TICK_VALUE)
# = $0.90 commission + $0.50 slippage = $1.40 round trip per contract
```

At 5 contracts per trade: $7.00 round trip. On a $150 target (5 MNQ, 15-point target): 4.7% friction. Not negligible.

### 5. Survivorship Bias

Not applicable for futures trading a single instrument (`MNQ` in the current project scope) — there is no survivorship bias when the instrument itself is the product.

---

## Recommended Reporting Standards

Based on the survey, a credible backtest must report:

**Per strategy, per timeframe:**
- Number of signals generated
- Win rate
- Average R (return per trade in risk units)
- Profit factor
- Sharpe ratio (annualized, after costs)
- Maximum drawdown (%)
- Calmar ratio (annualized return / max drawdown)
- **TopStep 50K Combine pass rate** (simulated across all walk-forward windows)
- Average Combine duration (days to reach $3K profit target)
- Consistency rule violation rate

**Model evaluation:**
- F1 (macro)
- ROC-AUC
- Confusion matrix (Long / Short / No Trade predictions vs. actuals)
- Performance delta: validation vs. test (large gap = overfit)

---

## Notes for Implementation

- The survey notes that Conv1D with causal padding is "the minimum viable temporal correctness guarantee" — padding='same' or 'valid' both allow some form of future leakage depending on implementation
- The survey recommends at least 3 years of training data for daily pattern generalization — our 2021–2024 window provides this
- The survey explicitly warns against using the same validation set for both model selection AND hyperparameter search — use nested cross-validation or a separate tuning set if compute allows

# NinjaScript Integration

**Status:** Future scope — not built yet
**Dependency:** Requires a trained `model_{strategy}.onnx` from the ML training phase
**Related:** [[Architecture Overview]], [[Prop Firm Rules]]

---

## Overview

After the CNN is trained and exported to ONNX, the execution layer is a NinjaScript indicator/strategy that:
1. Computes the same features the model was trained on in real time
2. Feeds them to the ONNX model at each bar close
3. Executes the trade if the model outputs a Long or Short signal above confidence threshold
4. Applies TopStep 50K risk rules (DLL, EOD trailing DD, max contracts)

NinjaTrader handles the broker connection natively for supported firms (Apex uses Rithmic, which NinjaTrader supports out of the box).

---

## ONNX Runtime Setup

**NuGet package:** `Microsoft.ML.OnnxRuntime`

Install via NinjaTrader's NuGet manager or manually add to NinjaTrader's reference folder.

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

private InferenceSession onnxSession;

protected override void OnStateChange()
{
    if (State == State.SetDefaults)
    {
        // Load model once at startup
        string modelPath = @"C:\path\to\model_ifvg.onnx";
        onnxSession = new InferenceSession(modelPath);
    }
}
```

---

## Critical: Timezone Alignment

**Problem:** NinjaTrader serves bars in the exchange's native timezone (Chicago/Central time for CME futures via Rithmic). The training data was labeled and filtered using **Eastern time** (America/New_York).

**Impact:** Without correction, the session filter `09:30–15:00 ET` applied in NinjaScript using CT hour values would be off by 1 hour, causing:
- Trades executing during pre-market or post-close
- Session-based features (NY AM high/low, time_of_day) computed from wrong time boundaries
- All time-encoded features shifted, producing garbage model inputs

**Fix:** Apply a +1 hour offset when converting bar times in NinjaScript:
```csharp
DateTime barTimeET = Time[0].AddHours(1);  // Convert CT to ET
int hourET = barTimeET.Hour;
int minuteET = barTimeET.Minute;

// Session gate
if (hourET < 9 || (hourET == 9 && minuteET < 30)) return;
if (hourET >= 15) return;
```

Note: During Daylight Saving Time transitions (March and November), verify the offset. CDT is UTC-5, EDT is UTC-4 — the offset between them is always exactly 1 hour regardless of DST transitions, so `+1 hour` is correct year-round for CT→ET conversion.

---

## Feature Order Contract

**This is the single most important integration requirement.**

The ONNX model expects features in a specific column order. If NinjaScript feeds features in a different order, the model produces wrong outputs silently — no error is thrown.

At ONNX export time, the Python training code must output `feature_order_{strategy}.json`:
```json
[
  "open_norm", "high_norm", "low_norm", "close_norm",
  "volume_log", "synthetic_delta",
  "return_1", "return_5",
  "atr_norm",
  "orb_vol_signal", "orb_wick_signal", "orb_ib_signal",
  "ifvg_signal", "ifvg_open_signal",
  "ttm_signal", "connors_signal",
  "session_pivot_signal", "session_pivot_break_signal",
  "h3_dist", "h4_dist", "s3_dist", "s4_dist",
  "h3_above", "h4_above", "s3_above", "s4_above",
  "asia_high_dist", "asia_low_dist",
  "london_high_dist", "london_low_dist",
  "premarket_high_dist", "premarket_low_dist",
  "ny_am_high_dist", "ny_am_low_dist",
  "prev_day_high_dist", "prev_day_low_dist",
  "prev_week_high_dist", "prev_week_low_dist",
  "time_of_day",
  "dow_sin", "dow_cos",
  "is_news_day"
]
```

NinjaScript must construct the input tensor with features in exactly this order. **Hard-code the array indices** — do not rely on dynamic ordering.

---

## Inference at Bar Close

```csharp
protected override void OnBarUpdate()
{
    if (BarsInProgress != 0) return;
    if (CurrentBar < WARMUP_BARS) return;  // wait for indicators to warm up

    // Compute the latest feature vector and append it to a rolling buffer.
    float[] latestFeatureVector = ComputeFeatures();
    sequenceBuffer.Push(latestFeatureVector);
    if (sequenceBuffer.Count < SEQ_LEN) return;

    // Flatten the rolling buffer into (1, SEQ_LEN, N_FEATURES)
    float[] flattened = sequenceBuffer.ToRowMajorArray();
    var inputTensor = new DenseTensor<float>(
        flattened, new int[] { 1, SEQ_LEN, N_FEATURES });

    var inputs = new List<NamedOnnxValue> {
        NamedOnnxValue.CreateFromTensor("input", inputTensor)
    };

    using var results = onnxSession.Run(inputs);
    float[] logits = results.First().AsEnumerable<float>().ToArray();

    // logits[0] = Long, logits[1] = Short, logits[2] = No Trade
    int prediction = ArgMax(logits);
    float confidence = Softmax(logits)[prediction];

    if (confidence > CONFIDENCE_THRESHOLD)
    {
        if (prediction == 0) EnterLong();
        else if (prediction == 1) EnterShort();
    }
}
```

---

## Risk Management in NinjaScript (Topstep 50K Rules)

See [[Prop Firm Rules]] for the full Topstep 50K rule set.

```csharp
// Track the highest END-OF-DAY balance reached, not the highest intraday balance.
private double maxEodBalance = 50000;

// EOD Trailing Drawdown check
double currentEquity = Account.Get(AccountItem.CashValue, Currency.UsDollar);
double trailingDD = maxEodBalance - currentEquity;
if (trailingDD >= 2000)
{
    CloseAllPositions();  // Liquidate
    IsEnabled = false;    // Disable strategy for the day
    return;
}

// Update maxEodBalance only once the session is closed and the final EOD balance is known.
if (Bars.IsLastBarOfSession)
{
    maxEodBalance = Math.Max(maxEodBalance, currentEquity);
    startOfDayEquity = currentEquity;
}

// Daily Loss Limit check
double dailyPnL = currentEquity - startOfDayEquity;
if (dailyPnL <= -1000)
{
    CloseAllPositions();
    IsEnabled = false;
    return;
}

// Max contracts
if (Position.Quantity >= 5)
{
    // Do not add more — at TopStep 50K MNQ position limit (5 contracts)
}
```

**Important:** The trailing drawdown floor must be based on the highest **end-of-day** balance reached. Do not replace it with an intraday high-water mark.

---

## Testing Before Live

1. Run ONNX inference test: load `model_{strategy}.onnx` in Python and verify output shape and values
2. Verify feature order in NinjaScript matches `feature_order_{strategy}.json` exactly
3. Backtest NinjaScript strategy against known historical data — results should approximately match Python backtest
4. Paper trade for at least 1 week before live deployment
5. Verify timezone alignment: manually check that first bar of session in NinjaScript is at 09:30 ET, not 09:30 CT

---

## Notes

- NinjaTrader is supported by Apex Funded, TopStepTrader, and other major prop firms that use Rithmic or NinjaTrader-native brokers
- No custom API integration is needed — NinjaTrader manages the broker connection
- The NinjaScript strategy file should be version-controlled alongside `model_{strategy}.onnx` and `feature_order_{strategy}.json`

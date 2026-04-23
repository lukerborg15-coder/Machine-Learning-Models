# Final Evaluation Report - Agent 3D

**Objective:** meta-label binary win/loss on strategy signal bars.

## Sharpe By Confidence Threshold

| Strategy | Old 3-class Sharpe | AUC-ROC | Brier | Sharpe @0.50 | Trades @0.50 | Sharpe @0.55 | Trades @0.55 | Sharpe @0.60 | Trades @0.60 | Sharpe @0.65 | Trades @0.65 | Sharpe @0.70 | Trades @0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ifvg | 0.271 | 0.530 | 0.253 | -1.229 | 90 | 0.060 | 48 | 1.107 | 23 | -0.879 | 3 | -1.461 | 2 |
| ifvg_open | -3.819 | 0.569 | 0.255 | -0.119 | 5 | 6.323 | 0 |  | 0 |  | 0 |  | 0 |
| orb_ib | 0.000 | 0.564 | 0.249 |  | 23 |  | 9 |  | 2 |  | 1 |  | 0 |
| orb_vol | 0.000 | 0.494 | 0.301 |  | 54 |  | 45 |  | 26 |  | 17 |  | 8 |
| orb_wick | 0.000 | 0.505 | 0.260 |  | 61 |  | 25 |  | 1 |  | 0 |  | 0 |
| orb_va | 0.000 | 0.429 | 0.255 |  | 31 |  | 1 |  | 0 |  | 0 |  | 0 |
| ttm | 0.389 | 0.557 | 0.255 | 0.120 | 67 | -0.020 | 34 | 0.656 | 16 | 1.336 | 10 | 0.991 | 2 |
| connors | -1.291 | 0.494 | 0.258 | 0.189 | 131 | -0.257 | 61 | 0.610 | 24 | -0.970 | 5 |  | 0 |

## Notes

- Sharpe and trade counts are medians across rolling OOS folds.
- Confidence thresholds gate meta-label win probability before trade simulation.

import pandas as pd
import os
strats = ["ifvg","ifvg_open","connors","ttm","orb_ib","orb_vol","orb_wick"]
print(f'{"strategy":<22} {"avg_sharpe":<12} {"avg_trades":<12} {"best":<10} {"worst":<10}')
print("-" * 70)
for s in strats:
    f = f"ml/artifacts/eval_{s}.csv"
    if not os.path.exists(f):
        print(f"{s:<22} MISSING")
        continue
    df = pd.read_csv(f)
    per_fold = df[df["fold"] != "summary"] if "fold" in df.columns else df
    sharpe = per_fold["test_sharpe"].mean()
    trades = per_fold["trade_count"].mean()
    best = per_fold["test_sharpe"].max()
    worst = per_fold["test_sharpe"].min()
    print(f"{s:<22} {sharpe:<12.3f} {trades:<12.1f} {best:<10.2f} {worst:<10.2f}")

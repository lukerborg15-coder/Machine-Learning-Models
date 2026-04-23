# NEXT_STEPS.md - Strat Lab Handoff

**Date:** 2026-04-22
**Status:** Pipeline working end-to-end on 5min only. No deployable models.
**Author:** Claude (working with Luke). This file = source of truth for what to do next on a different machine.

---

## TL;DR for the next AI / engineer

Picking up a futures (MNQ) ML trading pipeline. Pipeline runs but produces no deployable models. Two big problems:

1. **Performance:** A single 5min build takes ~95 minutes (86 min in `_compute_signal_features` alone). Building 1min would take 30+ hours unless we vectorize. Bottlenecks: per-row Python loops in `Implementation/camarilla_pivot_generator.py` (`compute_session_levels`) and IFVG sweep detection.

2. **Coverage:** Models are configured for 5min entry only, but several strategies (IFVG, IFVG_open, Connors, TTM, session_pivot, session_pivot_break) are designed for multiple entry timeframes. ORB family stays 5min by spec.

Fix 1 first. Then expand to multi-TF. Then re-evaluate.

---

## Pipeline state (verified 2026-04-22)
py -3.11 ml/dataset_builder.py --rebuild --instrument mnq --timeframe 5min
py -3.11 ml/train.py
py -3.11 ml/evaluate.py
py -3.11 ml/funded_sim.py

- 5min build: ~95 min, produces `ml/data/features_mnq_5min.parquet` (~27MB, 85k rows)
- Training: ~17 min total on GTX 1660 SUPER (4 models x 5 walk-forward folds)
- Eval + sim: fast (<2 min)

All 4 patched files (`dataset_builder.py`, `train.py`, `evaluate.py`, `funded_sim.py`) have:
- Module-level `mkdir(parents=True, exist_ok=True)` for output dirs
- `dataset_builder.py` has stage timing prints

---

## Current results (NO commission/slippage applied)

### Group models (4 trained)

| Model | Strategies | Avg Sharpe | Sharpe range | Combine pass | Avg trades/fold |
|---|---|---|---|---|---|
| model_1 | ifvg + connors | -1.48 | -3.62 to +0.25 | 0% | 60 |
| model_2 | ifvg_open + ttm | -0.93 | -2.42 to +1.86 | 0% | 55 |
| model_3 | orb_vol + session_pivot + session_pivot_break | -0.14 | -1.18 to +0.70 | 20% | 95 |
| model_4 | orb_ib + orb_wick | +0.13 | -1.20 to +1.71 | 20% | 72 |

**None deployable.** Best (model_4) is essentially random.

### Raw strategies (stale baselines)

| Strategy | Avg Sharpe | Avg trades/fold | Notes |
|---|---|---|---|
| ifvg | -0.05 | 10.2 | Tiny sample |
| connors | -0.10 | 17.0 | Negative |
| ttm | **+1.00** | 6.6 | **All 5 folds positive (+0.20 to +3.01)** but tiny sample |
| ifvg_open | NaN | 0 | Eval broken |
| orb_ib | NaN | 0 | Eval broken |
| orb_vol | NaN | 0 | Eval broken |
| orb_wick | NaN | 0 | Eval broken |

**TTM is the only signal of interest.**

---

## Phase 3 - Vectorize slow code (DO FIRST)

### Bottleneck #1: `compute_session_levels` (9 min)

**File:** `Implementation/camarilla_pivot_generator.py`, around line 166.

**Pattern:**
```python
for day in days:
    day_mask = df.index.normalize() == day  # O(n) scan per day
```

For 85k rows x ~1300 days = ~110M operations.

**Fix:**
```python
day_groups = df.groupby(df.index.normalize())
for day, group_df in day_groups:
    # work on group_df directly
```

`groupby` is O(n) total. Should drop 9 min -> <30 sec.

### Bottleneck #2: `_compute_signal_features` (86 min)

**File:** `ml/dataset_builder.py`, calls into `Implementation/`. Likely culprit: IFVG sweep detection.

**Find it:**
```powershell
Get-ChildItem -Path Implementation -Filter "*.py" -Recurse | Select-String -Pattern "def ifvg_combined|def _detect_sweep"
```

**Pattern to fix:** nested loops over bars. Replace with vectorized rolling-window numpy ops.

### Validation

Pin output of slow version BEFORE refactor:
```python
df_slow = build_feature_matrix("mnq", "5min")
df_slow[["ifvg_signal", "ifvg_open_signal"]].to_parquet("regression_baseline.parquet")
```

After refactor:
```python
assert df_fast["ifvg_signal"].equals(df_baseline["ifvg_signal"])
assert df_fast["ifvg_open_signal"].equals(df_baseline["ifvg_open_signal"])
```

If different, rollback. Don't ship semantics changes silently.

### Success: Phase 3

- 5min build <10 min (down from 95 min)
- 1min build feasible <2 hrs
- Regression test passes

---

## Phase 2 - Multi-TF expansion (AFTER Phase 3)

### 2.1: Build missing parquets

```powershell
py -3.11 ml/dataset_builder.py --rebuild --instrument mnq --timeframe 1min
py -3.11 ml/dataset_builder.py --rebuild --instrument mnq --timeframe 2min
py -3.11 ml/dataset_builder.py --rebuild --instrument mnq --timeframe 3min
```

### 2.2: Expand `MODEL_GROUPS` in `ml/train.py`

Per Luke's call: ORB family stays 5min. Everything else gets multiple entry TFs.

```python
MODEL_GROUPS = [
    # IFVG family - 1min, 2min, 3min, 5min
    {"model_name": "ifvg_1min",       "strategies": ["ifvg"],       "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "ifvg_2min",       "strategies": ["ifvg"],       "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "ifvg_3min",       "strategies": ["ifvg"],       "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "ifvg_5min",       "strategies": ["ifvg"],       "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    {"model_name": "ifvg_open_1min",  "strategies": ["ifvg_open"],  "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "ifvg_open_2min",  "strategies": ["ifvg_open"],  "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "ifvg_open_3min",  "strategies": ["ifvg_open"],  "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "ifvg_open_5min",  "strategies": ["ifvg_open"],  "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    # Connors
    {"model_name": "connors_1min",    "strategies": ["connors"],    "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "connors_2min",    "strategies": ["connors"],    "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "connors_3min",    "strategies": ["connors"],    "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "connors_5min",    "strategies": ["connors"],    "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    # TTM - investigate (only edge candidate)
    {"model_name": "ttm_1min",        "strategies": ["ttm"],        "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "ttm_2min",        "strategies": ["ttm"],        "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "ttm_3min",        "strategies": ["ttm"],        "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "ttm_5min",        "strategies": ["ttm"],        "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    # Session pivots
    {"model_name": "spivot_1min",       "strategies": ["session_pivot"],       "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "spivot_2min",       "strategies": ["session_pivot"],       "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "spivot_3min",       "strategies": ["session_pivot"],       "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "spivot_5min",       "strategies": ["session_pivot"],       "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    {"model_name": "spivot_break_1min", "strategies": ["session_pivot_break"], "timeframe": "1min", "parquet": "ml/data/features_mnq_1min.parquet"},
    {"model_name": "spivot_break_2min", "strategies": ["session_pivot_break"], "timeframe": "2min", "parquet": "ml/data/features_mnq_2min.parquet"},
    {"model_name": "spivot_break_3min", "strategies": ["session_pivot_break"], "timeframe": "3min", "parquet": "ml/data/features_mnq_3min.parquet"},
    {"model_name": "spivot_break_5min", "strategies": ["session_pivot_break"], "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},

    # ORB family - 5min only (matches spec)
    {"model_name": "orb_ib_5min",   "strategies": ["orb_ib"],   "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},
    {"model_name": "orb_vol_5min",  "strategies": ["orb_vol"],  "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},
    {"model_name": "orb_wick_5min", "strategies": ["orb_wick"], "timeframe": "5min", "parquet": "ml/data/features_mnq_5min.parquet"},
]
```

Total: ~27 models. Will take ~1.5 hrs to train on GPU.

### 2.3: Re-train + re-evaluate + rank

```powershell
py -3.11 ml/train.py
py -3.11 ml/evaluate.py
```

Rank by Sharpe + sample + consistency. Top 3-5 = candidates.

---

## Phase 1.1 - Cost model (DEFERRED)

Luke deferred. Documenting:

When ready: subtract per-trade cost from `pnl`. Apex MNQ approx: $2.68 commission + $0.50-1 slippage = ~$3.20/trade. Recompute Sharpe.

**Risk of deferring:** more trades per multi-TF = more invisible cost drag. Models that look profitable raw may be net negative.

---

## Setup on new machine

### Prereqs
- Python 3.11
- CUDA GPU (recommended for training)
- Git
- ~5GB disk

### Install
```powershell
git clone https://github.com/lukerborg15-coder/Machine-Learning-Models.git
cd Machine-Learning-Models
py -3.11 -m pip install -r requirements.txt --break-system-packages
```

### Data (NOT in repo)

Copy from Luke's machine or fetch from Databento:
- `data/mnq_1min_databento.csv` (~118MB)
- `data/mnq_2min_databento.csv` (~60MB)
- `data/mnq_3min_databento.csv` (~40MB)
- `data/mnq_5min_databento.csv` (~24MB)
- `data/mnq_15min_databento.csv` (~8MB)
- `data/mnq_30min_databento.csv` (~4MB)
- `data/mnq_1h_databento.csv` (~2MB)

### Verify
```powershell
py -3.11 test_mkdir_guards.py
```
Should print `ALL CHECKS PASSED`.

---

## Open issues

1. **4 raw evaluators broken** - `eval_ifvg_open.csv`, `eval_orb_*.csv` show 0 trades.
2. **Threshold sweep cols missing** in `eval_model_*.csv`. Code references them but they're absent. Fix - free signal.
3. **Meta-label may be over-aggressive** at default 0.5 threshold.
4. **MODEL_GROUPS mismatch with strategy specs** - Phase 2 fixes.
5. **No regression tests** on feature semantics. Add baseline parquets before any refactor.
6. **No requirements.txt** - generate before pushing if missing.
7. **Bottleneck file not pinpointed** for IFVG sweep - grep on actual repo.

---

## Decisions (don't relitigate without good reason)

- Cost model deferred (Phase 1.1) - Luke's call
- ORB stays 5min - spec compliance
- One strategy per model (not grouped) in v2
- Don't run `funded_sim.py` until models show edge

---

## Timing budget Phase 3 + Phase 2

| Task | Est. time |
|---|---|
| Find IFVG sweep file | 30 min |
| Vectorize compute_session_levels | 1-2 hrs |
| Vectorize IFVG sweep | 3-5 hrs |
| Regression test | 1 hr |
| Build 1/2/3min parquets | 4-8 hrs CPU |
| Update MODEL_GROUPS + retrain ~27 models | 2-3 hrs GPU |
| Eval + rank | 1 hr |
| **Total** | **~2 working days** |

---

## Contact

Luke owns this. Decisions go through him.

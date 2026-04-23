# Master Change Plan
**Date:** 2026-04-16
**Status:** Planning — no code or MD files have been written yet. This document must be read and confirmed before any agent begins work.

---

## Why This Change Is Happening

The Agent 3A/3B run produced zero deployment candidates. Three root causes identified:

1. **Models were starved of data.** Training was gated to only rows where a strategy's signal fired (~750–3,000 rows per model). Each model was making decisions on less than 3% of the available data.
2. **Session Level Pivots was never a strategy.** It existed only as feature engineering — producing levels as input columns but never generating its own signals. Edge was left on the table.
3. **Contract limit was wrong.** TopStep 50K allows 50 MNQ contracts (equivalent to 5 NQ mini). The risk simulator, position sizing tables, and stop calculations were based on 5 MNQ.

---

## The Single Most Important Architectural Change

**Every model trains on every bar in the dataset.**

This must be enforced explicitly throughout all agent specs, all code changes, and all audit checks. It is the foundation of everything else.

Current (wrong):
```
training rows per model = only bars where this strategy's signal != 0
result = 750 to 3,000 rows per model
```

New (correct):
```
training rows per model = ALL session bars in the full parquet
result = ~98,000 rows per model (5min timeframe)
         ~390,000 rows per model (1min timeframe)
```

How labels work on all bars:
```
Signal bar where strategy won  → Long (0) or Short (1) via triple-barrier
Signal bar where strategy lost → NoTrade (2) via triple-barrier
Non-signal bar                 → NoTrade (2) always
```

The CNN sees the full market — every bar, every regime, every time of day — for all 5 years of data. It learns what the market looks like when setups succeed AND what it looks like when they fail AND what it looks like when nothing is happening. Non-signal bars are not noise. They are the model learning when NOT to trade even if a signal fires.

No agent spec, no code change, and no audit check may gate training rows by signal activity. Any filter that reduces the training pool below the full session-bar count is a bug.

---

## Contract Limit Correction

**TopStep 50K Combine: 50 MNQ contracts maximum**
- MNQ point value: $2/point
- 50 MNQ × $2 = $100/point total exposure
- This is dollar-equivalent to 5 NQ (mini) contracts

Position sizing per trade is NOT 50 contracts. 50 is the hard ceiling. Actual per-trade sizing is confidence-based:

| Model confidence | Contracts per trade |
|---|---|
| 0.60–0.65 | 10 MNQ |
| 0.65–0.70 | 20 MNQ |
| 0.70–0.80 | 30 MNQ |
| 0.80+ | 50 MNQ |
| < 0.60 | No trade |

This must replace the fixed "5 MNQ contracts" assumption everywhere it appears.

---

## Model Architecture

**4 models, each trained on the full dataset. 9 strategies total.**

Signal columns tell each model which strategies it is responsible for. During training, signal-bar rows get triple-barrier labels. Non-signal rows get NoTrade. The model learns from all of them.

| Model | Strategies | Signal columns active | All other bars |
|---|---|---|---|
| Model 1 | IFVG + ConnorsRSI2 | ifvg_signal, connors_signal | NoTrade |
| Model 2 | IFVG Open + TTMSqueeze | ifvg_open_signal, ttm_signal | NoTrade |
| Model 3 | ORB Vol + Session Pivot Rejection + Session Pivot Break | orb_vol_signal, session_pivot_signal, session_pivot_break_signal | NoTrade |
| Model 4 | ORB IB + ORB Wick | orb_ib_signal, orb_wick_signal | NoTrade |

**ONNX export:** 4 files (model_1.onnx through model_4.onnx). Each model covers its strategy group. NinjaScript calls the correct model when its strategy fires.

**Why these pairings:**
- Model 1: trend-following sweep vs mean reversion RSI fade — opposite logic, minimal signal overlap
- Model 2: 9:30 open manipulation vs any-time volatility compression — different windows, different conditions
- Model 3: time-range breakout vs level rejection/bounce — structural opposites, both use pivot levels as context
- Model 3: volume breakout (ORBVol) + level fade (Pivot Rejection) + level continuation (Pivot Break) — all three read price behavior at structure from different angles
- Model 4: initial balance break + wick-filtered clean break — pure range breakout pair, no level dependency

---

## Files to CREATE

### Agent Specs (new)

**`Agents/Agent 4A - Pivot Features and Session Signal.md`**
- Wire `Implementation/camarilla_pivot_generator.py` into `ml/dataset_builder.py`
- Add `session_pivot_signal()` function to `ml/signal_generators.py`
- Rebuild all 4 feature parquets
- Verify pivot columns are non-zero and non-NaN in rebuilt parquets
- Verify session_pivot_signal fires on real data with correct directionality

**`Agents/Agent 4A-Audit - Pivot and Signal Review.md`**
- Independent parquet inspection: print describe() on all pivot columns
- Verify camarilla H3/H4/S3/S4 values match hand-calculated values from prior day OHLC
- Verify session_pivot_signal fires long at S4 touches and short at H4 touches
- Verify signal respects max 2/day cap
- Run full test suite

**`Agents/Agent 4B - Grouped Model Architecture.md`**
- Remove signal-bar gating from `dataset_builder.py` — ALL session bars go into training
- Add NoTrade label (class 2) for non-signal bars explicitly
- Update `train.py` TRAINING_JOBS from 8 separate to 4 grouped model configs
- Each model config specifies which signal columns it is responsible for
- Update `topstep_risk.py`: max_contracts = 50, confidence-based sizing table
- Update `funded_sim.py`: confidence-based position sizing
- Verify each model receives ~98,000 training rows on 5min data (not 750–3,000)
- Update 5-fold walk-forward config (from Agent 3C spec)

**`Agents/Agent 4B-Audit - Architecture Review.md`**
- Print training row count per model per fold — must be ~98,000 for 5min, ~390,000 for 1min
- Print NoTrade label percentage — must be 95%+ (verifies non-signal bars are included)
- Verify class weights are computed from actual label distribution, not hardcoded
- Verify max_contracts = 50 in topstep_risk.py
- Verify confidence tiers produce correct contract counts
- Verify no signal-bar filter exists anywhere in dataset_builder.py or train.py
- Run full test suite

**`Agents/Agent 3C-Audit - Rolling Walk Forward Review.md`**
- Independent fold overlap check (scripted, not eyeballed)
- Purge gap removes leaking label rows at fold boundaries
- Embargo size matches timeframe (5min=78, 3min=130, 1min=390)
- Deployment gate requires all 5 folds positive
- Training row count still ~98,000 per fold (all bars, not signal bars)
- Run full test suite

**`Agents/Agent 3D-Audit - Label and Objective Review.md`**
- Synthetic path tests: stop-hit-first, target-hit-first, short symmetry, same-bar conflict
- Session boundary truncation at 15:00
- Label sign matches trade sim P&L sign on 20 real signal bars
- Transaction cost subtracted in vertical barrier case
- Non-signal bars still labeled NoTrade (triple-barrier does not overwrite them)
- Run full test suite

**`Agents/Agent 3E-Audit - Statistical Review.md`**
- Stationary bootstrap wider than iid bootstrap on autocorrelated series
- Pass-rate resampled as day-sequences not individual trades
- Infinite value handling (profit factor, Calmar)
- Gate rejects on 4 specific mock cases
- Multiple-testing correction documented
- p50 Sharpe >= 0.5 second filter present
- Run full test suite

### Implementation Docs (new)

**`Implementation/NinjaScript Execution Rules.md`**
- Conflict resolution hierarchy (full decision tree)
- Two strategies fire same bar, same direction: both execute up to 50-contract ceiling
- Two strategies fire same bar, opposite direction: confidence delta >= 0.05 → higher wins, else skip all
- One strategy in trade, new same-direction signal: fill remaining contracts up to 50 ceiling
- One strategy in trade, new opposite-direction signal: skip unless open trade > 50% drawdown AND new confidence > 0.70
- Same model, both strategies fire opposite directions: CNN handles at inference, one output per bar
- Confidence-based position sizing table (10/20/30/50 MNQ tiers)
- End-of-session flat rule: all positions closed by 14:55 ET

---

## Files to UPDATE

### `Home.md`
- Contract limit: 5 MNQ → 50 MNQ in Prop Firm Rules Summary section
- Strategy count: 8 → 9 (Session Level Pivots becomes active)
- Session Level Pivots row in Active Strategies table: change from "Feature engineering" to active strategy with signal
- Strategy Signal Summary table: add session_pivot_signal row, update ONNX column to reflect 4 models
- Model architecture note: 8 ONNX files → 4 ONNX files
- Next Session block: full agent run order (4A → 4A-Audit → 4B → 4B-Audit → 3C → 3C-Audit → 3D → 3D-Audit → 3E → 3E-Audit)
- Build log: new entry for this change plan

### `Reference/Prop Firm Rules.md`
- Max contracts: 5 → 50 MNQ
- Stop sizing table: rewrite for 50 MNQ reality
  - At 50 MNQ × $2 = $100/point, a 10-point stop = $1,000 = full daily loss limit
  - Realistic per-trade sizing: 10–30 MNQ, 10–20 point stops
- TopStepRiskManager class: update max_contracts = 50, update position_size() to confidence-based tiers
- Transaction cost: MNQ round-trip stays ~$1.40/contract ($0.70 each way commission + spread)

### `Strategies/Session Level Pivots.md`
- Remove "Feature engineering only" designation
- Add full signal spec:
  - Long signal: price touches or penetrates S4 or S3 → same-bar close back above level (rejection wick) → price below prior-day close → signal = +1
  - Short signal: price touches H3 or H4 → same-bar close back below (rejection wick above) → price above prior-day close → signal = -1
  - Proximity: within 0.5 ATR of level counts as touching
  - Session high/low variant: Asia/London/Pre-Market highs/lows as secondary levels
  - Max 2 signals per day
  - Priority: H4/S4 > H3/S3 > session highs/lows
- Add to ONNX table: covered by Model 3 with ORB IB
- Add instrument: MNQ 5min

### `Implementation/Architecture Overview.md`
- ONNX section: 8 files → 4 files, add model grouping table
- Training approach section: add explicit all-bars paragraph
- Walk-forward section: update to 5 folds

### `Implementation/ML Operators Guide.md`
- Phase D training: all session bars, NoTrade for non-signal rows
- Model count: 8 → 4 grouped
- Contract limit: 50 MNQ
- Add explicit warning: "Training must never filter rows by signal activity. Every session bar must appear in the training pool."

### `Agents/Agent 3C - Rolling Walk Forward and Purging.md`
- Add to prerequisites: "Agent 4A and 4B must be complete and audited before running 3C"
- Add note: all-bars training is already in place from 4B — 3C only changes fold boundaries

### `Agents/Agent 3D - Triple Barrier Labels and Meta Labeling.md`
- Add to prerequisites: "Agent 4A and 4B must be complete and audited"
- Clarify: triple-barrier only labels signal bars. Non-signal bars already carry NoTrade from 4B. Triple-barrier must NOT overwrite NoTrade on non-signal bars.

### `Agents/Agent 3E - Bootstrap Confidence Intervals.md`
- Update contract limit references: 50 MNQ
- Update position sizing references in funded_sim gate

---

## Files That Do Not Change

- `Agents/Agent 1 - Data Pipeline.md` through `Agent 2.5 - Review Agent.md` — complete and locked
- `Agents/Agent 3A - Build Tune Funded Sim Export.md` — historical reference
- `Agents/Agent 3B - Independent Audit.md` — historical reference
- `Implementation/Testing Requirements.md` — valid, new tests added by individual agents
- `Implementation/NinjaScript Integration.md` — deployment spec correct, Execution Rules is a new separate doc
- `Implementation/ifvg_generator.py`, `camarilla_pivot_generator.py`, `ttm_squeeze_generator.py` — do not touch
- `Reference/CNN Research Notes.md`, `Reference/Tick Data - Delta Features.md`
- `Liquidity & Structure/Liquidity Levels.md`
- All `ml/AGENT*_STATUS.md` files — historical record, do not modify
- All `ml/artifacts/` — baseline artifacts kept for reference

---

## Python Files Agents Will Modify (not touching until agents run)

| File | Agent | Change |
|---|---|---|
| `ml/signal_generators.py` | 4A | Add session_pivot_signal() |
| `ml/dataset_builder.py` | 4A + 4B | Wire pivot generator (4A), remove signal-bar gating (4B) |
| `ml/train.py` | 4B | 4 grouped model configs, 5-fold walk-forward, all-bars training |
| `ml/topstep_risk.py` | 4B | max_contracts=50, confidence-based sizing |
| `ml/funded_sim.py` | 4B | Confidence-based position sizing |
| `ml/labels.py` | 3D | New file — triple-barrier label generator |
| `ml/evaluate.py` | 3D + 3E | Threshold sweep (3D), bootstrap integration (3E) |
| `ml/bootstrap.py` | 3E | New file — stationary block bootstrap |

---

## Agent Run Order

```
Agent 4A  →  Agent 4A-Audit  →  Agent 4B  →  Agent 4B-Audit
                                                      ↓
                              Agent 3C  →  Agent 3C-Audit
                                                      ↓
                              Agent 3D  →  Agent 3D-Audit
                                                      ↓
                              Agent 3E  →  Agent 3E-Audit
```

Each builder must complete and pass its audit before the next builder starts.
No step may be skipped. No step may run out of order.

---

## Explicit Training Data Guarantee

Every agent spec, every audit spec, and every code change must enforce this:

> **Each of the 4 models receives the full ~98,000 session bars (5min) or ~390,000 session bars (1min) as training input per fold. The training pool is never filtered by signal activity. Every bar that falls within the 09:30–15:00 ET session window and within the fold's train date range is a training row. Signal bars receive triple-barrier labels. Non-signal bars receive NoTrade labels. No exceptions.**

Any audit that finds a training row count below 90% of the expected full-session bar count must block and report it as a critical failure.

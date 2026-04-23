# Automated Trading Brain 🧠

**Project:** Automated Prop Firm Payout Harvesting System
**Current Phase:** ML Training Layer — Conv1D CNN on MNQ OHLCV data
**Prop Firm Target:** TopstepTrader 50K Combine
**Execution Path:** PyTorch → 4 ONNX models (grouped architecture, 2–3 strategies each) → NinjaScript wrapper DLL (NinjaTrader 8)
**Execution Fallback:** Contingency only — use a Python inference server (named pipe) or TradingView → webhook path only if the NinjaScript deployment path proves unworkable

This vault is the primary context document for all Claude sessions working on this project. Start every session by reading this file, then the relevant strategy or implementation notes.

---

## Current Status

- Agent 1A and Agent 1B have been verified successfully.
- Agent 2 baseline is complete as of 2026-04-09.
- Agent 2.5 QA review was added on 2026-04-10. Read [AGENT_REVIEW_REPORT](ml/AGENT_REVIEW_REPORT.md) before Agent 3 work.
- Agent 2.5 targeted hardening was completed on 2026-04-10. Read [AGENT2_5_HARDENING_STATUS](ml/AGENT2_5_HARDENING_STATUS.md) before Agent 3 work.
- Current verified repo state: Agent 3 full suite `68 passed, 2 skipped, 1 warning` from `python -m pytest ml/tests -v --tb=short`; post-audit focused Agent 3 tests pass under Python 3.14 and Python 3.13.
- Baseline Agent 2 artifacts now exist in `ml/artifacts/`: 8 scalers, 8 checkpoints, and 8 evaluation CSVs.
- The Agent 2 artifacts are pre-hardening baseline references; retrain/tune from the refreshed feature parquets before final ONNX export.
- Read [AGENT2_STATUS](ml/AGENT2_STATUS.md) before starting Agent 3.
- Read [Agent 2.5 - Review Agent](Agents/Agent%202.5%20-%20Review%20Agent.md) for the repeatable Agent 1/2 review checklist.
- Review [ML Cleanup Backlog](Implementation/ML%20Cleanup%20Backlog.md) before any refactor pass on Agent 1 or Agent 2 code.

### Locked Agent 2 Conventions (Superseded — see new architecture below)

> ⚠️ These conventions were the original Agent 2 baseline. The new architecture (Agent 4A → 3E) supersedes them. Kept here for historical reference only.

- Model inputs are the shared 35 feature columns only. `future_return` and `label` are target-only columns.
- Walk-forward training originally used two anchored folds (now replaced with 5 rolling folds).
- Each strategy model originally trained only on signal bars (now fixed — all-bars training).
- Agent 2 evaluation baseline used fixed 5 MNQ contracts (now fixed — 50 MNQ max, dynamic sizing).

### Next Session — Required Run Order

Agent 3A/3B completed with zero deployment candidates. Root causes identified and fully addressed in new architecture. The new complete run order is:

```
4A → 4A-Audit → 4B → 4B-Audit → 3C → 3C-Audit → 3D → 3D-Audit → 3E → 3E-Audit
```

Each Audit agent **blocks** the next build agent. Do not skip audits.

| Agent | Description | Spec File |
|---|---|---|
| **4A** | Camarilla pivot features + session_pivot_signal column | [Agent 4A](Agents/Agent%204A%20-%20Pivot%20Features%20and%20Session%20Signal.md) |
| **4A-Audit** | Adversarial check of prior-day-only calc, rejection logic, signal rate | [Agent 4A-Audit](Agents/Agent%204A-Audit%20-%20Pivot%20and%20Signal%20Review.md) |
| **4B** | 4-group model architecture, all-bars training, 5 rolling folds | [Agent 4B](Agents/Agent%204B%20-%20Grouped%20Model%20Architecture.md) |
| **4B-Audit** | Critical: training row count must be >36,000; no signal-bar filter | [Agent 4B-Audit](Agents/Agent%204B-Audit%20-%20Architecture%20Review.md) |
| **3C** | Rolling walk-forward + purge/embargo | [Agent 3C](Agents/Agent%203C%20-%20Rolling%20Walk%20Forward%20and%20Purging.md) |
| **3C-Audit** | 5 folds, no test overlap, purge gap ≥ forward horizon | [Agent 3C-Audit](Agents/Agent%203C-Audit%20-%20Rolling%20Walk%20Forward%20Review.md) |
| **3D** | Triple-barrier labels, meta-labeling | [Agent 3D](Agents/Agent%203D%20-%20Triple%20Barrier%20Labels%20and%20Meta%20Labeling.md) |
| **3D-Audit** | 10 adversarial label checks with synthetic paths | [Agent 3D-Audit](Agents/Agent%203D-Audit%20-%20Label%20and%20Objective%20Review.md) |
| **3E** | Block bootstrap CIs, 4-condition deployment gate | [Agent 3E](Agents/Agent%203E%20-%20Bootstrap%20Confidence%20Intervals.md) |
| **3E-Audit** | 8 statistical checks, final deployment summary | [Agent 3E-Audit](Agents/Agent%203E-Audit%20-%20Statistical%20Review.md) |

Before starting: read `ml/AGENT_REVIEW_REPORT.md` and `ml/AGENT2_5_HARDENING_STATUS.md`.

## Active Strategies

| Strategy | Type | Timeframes | Status |
|---|---|---|---|
| [[IFVG]] | SMC / Price Action | 1min, 2min, 3min, 5min | ✅ Fully defined |
| [[IFVG - Open Variant]] | SMC / Open Manipulation | 1min, 2min, 3min, 5min | ✅ Fully defined |
| [[ORB IB - Initial Balance]] ⭐ | ORB — 60min range | 5min | ✅ Fully defined |
| [[ORB Volatility Filtered]] ⭐ | ORB — ATR percentile filter | 5min | ✅ Fully defined |
| [[ORB Wick Rejection]] ⭐ | ORB — body momentum filter | 5min | ✅ Fully defined |
| [[ORB Volume Adaptive]] | ORB — volume confirmation filter | 5min | ✅ Fully defined |
| [[Session Level Pivots]] | Active strategy — Camarilla H3/H4/S3/S4 rejection | 5min | ✅ Fully defined |
| [[TTMSqueeze]] | Volatility compression breakout | 5min | ✅ Fully defined |
| [[ConnorsRSI2]] | Mean reversion — RSI(2) fade | 5min | ✅ Fully defined |

⭐ = confirmed top performer in backtest testing

---

## Strategy Signal Summary

All 9 strategies produce signals used as **CNN input features**. The CNN learns which conditions produce profitable outcomes. Strategies are grouped into **4 CNN models** — each model trains on all session bars (~98,000 bars per 5min parquet) using a NoTrade label for non-signal bars.

**4 ONNX models are deployed.** Each model covers 2–3 strategies. All models run simultaneously in NinjaScript with shared conflict-resolution rules (see `Implementation/NinjaScript Execution Rules.md`).

| Group | ONNX File | Strategies Covered | Signal Columns |
|---|---|---|---|
| Model 1 | model_1.onnx | IFVG + ConnorsRSI2 | ifvg_signal, connors_signal |
| Model 2 | model_2.onnx | IFVG Open Variant + TTMSqueeze | ifvg_open_signal, ttm_signal |
| Model 3 | model_3.onnx | ORB Vol + Session Pivot Rejection + Session Pivot Break | orb_vol_signal, session_pivot_signal, session_pivot_break_signal |
| Model 4 | model_4.onnx | ORB IB + ORB Wick | orb_ib_signal, orb_wick_signal |

**All signal columns:** `{1=long, -1=short, 0=none}`. Non-signal bars are labeled NoTrade (class 2) during training — they are NOT filtered out.

| Strategy | Signal Column | Max/Day | Notes |
|---|---|---|---|
| IFVG | ifvg_signal | 2 (shared w/ open) | Liq sweep + HTF FVG required |
| IFVG Open Variant | ifvg_open_signal | 2 (shared w/ base) | 9:30–9:35 sweep only |
| ORB IB | orb_ib_signal | 1 | 60min range |
| ORB Volatility Filtered | orb_vol_signal | 1 | ATR pct filter gate |
| ORB Wick Rejection | orb_wick_signal | 1 | Body pct filter gate |
| TTMSqueeze | ttm_signal | Unlimited | Squeeze bars ≥ 5 required |
| ConnorsRSI2 | connors_signal | Unlimited | 200 SMA trend filter |
| Session Pivot Rejection | session_pivot_signal | 2 (shared) | Camarilla H3/H4/S3/S4 fade |
| Session Pivot Break | session_pivot_break_signal | 2 (shared) | Camarilla H4/S4 close-through continuation |

---

## Data

**Location:** `data/` folder in this vault
**Instrument:** MNQ
**Timeframes:** 1min, 2min, 3min, 5min, 15min, 30min, 1H
**Date range:** 2021-03-19 to 2026-03-18
**Format:** CSV — `datetime,open,high,low,close,volume`
**Timezone:** Eastern time with UTC offset in datetime column
**Session filter for training:** 09:30–15:00 ET
**Future expansion:** `MES` can be added later, but the current training, evaluation, export, and execution docs assume `MNQ` only

---

## Architecture at a Glance

```
OHLCV CSVs → Feature Engineering (incl. Camarilla pivots)
                    ↓
          4 Parquets (1min/3min/5min + all signals)
                    ↓
          Conv1D CNN — ALL BARS TRAINING
          (NoTrade label for non-signal bars)
          5 Rolling Walk-Forward Folds
          Triple-Barrier Labels
                    ↓
          4 ONNX Models (model_1 … model_4)
          + feature_order.json + scaler_params.json (per model)
                    ↓
          NinjaScript (NinjaTrader 8)
          50 MNQ max — dynamic confidence sizing
          Conflict resolution per execution rules
          NinjaTrader → Rithmic → TopStep 50K
```

See [[Architecture Overview]] for full detail.

---

## Prop Firm Rules Summary

**TopStep 50K Combine:**
- Profit target: $3,000
- Max trailing drawdown: $2,000 (trails highest EOD balance)
- Daily loss limit: $1,000
- Consistency rule: no single day > 40% of total profits
- **Max contracts: 50 MNQ** (= 5 NQ mini, $2/point). Dynamic sizing via `position_size(stop_pts, confidence)`.

See [[Prop Firm Rules]] for full detail including risk simulator code.

---

## Critical Logic Gaps (Must Test For)

See [[Testing Requirements]] for the full test suite. Top 8 most dangerous:

1. **Scaler fit on full dataset** → leaks validation stats into training → silent fraudulent results
2. **Wrong shift direction on labels** → `shift(+1)` is backward-looking, `shift(-1)` is forward-looking → easy mistake
3. **High == low doji bars** → synthetic delta formula divides by zero → guard required
4. **SMA(200) warmup ignored** → ConnorsRSI2 signals computed before bar 200 use NaN trend filter
5. **Causal convolution padding** → `padding='same'` leaks future data in Conv1D → PyTorch implementation must use left-only causal padding
6. **NinjaScript timezone** → NT bars are Central time, training is Eastern → +1 hour offset required
7. **TopStep trailing DD** → trails from highest EOD, not account start → wrong if coded as `account_start - current > 2000`
8. **TopStep consistency rule** → no single day > 40% of profits → must be checked in risk simulator

---

## Implementation Notes

| File | Contents |
|---|---|
| [[Architecture Overview]] | Full system design, data layer, 4-model spec, ONNX export |
| [[ML Operators Guide]] | Phase-by-phase build instructions, agent handoff protocol |
| [[Testing Requirements]] | Full test suite with code — run after every agent session |
| [[NinjaScript Integration]] | Timezone fix, ONNX loading, feature order contract |
| [[NinjaScript Execution Rules]] | Conflict resolution, sizing, session-end rules, drawdown guard |

---

## Reference

| File | Contents |
|---|---|
| [[Prop Firm Rules]] | TopStep 50K exact rules + risk simulator Python class |
| [[Tick Data - Delta Features]] | Synthetic delta formula, normalization, real data options |
| [[CNN Research Notes]] | Survey findings: Conv1D confirmed, failure modes, reporting standards |
| [[Liquidity Levels]] | Session times, valid sweep levels for IFVG setups |

---

## Agent Handoff Protocol

Every new Claude session working on this project must:
1. Read `Home.md` (this file)
2. Read the relevant strategy notes for the current task
3. Read `Implementation/ML Operators Guide.md` for the current phase
4. Check for `ml/AGENT*_STATUS.md` — if present, read the latest one before doing anything else
5. Run `python -m pytest ml/tests/ -v --tb=short` before making changes
6. Run tests again after changes
7. Write your status to an `AGENT*_STATUS.md` file before ending the session

---

## Build Log

| Date | Update |
|---|---|
| 2026-04-07 | Full vault rebuild. All 8 strategies defined. Implementation + Reference folders added. Data copied. TopStep 50K rules documented. Logic gaps and test suite documented. |
| 2026-04-09 | Agent 1A and 1B verified. Agent 2 baseline completed. Full test suite passed (`27 passed, 1 warning`). Baseline scalers, checkpoints, eval CSVs, and `ml/AGENT2_STATUS.md` written. |
| 2026-04-12 | Agent 3A scaffold added for Express Funded simulation, validation-only HPO guards, and research ONNX export. Agent 3B audit fixed funded consistency/scaling/minimum-payout issues. No deployment candidates approved yet. |
| 2026-04-12 | Full Agent 3A run completed: 20 HPO trials per strategy, all 8 strategies retrained from selected configs, final fold-2 ranking written, zero research/deployment candidates, and stale TTM ONNX smoke export removed. Agent 3B found no deployment blocker. |
| 2026-04-13 | Vault cleanup: removed 54 empty `pytest-cache-files-*` dirs, 3 `__pycache__` dirs, and empty `Raw/` folder. Added Agent 3C/3D/3E specs to address the ~55-day final test window and label/objective mismatch — rolling walk-forward, triple-barrier labels, and block-bootstrap CIs. |
| 2026-04-16 | Major architecture overhaul. Root causes of zero deployment candidates fully addressed: (1) signal-bar gating removed — all 4 models now train on full ~98k session bars; (2) 2 anchored folds → 5 rolling folds; (3) shift(-N) labels → triple-barrier; (4) 8 per-strategy models → 4 grouped models. Session Level Pivots promoted from feature engineering to active 9th strategy. Contract limit corrected: 5 MNQ → 50 MNQ. Dynamic confidence-based position sizing added. NinjaScript Execution Rules documented. MASTER_CHANGE_PLAN.md created. Full agent spec set: 4A, 4A-Audit, 4B, 4B-Audit, 3C-Audit, 3D-Audit, 3E-Audit. |

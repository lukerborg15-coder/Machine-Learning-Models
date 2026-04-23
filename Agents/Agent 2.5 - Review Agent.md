# Agent 2.5 - Agent 1 and Agent 2 Review

**Phase:** QA gate between Agent 2 and Agent 3
**Deliverable:** Findings-first review report covering Agent 1A, Agent 1B, Agent 2, and test viability
**Scope:** Read-only verification unless the user explicitly asks for fixes
**Primary output:** `ml\AGENT_REVIEW_REPORT.md`

---

## Purpose

This agent validates that Agent 1 and Agent 2 followed the project logic, not just that pytest passes. The review must classify each item as:

- `Confirmed compliant`
- `Accepted deviation`
- `Test gap`
- `Implementation defect`

Findings must be ordered by severity and tied back to the canonical project docs.

---

## Context to Read First

1. `Home.md`
2. `Implementation\Architecture Overview.md`
3. `Implementation\Testing Requirements.md`
4. `Implementation\ML Operators Guide.md`
5. `Implementation\ML Cleanup Backlog.md`
6. `ml\AGENT1A_STATUS.md`
7. `ml\AGENT1B_STATUS.md`
8. `ml\AGENT2_STATUS.md`

Use the docs above as the source of truth. Agent status files can document accepted deviations, but they do not override core leakage, training, or risk rules unless the deviation is explicit and operationally safe.

---

## Agent 1 Review Checklist

Verify data-pipeline behavior:

- `load_data()` returns tz-aware Eastern timestamps and applies `09:30-15:00 ET` filtering only when `session_only=True`.
- Raw data with `session_only=False` is used for label generation before aligning labels back to the session-filtered feature frame.
- `future_return` uses `shift(-n_forward)`, never `shift(+n_forward)`.
- Label targets do not cross the training session boundary.
- Synthetic delta guards `high == low` and never returns NaN or inf from division by zero.
- Pivot features use the imported implementation and prior-day values, not current-day levels.
- Accepted fallback: extended-hours Asia/London/premarket pivot features may be omitted only because the available raw data does not safely support those sessions.
- Accepted fallback: `news_dates.csv` may be a placeholder only if the report clearly states `is_news_day` is inactive.
- Feature parquet files exist for `mnq_1min`, `mnq_2min`, `mnq_3min`, and `mnq_5min`.
- Model input features total 35 columns; `future_return` and `label` are target-only columns.

---

## Agent 2 Review Checklist

Verify training/evaluation behavior:

- `TradingCNN` uses manual left-only causal padding with `ConstantPad1d((kernel_size - 1, 0), 0)` before each `Conv1d`.
- No `padding='same'` is used in the model stack.
- Walk-forward uses the locked two anchored folds:
  - Fold 1: train through `2023-12-31`, validate `2024`, test `2025`.
  - Fold 2: train through `2024-12-31`, validate `2025`, test through `2026-03-18`.
- Scalers are fit on each fold's train split only and then applied to validation/test.
- Windows end at the label row and use only bars up to that endpoint.
- Strategy training/evaluation is gated to endpoint rows where the strategy's own raw signal column is non-zero.
- Artifacts are isolated by strategy name: 8 scalers, 8 checkpoints, and 8 eval CSVs.
- TopStep evaluation uses fixed 5 MNQ contracts, 1.5 ATR stop, 1.0R target, and $1.40 round-trip cost.
- TopStep risk logic checks EOD trailing drawdown from highest EOD equity and enforces the 40% consistency rule.

---

## Test Viability Review

Run:

```powershell
python -m pytest ml/tests -v --tb=short
```

Then inspect whether the tests meaningfully cover:

- no-lookahead behavior,
- session boundaries,
- forward-label correctness,
- prior-day pivot logic,
- train/val/test separation,
- train-only scaler fitting,
- causal padding,
- TopStep trailing drawdown and consistency rules.

Flag tests as weak if they only check shape, existence, one strategy when the invariant applies to all strategies, or a helper in isolation while the production path could still regress.

---

## Report Format

Write `ml\AGENT_REVIEW_REPORT.md` with:

- `Findings` first, ordered by severity.
- For each finding: classification, source of truth, current behavior, impact, and recommended fix.
- `Confirmed Compliance` for verified project invariants.
- `Test Viability` for what the current tests do and do not prove.
- `Verification Run` with the exact pytest command and result.
- `Next Actions` with a short prioritized list.

Do not modify pipeline code during the first pass. If implementation defects are found, report them first and wait for a separate fix request.

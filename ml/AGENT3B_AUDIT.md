# Agent 3B Audit - Agent 3A Scaffold

**Date:** 2026-04-12 local session  
**Verdict:** Not deployment-ready. Scaffold is useful, tests pass, but final HPO/retraining and all-strategy exports are still pending.

## Findings

1. **Resolved implementation defect:** Consistency XFA payout logic was initially too permissive because it used only positive days. It now uses largest winning day divided by total net profit in the payout window, matching the current Topstep payout policy language.

2. **Resolved implementation defect:** Express Funded simulation initially used fixed 5 MNQ contracts. It now caps the requested contract count by a 50K XFA-style scaling plan and only increases the cap on the next session after EOD balance reaches a higher tier.

3. **Resolved implementation defect:** Payout simulation initially allowed tiny positive payout requests. It now enforces the current `$125` minimum payout request.

4. **Improved weak test:** HPO validation-only protection was initially only asserted in dry-run metadata. A non-dry-run monkeypatched test now proves the real HPO path passes train and validation frames to training/evaluation and does not use the test split.

5. **Improved weak test:** ONNX parity initially used a zero tensor only. It now tests a deterministic batch containing zeros and nonzero random inputs.

6. **Remaining deployment blocker:** No real staged HPO has been run, no strategy has been retrained from refreshed parquets, and no real `hyperparam_{strategy}.csv` outputs exist.

7. **Remaining deployment blocker:** Only TTM has a research-only ONNX smoke export, and it comes from the pre-hardening Agent 2 checkpoint. It must not be used as a final deployment model.

8. **Residual modeling risk:** Funded DLL/MLL failures are still checked after simulated exits at bar-level. That can miss exact intratrade auto-liquidation behavior, so final deployment approval should either implement intratrade threshold liquidation or document the approximation as research-only.

## Audit Commands

- `python -m pytest ml/tests/test_agent3.py -v --tb=short`: `13 passed, 2 skipped, 1 warning`.
- `py -3.13 -m pytest ml/tests/test_agent3.py -v --tb=short`: `15 passed, 1 warning`.
- `python -m pytest ml/tests -v --tb=short`: `64 passed, 2 skipped, 1 warning`.
- TTM ONNX parity under Python 3.13: max diff `1.9744038581848145e-07`, under `1e-5`.

## Safe-To-Use Decision

`ml/artifacts/FINAL_EVAL_REPORT.md` is safe as a planning report only. It is not safe as a NinjaScript deployment approval report until Agent 3A runs real HPO/retraining, exports every candidate artifact from the winning configs, and Agent 3B re-audits those outputs.

## Final Agent 3B Audit Update

**Date:** 2026-04-12 late session

Agent 3B audited the completed Agent 3A run and found no deployment blocker. The final report and rankings are consistent: all 8 strategies are `not_exported`, no research/deployment candidates exist, no stale ONNX sidecars remain, and no funded ledgers were expected because no strategy passed the Combine gate.

Audit findings addressed after review:

- Fixed future-run `save_fold_name` behavior so a requested final fold must be available instead of silently falling back to another fold.
- Added symmetric intratrade liquidation tests for short DLL and long MLL, complementing the existing long DLL and short MLL tests.

Residual note:

- The Topstep scaling-plan thresholds are still encoded as the current configured 50K XFA scaffold. Re-verify the exact tier table before any future live deployment decision, because the public page exposes part of that table as an image.

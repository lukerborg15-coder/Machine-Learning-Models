# Agent 1 — Data Pipeline & Feature Engineering (SPLIT)

> **This agent has been split into two sub-agents for better context management.**

| Sub-Agent | Spec File | Scope |
|---|---|---|
| **Agent 1A** | [[Agent 1A - Signal Generators]] | Data loader, all 7 signal generators, signal tests |
| **Agent 1B** | [[Agent 1B - Feature Engineering]] | OHLCV features, pivot features, time features, labels, parquet assembly, remaining tests |

## Why the Split

Agent 1 was doing ~60% of the total project work in a single context window. The signal generators (4 ORB + ConnorsRSI2 from spec, 3 pre-built integrations) and the feature engineering pipeline are distinct enough to run in separate sessions without risk.

## Handoff Chain

```
Agent 1A → AGENT1A_STATUS.md → Agent 1B → AGENT1B_STATUS.md → Agent 2 → AGENT2_STATUS.md → Agent 3
```

Agent 1B reads Agent 1A's status file before starting. Agent 2 reads Agent 1B's status file.

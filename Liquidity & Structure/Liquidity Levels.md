# Liquidity Levels

Reference for all liquidity levels used across strategies, particularly [[IFVG]] and [[IFVG - Open Variant]].

---

## Session Levels

These levels are marked and updated each day. A sweep of any session high/low qualifies as a liquidity grab.

| Session | Time (ET) | Color |
|---|---|---|
| Asia | 20:00 – 02:00 | Blue |
| London | 02:00 – 07:00 | Red |
| NY Pre-Market | 07:00 – 09:30 | Orange |
| NY AM | 09:30 – 12:00 | Green |
| NY Lunch | 12:00 – 14:00 | Yellow |
| NY PM | 14:00 – 16:00 | Purple |

> **Note:** These times are aligned with [[Session Level Pivots]] — the canonical reference for all session boundaries in this project.

---

## Time-Based Levels

| Level | Description |
|---|---|
| Previous Day High | Prior regular session high |
| Previous Day Low | Prior regular session low |
| Previous Week High | Prior weekly candle high |
| Previous Week Low | Prior weekly candle low |

---

## Intraday Structural Levels

| Level | Description |
|---|---|
| 1H High / Low | Most recent 1-hour candle high and low |
| 4H High / Low | Most recent 4-hour candle high and low |

---

## Usage in Strategies

### [[IFVG]] — Liquidity Sweep Prerequisite
Before entering an IFVG setup, price must first sweep (wick through or close through) one of the levels above. The sweep establishes that resting orders have been triggered, increasing the probability of a reversal.

### [[IFVG - Open Variant]] — Opening Manipulation
The sweep must occur between 9:30–9:35 ET. Any level qualifies — size of the sweep does not matter, only that a defined level was taken out.

### ORB Strategies — [[ORB IB - Initial Balance]] / [[ORB Volatility Filtered]] / [[ORB Wick Rejection]] / [[ORB Volume Adaptive]]
Session and daily levels used for directional bias confirmation and target placement.

---

## Notes

- Equal highs/lows (two or more touches at the same price) represent the strongest liquidity pools
- Previous day/week levels are the most watched by institutions — sweeps of these levels carry the most weight
- Session levels update in real time — always use the completed session's high/low, not the current one

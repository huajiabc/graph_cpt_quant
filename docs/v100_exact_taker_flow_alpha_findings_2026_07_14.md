# v10.0 Exact Taker-Flow Alpha Findings

Status: `reject_exact_taker_flow_overlay`. No selector, shadow, PaperLive,
canary, leverage, or live permission changed.

## What was tested

The test used exact Bybit public trades around independently defined v0.1
`short_squeeze` and `momentum_ignition` events. Bybit's `side` field was used
as taker side. Every feature window ended before the signal minute, and entry
used the first minute open at or after the signal.

After the frozen 60-minute per-symbol cooldown, all 529 events had complete
5/15-minute feature and 15/60/240-minute outcome coverage across 12 symbols
and 62 active days. The primary decision horizon was 60 minutes and the focal
round-trip cost was 20 basis points.

## Frozen 60-minute result

| State | Full n | Full net20 | Validation net20 | Holdout net20 | Random family percentile |
|---|---:|---:|---:|---:|---:|
| OF1 active-buy confirmation | 81 | -0.0641% | +0.0210% | -0.3395% | 32.2% |
| OF2 sell absorption | 54 | -0.0871% | -0.3118% | -0.0193% | 24.2% |
| OF3 avoid buy exhaustion | 479 | -0.1460% | -0.1178% | -0.2973% | 5.0% |
| All covered events | 529 | -0.1367% | -0.1181% | -0.2511% | n/a |

All three candidates failed validation/holdout consistency, 30bp cost,
day-block bootstrap, familywise same-symbol/same-day random timing, or path
consistency. OF3 also failed to beat the +60-minute timing placebo.

The result is not a simple sample-size rejection. OF3 had 479 trades and its
day-block bootstrap interval for net20 was entirely negative
(-0.2713%, -0.0270%). OF1 improved the losing all-event baseline, but the
improvement did not survive holdout and its bootstrap interval crossed zero
(-0.3155%, +0.2365%).

## What remains informative

The exact-flow state is not useful as an immediate 60-minute overlay, but OF1
has a delayed diagnostic shape at 240 minutes:

- full net20 +0.4188%;
- development +0.9096%;
- validation -0.0685%;
- holdout +0.1243% (17 trades).

This cannot rescue v10.0 because 240 minutes was explicitly secondary and the
validation mean is negative. It motivates a separate, post-discovery v10.1
persistence test with fresh gates, BTC attribution, matched random timing, and
no threshold changes.

## Interpretation

Single-venue active flow does not provide a validated short-horizon timing
edge for the existing long events. The data weakly suggest that unusually
coherent active buying may describe a slower persistence regime, but it may
also be ordinary BTC beta or a small-sample artifact.

The event-conditioned archive cannot support unconditional intraday claims.
Synchronized Binance-to-Bybit flow remains a distinct forward-data hypothesis.

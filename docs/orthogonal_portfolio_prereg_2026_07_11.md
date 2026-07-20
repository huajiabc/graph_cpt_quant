# Orthogonal Context And P2 Portfolio Shadows - Pre-Registration

## Permissions

All additions in this document are `shadow/counterfactual_only`.
`P2_MAX8_BASELINE` remains the primary forward paper ledger. No field below may
skip, size, or enable a primary action before a separate promotion audit.

## A. Token Attention Context

### Frozen hypothesis

Same-token DEX pool attention visible during the 24 hours before entry may
identify a different continuation environment for CIC1. The context is useful
only if it beats both fixed placebo controls and remains distributed over time.

### Required forward fields

- mapping coverage and confidence;
- event time, event available time, and publication latency;
- latest visible event age at entry;
- dataset latest available time and dataset freshness at entry;
- actual same-token prior-24h flag/count;
- same-token 7-day-shift placebo flag/count;
- deterministic same-chain random-token prior-24h flag/count;
- source and event-type provenance;
- `live_action_allowed=false`.

### Frozen controls

- Placebo window: entry minus 8 days through entry minus 7 days.
- Random token: stable hash of the trade ID into the sorted A/B mapping on the
  same chain, excluding the traded token.
- Main lookback: 24 hours. Other windows remain descriptive.
- A token dataset is stale when its latest available event is more than 48 hours
  behind the entry time. Stale rows cannot support an attribution claim.

### Decision gate

No action claim before 100 timely P2 trades, at least 30 token-prior trades,
three active months, max best-month contribution 35%, positive 30bp stress,
actual lift above both placebo controls, and burst-clustered bootstrap CI above
zero.

## B. P2 Portfolio Risk Shadows

All arms use the identical P2 candidate pool, entry rule, and exit rule.

### Arms

1. `P2_EW`: current first-come max8, size 1.0.
2. `P2_VOL`: size `0.025 / stop_distance`, clipped to `[0.50, 1.25]`, with
   total notional exposure capped at 8 units.
3. `P2_BETA`: size 1.0 subject to total absolute rolling 7-day BTC-beta exposure
   capped at 6 units; a residual size below 0.25 is skipped.
4. `P2_CORR`: size 1.0, max8, with no more than two simultaneous positions in
   the same as-of correlation cluster. Missing clusters are logged and treated
   as unique-symbol clusters.

### P2_CORR cluster construction addendum (frozen 2026-07-12)

- Recompute membership at each UTC month boundary.
- Use only observations strictly before the month boundary, with a trailing
  30-day lookback; rows from the evaluated month must never affect membership.
- Similarity input is overlapping 1-hour return sampled on the existing
  15-minute feature grid.
- A symbol needs at least 672 observations (seven days) to receive covered
  membership.
- Create an undirected edge only when correlation is at least 0.70 and the two
  symbols are mutually within their three highest eligible correlations.
- Connected components define the frozen monthly correlation clusters;
  eligible isolated symbols receive singleton clusters.
- Symbols without sufficient input retain blank membership and follow the
  already-frozen unique-symbol fallback in `P2_CORR`.
- These parameters may not be changed after reading cluster coverage or replay
  performance. Any alternative threshold is a new hypothesis.

### Frozen risk inputs

- BTC beta and BTC correlation use entry-visible rolling 7-day 1-hour returns,
  with at least 48 hours of observations.
- Volatility sizing uses the frozen entry stop distance, not future volatility.
- Correlation-cluster membership must be as-of the entry month.
- Final burst size is never used.

### Evaluation

Report selected/skipped rows, missing-input coverage, total exposure, BTC-beta
exposure, cluster concentration, weighted net10/net20/net30, worst burst, worst
month, and turnover. Historical replay is diagnostic only. Promotion requires
100 timely core trades and at least 30 capacity-constrained decisions.

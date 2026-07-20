# v23.34-v23.42 Liquidation Graph Alpha Round Summary

## Outcome

This round opened a genuinely new information source: public OKX forced-liquidation
flow mapped to the frozen 17-coin graph. The source is usable for strict forward
research, but it does not yet establish tradable alpha.

The retained hypothesis is narrow:

> Aggregate cross-coin liquidation notional and source breadth, penalized by
> concentration, may improve forecasts of broad future market volatility.

Standalone direction, continuation breakout, and exhaustion fade are not retained.

## Data and causal integrity

- 17/17 Bybit research symbols map to live linear OKX USDT swaps.
- 4,927 unique liquidation events are stored with no missing notionals or duplicate
  event keys.
- Total observed notional is $32.54m: $27.35m long-position forced sells and $5.19m
  short-position forced buys.
- `event_time <= first_seen_at <= last_seen_at` is enforced.
- Knowledge time is the per-response receipt timestamp, not the batch start.
- The initial roughly 24-hour snapshot is explicitly retrospective and cannot be
  used before it became known.
- v23.34 passed 18/18 data checks.

## Mechanism findings

The one-day retrospective mechanism pilot found that liquidation flow mainly marks
an already-active volatility regime:

- 15-minute total liquidation versus next-60-minute BTC range: Spearman +0.374.
- Total liquidation versus preceding 60-minute BTC range: Spearman +0.593.
- Partial rank to future BTC range after controlling prior range: +0.111.
- Alt own-symbol forced-flow continuation over 60 minutes: -3.28 bp mean signed
  return, with 45.4% continuation.

This rejects liquidation flow as a standalone directional or continuation signal.

## Execution findings

High liquidation intensity predicts barrier touches but not profitable continuation:

- 0.625-sigma BTC OCO trigger rate rises from 62.5% in the bottom liquidation
  quartile to 95.8% in the top quartile.
- Top-quartile continuation OCO earns -9.89 bp/decision at 10 bp cost and -19.47 bp
  at 20 bp cost.
- One-hour top-quartile exhaustion fade earns only +0.30 bp gross and -9.28 bp after
  10 bp cost.
- Four-hour top-quartile fade earns +0.93 bp gross and -9.07 bp after 10 bp cost.
- Alt-only one-hour fade reaches +5.58 bp gross, but falls to -4.01 bp at the primary
  10 bp cost assumption.

No execution variant is economically retained.

## Graph and bucket findings

The graph aggregation result is materially stronger than the single-coin result:

- All-source liquidation bucket versus future 17-coin median 60-minute range:
  raw Spearman +0.414; partial rank +0.285 after controlling prior market range.
- Alt-only aggregate: raw +0.366; partial +0.220.
- Active-source breadth: raw +0.394; partial +0.256.
- Source notional HHI: raw -0.168; partial -0.093.
- Aggregate partial circular-shift percentile: 93.3%.
- Best single source partial rank is BTC at +0.241; best single alt is 1000PEPE at
  +0.148. Both are weaker than the corresponding aggregate buckets.
- Removing any one source leaves partial rank at or above +0.220.
- Removing any one receiver leaves partial rank at or above +0.252.
- On 22 non-overlapping hourly observations, partial rank remains +0.211.

This supports broad, distributed liquidation cascades as graph-level volatility
state information. It does not authorize selecting individual source-receiver edges.

## Forward contract

v23.35 freezes 5-, 15-, and 60-minute causal features for liquidation total,
forced-side imbalance, breadth, BTC share, concentration HHI, event sizes, and burst
ratios. Every feature requires:

`decision_time - window <= event_time < decision_time`

and

`first_seen_at <= decision_time`.

The hourly forward evaluation requires at least 336 decisions across 14 UTC days.
Regularized interactions are forbidden below 1,000 decisions; boosted/tree models
are forbidden below 2,000 and then require nested walk-forward validation.

The v23.42 append-only hourly feature ledger is active. Its frozen causal start is
2026-07-17 04:25:50 UTC. At the time of this summary the first complete hourly
decision had not yet occurred, so the ledger correctly contains zero rows and no
outcomes.

## Governance

No PaperLive, live, leverage, remote, application, order, or strategy lifecycle state
was changed. Retrospective findings cannot promote a candidate. Only the frozen
forward graph-bucket ledger may provide confirmation evidence.

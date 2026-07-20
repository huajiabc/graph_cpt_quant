# v18.5 BTC Leverage-Flow Directed Graph Preregistration

Status: frozen before inspecting graph-selected future bucket returns.

## Orthogonal mechanism

This round replaces price-correlation edges with synchronized Binance USD-M
positioning metrics. The source is BTC aggressive taker direction interacted with
new open-interest build or open-interest unwind. The target is next-bar altcoin
BTC-beta-neutral residual return.

## Data and timing

- Exact archived 15-minute close timestamps only; no forward fill.
- Forty-six price/metrics symbols, BTC as source and 45 possible alt receivers.
- A decision requires all BTC source fields and at least 40 cross-sectional metric
  symbols.
- Development: before 2026-01-01 UTC; validation: January-February 2026;
  holdout: March 2026 onward.

## Frozen source features

- `flow = log(BTC taker long/short volume ratio)`.
- `oi_change = log(BTC open-interest quantity).diff()`.
- `build_impulse = flow * max(oi_change, 0)`.
- `unwind_impulse = flow * max(-oi_change, 0)`.

Source events use shifted prior-30-day thresholds with at least 20 days:

- absolute BTC 15-minute return at or above q90;
- price direction times `flow` at or above q75 of absolute flow;
- build event: OI change at or above q70;
- unwind event: OI change at or below q30;
- one-hour cooldown separately for build and unwind.

The q85/q95 return thresholds are diagnostics only.

## Frozen monthly directed graph

For each calendar month, use only the preceding 30 days and require at least
2,000 paired observations per edge.

1. Estimate the alt's contemporaneous BTC beta and residual return.
2. For build and unwind separately, compute Spearman correlation from the BTC
   impulse at `t` to alt residual at `t+1`.
3. Compute the reverse correlation from the same alt impulse at `t` to BTC return
   at `t+1`.
4. Direction advantage is absolute forward minus absolute reverse correlation.
5. Edge score is absolute forward correlation plus half the direction advantage.
6. Select up to eight receivers with positive direction advantage, ranked by edge
   score. Require at least five receivers for a trade.

The sign of each frozen forward edge determines that receiver's trade direction;
it is not selected from event-month returns.

## Frozen candidates

- `LFG1_BTC_BUILD_DIRECTED_RESIDUAL_BUCKET` uses build events and build edges.
- `LFG2_BTC_UNWIND_DIRECTED_RESIDUAL_BUCKET` uses unwind events and unwind edges.

At the completed source bar, receiver weight sign is BTC source-flow sign times
the frozen edge sign. Alt weights are equal gross; their frozen beta exposure is
hedged with BTC and the book is normalized by total gross exposure.

- Primary holding: two bars / 30 minutes.
- Primary/stress round-trip costs: 30/40 bp.
- One-bar and four-bar holds are sensitivity diagnostics.

## Controls and gates

- 500 deterministic random receiver buckets from the same monthly eligible edge
  universe, retaining each random edge's frozen sign; compare against the maximum
  build/unwind family per draw.
- Exact reversed direction and one-bar delayed entry.
- q85/q95 source-return and 15/60-minute holding diagnostics.
- Day-block bootstrap with 2,000 iterations.
- At least 100 full-sample, 20 validation, and 25 holdout trades per candidate.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean and bootstrap 95% lower bound.
- At or above the random-family 95th percentile.
- Must beat reversed direction and one-bar delay.
- Both source-threshold and holding-period diagnostics remain positive.
- No single profitable month supplies more than 35% of total positive monthly PnL.

No PaperLive, live, application, leverage, remote, or real-order scope may change
in this research round.

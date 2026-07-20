# v17.9 BTC Receiver Propagation Preregistration

Status: frozen before first outcome reveal.

## Question

After an extreme BTC 15-minute price move confirmed by same-direction active taker
flow and elevated turnover, does the move propagate into causally ranked altcoin
receivers during the next completed bar? Can a receiver-versus-insulator graph
spread retain the effect after common BTC exposure and conservative costs?

This is distinct from v17.8. No current-bar laggard condition is used and no
catch-up claim is made.

## Data and split

- Binance USD-M 15-minute completed bars only.
- Universe: the locally complete high-liquidity panel; BTC is the only source;
  BTC and XAUT are excluded as receivers.
- Development: timestamps before 2026-01-01 UTC.
- Validation: 2026-01-01 through 2026-02-28 UTC.
- Holdout: timestamps on or after 2026-03-01 UTC.

## Frozen source event

- Prior-data-only rolling 30-day thresholds, shifted by one completed bar.
- Absolute BTC return at or above rolling q97.5.
- Direction-confirmed absolute taker imbalance at or above rolling q80.
- BTC turnover at or above rolling q75.
- Four-bar / one-hour cooldown.

Sensitivity diagnostics use return q95 and q99 without changing the primary rule.

## Frozen monthly causal graph

For every calendar month, use only the preceding 30 days and require at least
2,000 paired bars per altcoin.

1. Estimate contemporaneous BTC beta and alt residual return.
2. Measure Spearman correlation from BTC return at `t` to alt residual at `t+1`.
3. Measure absolute BTC shock to absolute residual response at `t+1`.
4. Subtract reverse absolute-flow correlation from alt at `t` to BTC at `t+1`.
5. Rank by signed-forward correlation plus half the absolute-direction advantage.

The receiver bucket is the top eight names with positive signed-forward
correlation. The insulator bucket is the bottom eight remaining names. A trade
requires at least five names in each required leg. Graph ranks are frozen for the
month and never use the event month itself.

## Frozen candidates

- `BRP1_SOURCE_DIRECTION_RECEIVER_BASKET`: direction times equal-weighted future
  return of the receiver bucket. Primary/stress round-trip costs: 20/30 bp.
- `BRP2_BTC_NEUTRAL_RECEIVER_PROPAGATION`: direction times receiver return less
  its frozen BTC-beta hedge, normalized by total gross exposure. Costs: 30/40 bp.
- `BRP3_RECEIVER_INSULATOR_PROPAGATION_SPREAD`: half-long receivers and half-short
  insulators in BTC-shock direction, hedged for residual BTC beta and normalized
  by total gross exposure. Costs: 30/40 bp.

Primary entry is the completed source-bar close and primary exit is one bar / 15
minutes later. Two-bar and four-bar holds are sensitivity diagnostics.

## Frozen controls and gates

- 500 deterministic random bucket draws with identical event timing and bucket
  sizes; eligibility is measured against the maximum candidate family per draw.
- One-bar delayed entry, reversed trade direction, and receiver/insulator rank
  reversal controls.
- Day-block bootstrap with 2,000 iterations.
- At least 100 full-sample, 20 validation, and 25 holdout events.
- Positive primary net return in development, validation, and holdout.
- Positive full-sample stress net return and positive bootstrap 95% lower bound.
- At or above the 95th percentile of the random family.
- Must beat one-bar delay and reversed direction.
- q95, q99, 30-minute, and 60-minute sensitivities must remain positive.
- No single profitable month may supply more than 35% of total positive monthly
  PnL.

No PaperLive, application scope, leverage, remote host, or real-order permission
may change from this research round.

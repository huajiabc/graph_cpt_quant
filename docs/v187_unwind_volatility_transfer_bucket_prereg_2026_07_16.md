# v18.7 Unwind Volatility-Transfer Bucket Preregistration

Status: frozen before inspecting future returns of the selected altcoin buckets.

## Mechanism

v18.6 retained BTC OI-unwind reversal as a statistically unusual but sub-cost
timing primitive. This round uses that frozen clock and asks whether the event
bar identifies altcoins that absorbed disproportionate price or leverage stress
and subsequently mean revert relative to BTC.

## Frozen source and risk model

- Use only the v18.5 q90 BTC unwind events; do not use build events.
- Exact completed 15-minute close and Binance USD-M metric timestamps only.
- At least 40 metric symbols must be present at the source event.
- For every calendar month and altcoin, estimate BTC beta and residual volatility
  using only the preceding 30 days, with at least 2,000 paired bars.
- Event-bar residual is alt return minus frozen beta times BTC return. Residual
  z-score divides it by the frozen residual volatility.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.

## Frozen candidates

### `VTR1_UNWIND_RESIDUAL_OVERSHOOT_REVERSAL`

Rank alts by BTC source sign times event-bar residual z-score. Retain positive,
finite scores and select the top eight (minimum five). Trade every selected alt
opposite the BTC source sign, then hedge their frozen aggregate BTC beta.

### `VTR2_UNWIND_SYNCHRONIZED_STRESS_REVERSAL`

An alt is eligible when all three event-bar quantities are positive:

1. BTC source sign times standardized alt return;
2. BTC source sign times log taker long/short volume ratio;
3. negative log open-interest change (an unwind).

Within the eligible cross-section, percentile-rank all three quantities and use
their equal-weight mean as the stress score. Select the top eight (minimum five),
trade opposite the BTC source sign, and hedge frozen aggregate BTC beta.

Both books use equal alt gross weight and normalize by alt plus BTC-hedge gross.
Entry is the completed event close; primary exit is 30 minutes later. Primary
and stress round-trip book costs are 30 and 40 bp. The one- and four-bar holds
are diagnostics.

## Frozen controls

- One-bar delayed entry keeps the event-time selected bucket unchanged.
- Bottom-ranked eligible bucket with the same size is a receiver-ranking control.
- q85/q95 BTC source-return thresholds and 15/60-minute holds.
- 500 deterministic random buckets of the same size from each event's eligible
  receiver universe. Each iteration is evaluated by the maximum mean across the
  two-candidate family.
- Day-block bootstrap with 2,000 iterations.

## Frozen eligibility gates

- At least 100 full-sample, 20 validation, and 25 holdout events.
- Positive primary mean in development, validation, and holdout.
- Positive full-sample stress mean and bootstrap 95% lower bound.
- At or above the random-receiver family-max 95th percentile.
- Beat one-bar delay and the bottom-ranked bucket.
- q85, q95, 15-minute, and 60-minute diagnostics remain positive after primary
  costs.
- No single profitable month supplies more than 35% of positive monthly PnL.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.

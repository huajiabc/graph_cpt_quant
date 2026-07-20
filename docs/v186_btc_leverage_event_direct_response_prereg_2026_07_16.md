# v18.6 BTC Leverage-Event Direct Response Preregistration

Status: frozen before inspecting BTC future returns for these direct candidates.

## Question isolated from v18.5

v18.5 established that a monthly directed altcoin receiver graph does not turn
BTC leverage-flow events into a tradable residual bucket. This round removes
the graph and receiver selection entirely. It tests whether the already frozen
BTC source events contain a direct BTC price response.

## Frozen data and source events

- Reuse the exact completed 15-minute BTC and Binance USD-M metrics panels and
  the v18.5 source construction without changing its feature definitions.
- `flow = log(BTC taker long/short volume ratio)`.
- `oi_change = log(BTC open-interest quantity).diff()`.
- Event thresholds use shifted prior-30-day data with at least 20 days:
  absolute BTC return q90, price-confirmed absolute flow q75, and OI q70/q30.
- One-hour cooldown remains separate for build and unwind events.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.

The q85 and q95 source-return thresholds are diagnostics only.

## Frozen candidates

- `LDR1_BTC_BUILD_CONTINUATION`: on an OI-build event, trade BTC in the source
  price/flow direction.
- `LDR2_BTC_UNWIND_REVERSAL`: on an OI-unwind event, trade BTC opposite the
  source price/flow direction, testing exhaustion after forced deleveraging.

Entry is the completed source-bar close. The primary exit is two bars / 30
minutes later. Primary and stress round-trip costs are 10 and 15 bp. One-bar
and four-bar holds are diagnostics.

## Frozen controls

- Exact reversed trade direction.
- Entry delayed by one completed bar while retaining each source event and
  trade direction.
- q85/q95 source-return thresholds and 15/60-minute holds.
- 500 deterministic within-calendar-month circular time shifts. One common
  nonzero bar offset is drawn per candidate-month and iteration, preserving
  event counts, directions, spacing, clustering, and the BTC return path while
  breaking source timing. Each draw is evaluated by the maximum result across
  the two-candidate family.
- Day-block bootstrap with 2,000 iterations.

## Frozen eligibility gates

- At least 100 full-sample, 20 validation, and 25 holdout events.
- Positive primary-cost mean in development, validation, and holdout.
- Positive full-sample stress-cost mean and positive bootstrap 95% lower bound.
- At or above the circular-shift family-max 95th percentile.
- Beat the exact reversed direction and one-bar delayed entry.
- q85, q95, 15-minute, and 60-minute diagnostics remain positive after primary
  costs.
- No single profitable month supplies more than 35% of total positive monthly
  PnL.

Passing would create an offline research candidate only. It cannot change any
PaperLive, live, application, leverage, remote, or real-order scope.

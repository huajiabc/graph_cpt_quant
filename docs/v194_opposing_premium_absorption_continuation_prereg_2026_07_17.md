# v19.4 Opposing-Premium Absorption Continuation Preregistration

Status: frozen after the v19.3 feature-only audit and before inspecting any
candidate future return.

## Mechanism and frozen coverage

The source is a large BTC price move accompanied by an expanded premium-index
range, while both the premium-index body and its close location point opposite
the BTC price move. The hypothesis is that opposing perpetual pressure is being
absorbed by stronger directional spot flow, so the price move should continue.

The primary q85 price threshold, BTC premium range z-score at or above 1.0, and
opposing body/close-location thresholds at or below -0.5 yield 178 direct events
(73 development, 42 validation, 63 holdout). Sixty-nine events (18/21/30) have
at least five eligible alt receivers. These counts were frozen without
calculating future candidate returns.

## Frozen features and timing

- BTC absolute 15-minute return must exceed its shifted prior-30-day q85,
  requiring at least 20 days.
- Premium range z-score is the completed high-low range minus its shifted
  prior-30-day mean, divided by its shifted prior-30-day standard deviation.
- Premium body z-score is completed close minus open divided by shifted
  prior-30-day body standard deviation.
- Close location is `2 * (close - low) / (high - low) - 1`.
- BTC price sign times premium body z-score and close location must both be at
  or below -0.5. BTC premium range z-score must be at least 1.0.
- Eligible alt receivers use the same aligned shape and require range z-score
  at least 1.0. Rank them by range z-score, select the top eight, and require at
  least five.
- Monthly BTC beta for alt hedging uses only the preceding 30 days and at least
  2,000 paired price bars.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.
- No premium, metric, or price value is forward filled.

## Frozen candidates

### `PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION`

Trade BTC in the source price direction. Primary hold is 30 minutes;
primary/stress round-trip costs are 10/15 bp.

### `PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET`

Trade every selected receiver in the source price direction, hedge frozen
aggregate BTC beta, and normalize by total gross exposure. Primary/stress book
costs are 30/40 bp.

## Frozen controls and diagnostics

- Exact reversed direction and one-bar delayed entry.
- Direct shape control: q85/range-z 1.0 through-pressure events, whose aligned
  premium body and close location are both at least +0.5, traded in the source
  direction.
- Bucket ranking control: bottom-ranked eligible absorption receivers of equal
  size.
- q90 price threshold and range-z 1.5 absorption subsets are diagnostics.
- 15- and 60-minute holding diagnostics.
- Up-move and down-move source means are reported and must both be positive.
- 500 deterministic controls: month-matched random q85 BTC price-shock subsets
  for the direct candidate and same-event random eligible receiver buckets for
  the alt candidate. Each draw uses the maximum mean across the two-candidate
  family.
- Day-block bootstrap with 2,000 iterations.

## Frozen gates

- Direct candidate: at least 150/30/50 full/validation/holdout events.
- Bucket candidate: at least 60/15/20 full/validation/holdout events.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean and bootstrap 95% lower bound.
- At or above the random family-max 95th percentile.
- Beat reversed direction, one-bar delay, and the candidate-specific shape or
  ranking control.
- q90, range-z 1.5, 15-minute, and 60-minute diagnostics remain positive after
  primary costs.
- Both up-move and down-move full-sample means remain positive.
- No single profitable month supplies more than 35% of positive monthly PnL.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.

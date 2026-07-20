# v18.8 Top-Trader Absorption Preregistration

Status: frozen after feature-only coverage audit and before inspecting candidate
future returns.

## Motivation and frozen coverage choice

The v18.6 BTC unwind-reversal clock was unusual but below costs, while v18.7
showed that contemporaneous price/OI stress alone cannot identify receivers.
This round adds the previously unused top-trader position ratio as an absorption
signal.

The feature-only audit compared q85/q90 source-return and q50/q55/q60/q65
absorption thresholds without calculating any future candidate return. The
primary source threshold is q85 and absorption threshold is q50 because this is
the least selective frozen pair with adequate coverage: 164 direct events
(110 development, 15 validation, 39 holdout) and 151 events with at least five
eligible alt receivers. q90 and absorption q55 are diagnostics.

## Frozen features and timing

- Reuse the exact v18.5 BTC unwind definition, except the preregistered primary
  absolute-return threshold is q85 rather than q90.
- `top_position = log(sum_toptrader_long_short_ratio)`.
- `btc_absorption = -sign(BTC return) * diff(BTC top_position)`.
- Require BTC absorption at or above its shifted prior-30-day q50, with at least
  20 days of history.
- All decisions use exact completed 15-minute bars; no forward fill.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.

## Frozen candidates

### `TDA1_BTC_TOPTRADER_ABSORPTION_REVERSAL`

At a filtered BTC unwind event, trade BTC opposite the source price/flow sign.
Primary/stress round-trip costs are 10/15 bp.

### `TDA2_ALT_TOPTRADER_ABSORPTION_BUCKET`

For each calendar month, estimate alt BTC beta and return volatility from only
the preceding 30 days with at least 2,000 paired bars. An alt is eligible when:

1. its event-bar return is aligned with the BTC source sign;
2. its log taker long/short volume ratio is aligned with the source sign;
3. its log open interest decreases;
4. its top-trader position-ratio change opposes the source sign.

Percentile-rank standardized aligned return, aligned taker flow, OI unwind, and
top-trader absorption; their equal-weight mean is the absorption-stress score.
Select the top eight, requiring at least five. Trade every alt opposite the BTC
source sign, hedge frozen aggregate BTC beta, and normalize by total gross.
Primary/stress book costs are 30/40 bp.

Both candidates enter at the completed source close and hold for 30 minutes.

## Frozen controls and diagnostics

- One-bar delayed entry retaining the event and selected bucket.
- Direct candidate: the non-absorbing q85 unwind complement is the ranking
  control. Bucket candidate: the bottom-ranked eligible bucket of equal size.
- q90 source-return and q55 absorption-threshold diagnostics.
- One- and four-bar holding diagnostics.
- 500 deterministic controls: month-matched random q85 unwind event subsets for
  the direct candidate and same-event random eligible receiver buckets for the
  alt candidate. Each iteration uses the maximum result across the two-candidate
  family.
- Day-block bootstrap with 2,000 iterations.

## Frozen eligibility gates

- At least 100 full-sample, 10 validation, and 25 holdout events.
- Positive primary mean in development, validation, and holdout.
- Positive full-sample stress mean and bootstrap 95% lower bound.
- At or above the random-control family-max 95th percentile.
- Beat the candidate-specific ranking control and one-bar delayed entry.
- q90, q55, 15-minute, and 60-minute diagnostics remain positive after primary
  costs.
- No single profitable month supplies more than 35% of positive monthly PnL.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.

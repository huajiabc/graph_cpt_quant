# v19.2 Premium-Innovation Unwind Reversal Preregistration

Status: frozen after data/feature audits and before inspecting candidate future
returns.

## Mechanism and frozen coverage

An OI unwind accompanied by an impact-price premium move in the same direction
is a closer proxy for forced execution than OI or taker ratios alone. The v19.1
feature audit calculated no future returns and froze q85 BTC source events with
aligned BTC premium innovation at or above 1.0 standard deviation.

This yields 103 direct events (68 development, 10 validation, 25 holdout) and 92
events with at least five eligible alt receivers (61/8/23). Long-liquidation and
short-cover sides have 59 and 44 direct events; they are mechanism diagnostics,
not independently tuned candidates.

## Frozen features and timing

- Reuse the v18.5 OI-unwind event with q85 absolute BTC return threshold.
- Premium innovation is the exact completed 15-minute premium-index close
  difference divided by its shifted prior-30-day standard deviation, requiring
  at least 20 days.
- A source premium shock requires BTC source sign times BTC premium innovation
  z-score at or above 1.0.
- Monthly BTC beta for alt hedging uses only the preceding 30 days and at least
  2,000 paired price bars.
- Development is before 2026-01-01 UTC, validation is January-February 2026,
  and holdout is March 2026 onward.
- No premium, metric, or price value is forward filled.

## Frozen candidates

### `PIR1_BTC_PREMIUM_SHOCK_REVERSAL`

Trade BTC opposite the source price/flow sign. Primary hold is 30 minutes;
primary/stress round-trip costs are 10/15 bp.

### `PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET`

At the same source event, retain alts whose premium innovation is aligned with
the BTC source sign and exceeds 1.0 standard deviation. Rank by aligned premium
innovation, select the top eight and require at least five. Trade every selected
alt opposite the source sign, hedge frozen aggregate BTC beta, and normalize by
total gross. Primary/stress book costs are 30/40 bp.

## Frozen controls and diagnostics

- Exact reversed direction and one-bar delayed entry.
- Direct candidate: the q85 unwind complement without a BTC premium shock.
- Bucket candidate: bottom-ranked eligible receivers of equal size.
- q90 source threshold and q85 broad-premium-shock subset are diagnostics. Broad
  shock means premium-innovation breadth is at or above shifted prior-30-day q70.
- 15- and 60-minute holding diagnostics.
- Long-liquidation and short-cover full-sample primary means are reported and
  must both be positive.
- 500 deterministic controls: month-matched random q85 unwind subsets for the
  direct candidate and same-event random eligible receiver buckets for the alt
  candidate. Each draw uses the maximum mean across the two-candidate family.
- Day-block bootstrap with 2,000 iterations.

## Frozen gates

- Direct candidate: at least 100/10/25 full/validation/holdout events.
- Bucket candidate: at least 75/8/20 full/validation/holdout events.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean and bootstrap 95% lower bound.
- At or above the random family-max 95th percentile.
- Beat reversed direction, one-bar delay, and the candidate-specific ranking
  control.
- q90, broad-shock, 15-minute, and 60-minute diagnostics remain positive after
  primary costs.
- Both long-liquidation and short-cover full-sample means remain positive.
- No single profitable month supplies more than 35% of positive monthly PnL.

Passing creates an offline research candidate only. No PaperLive, live,
application, leverage, remote, or real-order scope may change.

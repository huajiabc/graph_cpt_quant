# v10.9 Residual Graph-Community Dispersion Spread Preregistration

Date: 2026-07-14

## Distinct mechanism

Earlier graph tests predicted a target or downstream bucket from neighbor movement. v10.9
instead asks whether an extreme cross-sectional split inside an as-of residual-return community
creates a tradable market-neutral bucket spread.

## Frozen community graph

- Input: the continuous 73-symbol feature panel already materialized by v10.8.
- Exclude BTC from community membership and use it only for residualization.
- At each month start, estimate static BTC beta and residual one-hour returns from the preceding
  30 days; use one observation per hour and require 500 common samples.
- Build the complete residual-correlation graph, extract its maximum spanning tree, then remove
  the seven weakest tree edges to form exactly eight communities. This avoids selecting a
  correlation threshold from outcomes.
- Community membership and beta are frozen for the target month. Communities smaller than six
  symbols are recorded but are not traded.

## Frozen extreme-dispersion state

For each eligible community, calculate the 75th-minus-25th percentile of member residual
one-hour returns. Its threshold is the 95th percentile of that same statistic in the historical
30-day window. The target-month state is active when current dispersion reaches the frozen
threshold.

Only false-to-true transitions are accepted, with a four-hour cooldown per community. Split
members into current residual-return thirds:

1. `GDS1_COMMUNITY_CONVERGENCE`: long the bottom third and short the top third.
2. `GDS2_COMMUNITY_CONTINUATION`: long the top third and short the bottom third.

Each sleeve is normalized to 0.5 long plus 0.5 short. At most three disjoint community sleeves
are held per timestamp, ranked by dispersion divided by its frozen threshold.

## Outcomes, costs, and controls

- Primary horizon: four hours.
- Primary return: BTC-residual long-short spread; raw long-short spread is secondary.
- Total round-trip cost stresses: 20, 30, and 50 bp on normalized gross exposure.
- Development: before 2026-01-01; validation: 2026-01 through 2026-03; holdout: 2026-04 onward.
- Fifty random monthly partitions preserving all real community sizes.
- One-day shifted dispersion/rank signal with contemporaneous outcomes unchanged.
- Two-thousand entry-day block bootstrap samples.

## Forward-watch gate

Require at least 100 full, 25 validation, and 25 holdout observations; positive validation and
holdout residual net20; positive full net30; random-family percentile >= 90%; beat the shifted
placebo; bootstrap lower bound positive; all five chronological buckets nonnegative; and no
positive month or community above 35% of positive PnL.

Failure closes this exact community-dispersion mechanism, not all graph allocation uses. No
PaperLive, leverage, or live permission is changed.

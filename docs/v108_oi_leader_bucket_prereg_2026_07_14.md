# v10.8 OI-Leader Downstream Bucket Preregistration

Date: 2026-07-14

## Question

Can cross-coin open-interest shocks define an orthogonal directed graph whose downstream
multi-coin bucket has predictable return?

This is separate from the accumulating v10.7 cross-venue taker-flow tape. It uses the existing
continuous Bybit OI panel and does not substitute price for flow.

## Frozen data and graph

- Universe: the 73 symbols represented in the existing v0.7b graph universe; exclude BTC as a
  follower and retain it only for residualization.
- Grid: 15 minutes, warm-up-complete rows.
- First eligible target month requires at least 28 historical days.
- Rebuild monthly using only the preceding 30 days, ending four hours before month start so
  every historical target is realized as of graph freeze.
- Estimate static BTC beta from historical one-hour returns.
- Source variable: leader `oi_value_delta_z_1h`, clipped to [-5, 5].
- Target variable: follower future four-hour return minus frozen beta times BTC future four-hour
  return.
- Sample one observation per hour to reduce overlap. Require at least 500 common samples.
- Edge score is positive Spearman source/target correlation minus the reverse-pair correlation,
  with `sqrt(n/(n+500))` shrinkage. Retain three leaders per follower.

## Frozen states

1. `OIG1_POSITIVE_OI_PROPAGATION`: active leader has OI z-score >= +2 and positive one-hour
   return; long the downstream follower bucket.
2. `OIG2_NEGATIVE_OI_PROPAGATION`: active leader has OI z-score >= +2 and negative one-hour
   return; short the downstream follower bucket.

For each follower, aggregate active leaders by frozen positive edge weights. At least three
followers must be active; select up to five by pressure. Each candidate has a global four-hour
cooldown.

## Outcomes and controls

- Primary attribution: BTC-beta-neutral four-hour bucket return after 40 bp total round-trip
  cost for bucket plus BTC hedge.
- Secondary tradable outcome: naked directional bucket return after 20/30/50 bp total
  round-trip cost.
- Development: before 2026-01-01; validation: 2026-01 through 2026-03; holdout: 2026-04 onward.
- Fifty within-month random-leader graphs preserving follower indegree and edge weights.
- Reversed-edge graph and one-day shifted source signal.
- Two-thousand entry-day block bootstrap samples.

## Forward-watch gate

Require at least 100 full, 25 validation, and 25 holdout observations; positive residual net40
and raw net20 in validation and holdout; random-family percentile >= 90%; real beats reversed
and shifted controls; bootstrap lower bound positive; positive month and symbol concentration
below 35%; and positive raw return at 30 bp.

No PaperLive, leverage, or live permission changes are authorized.

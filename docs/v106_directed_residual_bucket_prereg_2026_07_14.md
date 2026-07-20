# v10.6 Directed Residual Graph-Bucket Preregistration

Date: 2026-07-14

## Question

Does a genuinely directed, BTC-residual lead-lag graph predict the return of a downstream
multi-coin bucket?

This changes the information content of the graph after v10.3-v10.5 rejected contemporaneous
top-five return-correlation buckets. It does not retune those rejected strategies.

## As-of graph

- Source panel: the unique 15-minute symbol rows preserved in the v10.3 feature panel.
- First test month: 2025-09; 2025-08 is history only.
- At each month start, use only the preceding 30 calendar days.
- Estimate a static symbol BTC beta from 15-minute returns in that history window.
- Residual return is `symbol_return - beta * BTC_return`.
- Candidate lags are frozen at 15, 30, and 60 minutes.
- For each ordered pair and lag, calculate
  `corr(leader_t, follower_t+lag) - corr(follower_t, leader_t+lag)` on residual returns.
- Require at least 1,000 complete common observations and a positive lag correlation and
  positive direction advantage.
- Apply sample shrinkage `sqrt(n / (n + 500))` and retain the strongest three leaders for
  each follower. Lag selection and edge ranking use history only.

## Signal and portfolios

At each target-month timestamp, aggregate a follower's leader residual one-hour returns using
the frozen positive edge weights. Record leader-positive breadth and the follower's own
residual one-hour return.

Two fixed candidate buckets are tested:

1. `DRB1_DIRECTED_PROPAGATION`: followers with predicted residual impulse >= 0.30%,
   cross-sectional score rank >= 80%, and leader-positive breadth >= two-thirds.
2. `DRB2_DIRECTED_LAGGARD`: DRB1 plus predicted-minus-own residual return >= 0.20%.

For each candidate, accept at most one timestamp every four hours. Select up to five followers
by predicted impulse and require at least three with finite forward outcomes. The trade is the
equal-weight downstream follower bucket.

## Outcomes and costs

- Primary: four-hour BTC-beta-neutral residual bucket return minus 40 bp total round-trip cost
  for follower bucket plus BTC hedge.
- Secondary: naked follower-bucket four-hour return minus 20/30/50 bp total round-trip cost.
- Development: 2025-09 through 2025-12.
- Validation: 2026-01 through 2026-03.
- Holdout: 2026-04 onward.

## Controls

- Fifty within-month random-leader graphs preserving every follower's indegree and edge weights.
- Reversed-edge graph.
- One-day shifted real-graph signal with contemporaneous outcomes unchanged.
- Two-thousand day-block bootstrap samples.
- The real candidate is compared with the maximum primary return across both candidates for
  every random graph.

## Forward-watch gates

All must pass:

- at least 100 full, 25 validation, and 25 holdout observations;
- positive primary residual net40 in full, validation, and holdout;
- positive secondary raw net20 in validation and holdout;
- at or above the random-family 90th percentile;
- beats both reversed-edge and one-day shifted controls;
- bootstrap 95% lower bound above zero;
- no positive month contributes more than 35% of positive primary PnL.

Failure changes no PaperLive, leverage, or live permission.

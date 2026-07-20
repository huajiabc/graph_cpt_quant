# v10.5 Direct Graph-Bucket Portfolio Preregistration

Date: 2026-07-14

## Question

Can a monthly frozen correlation graph define a directly tradable multi-coin bucket whose
future return is more predictable than buckets built from random neighbors?

This test does not trade a lagging target coin. Each source node defines an equal-weight
sleeve made from its five as-of neighbors. Portfolio observations are equal-weight averages
of at most three sleeves selected at a timestamp; overlap therefore represents an intentional
higher weight in coins shared by several graph neighborhoods.

## Frozen inputs

- Graph: monthly as-of top-five `return_corr_30d` edges from v0.7b.
- Feature grid: 15-minute, warm-up complete rows from the v10.3 panel.
- Primary horizon: 4 hours.
- Development: before 2026-01-01.
- Validation: 2026-01-01 through 2026-03-31.
- Holdout: from 2026-04-01.

## Candidate states

The common strong-bucket state is frozen as:

- neighbor bucket 1-hour return >= 0.50%;
- cross-sectional bucket percentile >= 80%;
- positive-neighbor breadth >= 60%;
- bucket excess over the market median >= 0.20%.

Two hypotheses are tested as one family:

1. `GBM1_BROAD_BUCKET_CONTINUATION`: long the neighbor bucket when the common state turns
   false-to-true.
2. `GBM2_BUCKET_TURN_REVERSAL`: short the neighbor bucket when the common state is true but
   its 15-minute return is no longer positive.

Each source has a four-hour cooldown. At most three source buckets are accepted per timestamp,
ranked by bucket excess return. No thresholds may be tuned after looking at outcomes.

## Costs and controls

- Explicit total round-trip costs: 20, 30, and 50 bp per bucket sleeve.
- Fifty within-month random-neighbor graph controls, preserving source count and bucket size.
- One-day shifted signal placebo with contemporaneous outcomes unchanged.
- Two-thousand day-block bootstrap samples.
- The family is judged against the best candidate produced by every random graph.

## Promotion gates

A candidate is only eligible for forward watch if all are true:

- at least 100 full-period, 25 validation, and 25 holdout portfolio observations;
- full, validation, and holdout mean net return at 20 bp are positive;
- real result is at or above the 90th percentile of the random-graph family maximum;
- real result beats the one-day shifted placebo;
- bootstrap 95% lower bound is positive;
- no single positive month supplies more than 35% of positive PnL.

Failure leaves all PaperLive and strategy permissions unchanged.

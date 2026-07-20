# v11.1 Sparse Topology-Continuation Preregistration

## Status and scope

This is a result-informed mechanism follow-up to v11.0. The v11.0 validation and holdout outcomes
have already been inspected, so this run is retrospective research even if every numerical gate
passes. It cannot authorize PaperLive. New forward observations are required for promotion.

The frozen question is whether the gross continuation found after balanced-community topology
breaks becomes cost-surviving when entries are restricted by a severity threshold learned only from
prior months, and whether the edge persists beyond four hours.

## Frozen graph and base event

- Reuse the v11.0 monthly graph: trailing 30 calendar days, hourly BTC residuals, at least 500
  complete rows, eight deterministic recursive-spectral communities of nine coins.
- Reuse the v11.0 break event: 12-hour average standardized pairwise cross-product crossing below
  its trailing-history fifth percentile, false-to-true transitions only, four-hour cooldown.
- Rank community members by trailing four-hour residual return. The only directional candidate is
  continuation: long the top third and short the bottom third, normalized to 0.5 plus 0.5.

## Frozen sparse rule

At each month start, pool base-event break severities from all strictly earlier evaluated months.
After at least 100 prior events exist, set the entry threshold to their expanding 80th percentile.
The threshold is frozen for that month. Months without 100 prior events do not trade. At most three
simultaneous communities are retained by severity.

The 80th percentile is explicitly motivated by the post-hoc v11.0 severity diagnostic. It is not
an independently discovered threshold, which is why this run remains retrospective.

## Horizon and attribution matrix

Construct future simple returns by compounding the hourly returns at t+1 through t+H for fixed
horizons H = 1, 2, 4, 8, and 12 hours. Residualize every horizon with the month-frozen BTC beta.
Report for every horizon:

- normalized long-top/short-bottom residual spread;
- the long-top residual contribution and short-bottom residual contribution separately;
- 20, 30, and 50 bp round-trip spread costs;
- observation count and active-day/month counts for development, validation, and holdout labels.

Leg results are attribution only; they are not independently promotable candidates because a
standalone leg requires an additional BTC hedge and a separate execution model.

## Controls and decision rule

- Fifty random monthly partitions preserving the exact eight-by-nine size structure. Each random
  partition learns its own expanding historical severity threshold.
- A one-day (24 hourly bars) shifted-signal control using the same sparse rule.
- For each random iteration, compare against the best net-20 result across all five horizons. This
  family maximum corrects the real horizon search.
- Two thousand entry-day bootstrap resamples, five chronological slices, and positive-PnL month and
  community concentration limits of 35%.

A horizon is economically viable only with at least 100 observations, at least 25 validation and
holdout observations, positive validation and holdout net 20 bp, positive full net 30 bp, at least
the 90th percentile of the random family maximum, superiority to the same-horizon shifted control,
positive bootstrap lower bound, no negative chronological slice, and both concentration gates.

Passing yields `retrospective_forward_watch_only`, never PaperLive. Failure closes this exact
expanding-80th-percentile sparse rule, not the underlying topology state variable.

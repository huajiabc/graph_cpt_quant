# v23.32 Sparse Volatility-Tail Selector Preregistration

Status: frozen before running the 17-feature tail search. The same historical event
payoffs have been examined in earlier branches, so a pass can only nominate an
isolated forward-shadow candidate, never confirm alpha.

## Fixed universe and payoff

- Feature artifact and hash are the v23.29 159-event volatility-transmission matrix,
  hash `C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF`.
- Exactly the 17 ordered features in v23.29 metadata are eligible.
- Payoff remains the fixed 0.75-sigma, four-hour OCO with 10 bp primary and 20 bp
  stress round-trip costs.
- No combinations, transforms, barrier changes, direction choices, or horizons are
  searched.

## Fixed sparse selection algorithm

For every feature, create two candidates: values at or above its training 70th
percentile and values at or below its training 30th percentile. This gives exactly
34 candidates. Inside a training set, rank candidates by mean primary strategy
return per training opportunity, assigning zero to unselected events. Ties resolve
by v23.29 feature order and then high tail before low tail.

- Fit on development, apply the single winning rule to validation.
- Refit on development plus validation, apply its single winning rule to holdout.
- Quantiles and the winning feature/orientation use training data only.
- The algorithm may select a negative training winner; there is no outcome-dependent
  abstention or threshold change.

## Multiple-testing control and gates

One thousand random-label iterations independently permute the development and
development-plus-validation targets, repeat the complete 34-candidate search, and
apply each random winner to the actual next-period returns. The observed combined
OOS opportunity return must be at or above this null's 95th percentile.

Reject unless all gates pass:

1. At least 8 selected trades in validation and 8 in holdout.
2. Selected primary expectancy is positive in both periods.
3. Selected stress expectancy is positive in both periods.
4. Primary opportunity return is positive in both periods.
5. Stress opportunity return is positive in both periods.
6. Primary opportunity return exceeds unfiltered OCO in validation, holdout, and OOS.
7. The same feature and tail orientation win both temporal fits.
8. Full-search random-label percentile is at least 95.
9. A 5,000-draw calendar-month bootstrap has positive 5th percentile.
10. At least five active months, at least 60% positive active months, and every
    leave-one-month-out opportunity mean positive.

Failure means `rejected_sparse_volatility_tail_selector`. Passing means only
`research_candidate_requires_isolated_forward_shadow`. No live, PaperLive, leverage,
remote, application, or order change is authorized.

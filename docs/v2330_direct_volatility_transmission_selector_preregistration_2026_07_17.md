# v23.30 Direct Volatility-Transmission Selector Preregistration

Status: frozen before joining v23.29 features to the OCO payoff in this branch.
The aggregate 0.75-sigma OCO history was examined in earlier branches, so this is
an incremental hypothesis test, not a pristine new holdout. Crucially, neither the
score composition nor its cutoff uses any outcome.

## Input and payoff freeze

- Feature artifact: `reports/v23_29_event_volatility_transmission_feature_audit/event_volatility_transmission_features.parquet`.
- Feature hash: `C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF`.
- Events and development/validation/holdout labels remain the fixed 159-event set.
- Payoff is only the already-frozen 0.75 causal-hourly-sigma, four-hour OCO from
  v23.4, with its 10 bp primary and 20 bp stress round-trip cost columns.
- The selector controls trade/no-trade only. It does not choose barrier, direction,
  horizon, leverage, or event family.

## Outcome-free score

Exactly four causal price-volatility features are used with positive sign:

1. `btc_receiver_gap`: top residual-volatility leaders' current shock minus BTC's
   current standardized absolute move.
2. `alt_rv_acceleration_median`: median 4h realized volatility relative to its
   24h same-scale level.
3. `alt_residual_shock_breadth`: fraction of 16 alts with a beta-residual shock of
   at least one prior-30-day sigma.
4. `directed_edge_weight_mean`: mean positive, direction-advantaged alt-to-BTC
   one-hour volatility-edge weight estimated strictly before entry.

For validation, each feature is standardized with development-only mean and
population standard deviation. Their equal-weight mean is the transmission score,
and the cutoff is the development score's 70th percentile. For holdout, the same
calculation is refit on development plus validation features, still without using
returns. Zero-variance scales become one. An event is selected when its score is at
or above the training feature distribution's 70th percentile. No feature, sign,
weight, or percentile is searched.

## Frozen rejection gates

Reject unless every gate passes:

1. At least 8 selected trades in validation and 8 in holdout.
2. Selected primary expectancy is positive in validation and holdout.
3. Selected stress expectancy is positive in validation and holdout.
4. Primary opportunity return (zero for unselected events) is positive in both.
5. Stress opportunity return is positive in both.
6. Primary opportunity return exceeds the unfiltered 0.75-sigma OCO in validation,
   holdout, and combined OOS.
7. Score/primary-return Spearman IC is positive in validation and holdout.
8. Combined OOS opportunity return is at or above the 95th percentile of 1,000
   random same-count selections performed separately inside validation and holdout.
9. The 5th percentile of a 5,000-draw OOS calendar-month bootstrap is positive.
10. At least five OOS months contain a selection, at least 60% of active months are
    positive, and every leave-one-month-out opportunity mean is positive.

Passing means only `research_candidate_requires_isolated_forward_shadow` because
the historical event set has been seen. Failure means
`rejected_direct_volatility_transmission_selector`. Neither outcome authorizes live,
PaperLive, leverage, remote, application, or order changes.

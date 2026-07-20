# v23.27 Multisource Interaction-Ridge OCO Selector Preregistration

Status: frozen before loading any OCO outcome in this branch.

## Input freeze

- Candidate: `MSM1_MULTISOURCE_INTERACTION_RIDGE_OCO_SELECTOR`.
- Feature artifact: `reports/v23_26_multisource_oco_model_feature_audit/multisource_model_features.parquet`.
- Feature hash: `4B93B3F7A5D340776CF0CDAA5E16C37AFACF6A20AFC93ED25C74DC2BB393B081`.
- Exactly the 19 ordered features stored in the v23.26 metadata are allowed.
- The 159 events and their `development` / `validation` / `holdout` labels are fixed.
- The sole 15-of-16 derivatives cross-section remains in validation; it is not imputed.

## Fixed payoff and costs

- Join only the already-frozen `0.75` causal-hourly-sigma OCO outcome from
  `reports/v23_4_book_vacuum_oco_breakout/barrier_variant_outcomes.parquet`.
- Horizon and first-touch direction selection remain those of v23.4; no barrier,
  horizon, side, or event subset is retuned.
- Primary target is the v23.4 10 bp round-trip net return. Stress return is its
  frozen higher-cost column. The model never chooses direction; it only chooses
  whether an event is traded.

## Fixed models

Two deterministic models are evaluated.

1. Linear baseline: the 19 training-standardized features, ridge alpha `10`.
2. Primary interaction model: the 19 standardized features plus all 190 degree-two
   terms (squares and pairwise products), standardized again on training data,
   ridge alpha `100`.

Each ridge has an unpenalized intercept. Zero-variance columns are assigned scale
one. No alpha, feature, interaction, ensemble weight, or prediction cutoff is
searched. An event is selected exactly when predicted primary net return is above
zero.

## Temporal evaluation

- Fit development (63 events), predict validation (47 events).
- Refit development plus validation (110 events), predict holdout (49 events).
- All reported selection performance is therefore out of training time.
- The primary score is mean realized strategy return per OOS opportunity, assigning
  zero to unselected events. Selected-trade expectancy is reported separately.

## Frozen rejection gates

The interaction model is rejected unless every gate passes:

1. At least 8 selected trades in validation and 8 in holdout.
2. Selected-trade primary expectancy is positive in validation and holdout.
3. Selected-trade stress expectancy is positive in validation and holdout.
4. Primary opportunity return is positive in validation and holdout.
5. Stress opportunity return is positive in validation and holdout.
6. Primary prediction/return Spearman IC is positive in validation and holdout.
7. Interaction primary opportunity return is strictly above the linear ridge in
   validation, holdout, and combined OOS.
8. Interaction primary opportunity return is strictly above the unfiltered fixed
   OCO in validation, holdout, and combined OOS.
9. At least 5 OOS calendar months contain a selected trade and at least 60% of
   active months have positive primary opportunity return.
10. Every leave-one-OOS-month-out primary opportunity mean is positive.
11. A 5,000-draw calendar-month bootstrap has a strictly positive 5th percentile
    for primary opportunity return.
12. The observed combined-OOS primary opportunity return is at or above the 95th
    percentile of 1,000 training-label permutations. Each permutation repeats both
    temporal fits and uses the same zero cutoff.

Failing any gate means `rejected_no_incremental_complex_model_alpha`. Passing all
gates means only `research_candidate_requires_isolated_forward_shadow`; it does not
authorize PaperLive, leverage, deployment, or order changes.

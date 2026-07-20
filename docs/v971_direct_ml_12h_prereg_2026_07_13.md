# v9.7.1 Direct Cross-Sectional ML Alpha, 12h Addendum

## Sequential-search disclosure

This addendum was written after the frozen v9.7 four-hour run completed. That
run produced only a very small positive XGBoost IC and negative validation,
holdout, and full-OOS Top5 excess after costs. The best ablation removed the
funding/OI block. The four-hour result is therefore fixed and may not be
relabelled using the 12-hour outcome.

The sole new question is whether the same information becomes economically
usable at a lower turnover frequency.

## Frozen changes from v9.7

- Label: same-timestamp demeaned `future_ret_12h`.
- Decision timestamps: 00:00 and 12:00 UTC only, giving non-overlapping
  twelve-hour outcomes.
- Embargo: 12 hours before each evaluation month.
- Complete-month cutoff remains 2026-06-01.
- Dynamic monthly Top30, minimum 20-symbol cross-section, Top5 portfolio,
  walk-forward months, feature set, model hyperparameters, costs, controls,
  and complexity gates remain unchanged.
- Report root: `reports/v9_7_1_direct_ml_alpha_12h`.

No 12-hour hyperparameter tuning, threshold search, top-k search, or feature
addition is allowed. This is the last horizon tested in this direct-ML round.

## Interpretation

- Passing all gates can only yield `research_candidate_only` because May 2026
  remains the sole complete final holdout month.
- If XGBoost does not beat Ridge in both validation and holdout, there is no
  case for added complexity.
- If no model has positive validation and holdout excess at 20bp, the current
  as-of price/volume/funding/OI feature set is not a direct tradable selector
  at either four or twelve hours.

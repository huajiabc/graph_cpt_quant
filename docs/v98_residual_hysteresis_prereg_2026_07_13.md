# v9.8 Market-Neutral Residual Alpha and Hysteresis - Pre-Registration

## Scope and permission

This is an offline historical research lane. It cannot change P2, shadow,
paper-live, canary-live, or real-live permissions.

The frozen question is whether changing the prediction target from raw
cross-sectional return to BTC-beta-neutral residual return, then applying a
rank-band no-trade zone, can produce a lower-turnover tradable alpha from the
existing as-of feature set. Model complexity is secondary: it is eligible only
if residual Ridge first establishes a stable economic baseline.

## Frozen data and chronology

- Source: `data/processed/v0_3/perp_pressure_features_all_eligible.parquet`.
- Universe: `universe_dynamic_monthly_top30 == True`; static-current universe
  fields are forbidden.
- Hourly observations are used only to estimate trailing beta. Predictions are
  made at non-overlapping four-hour or twelve-hour anchors.
- Sample: 2025-07-01 UTC through 2026-06-01 UTC, excluding incomplete June.
- Minimum prediction cross-section: 20 symbols.
- Walk-forward months and disclosure order are unchanged from v9.7:
  January-February development, March-April validation, May holdout.
- Training uses only earlier calendar months and applies an embargo equal to
  the prediction horizon.

## Frozen residual target

For symbol `i` and horizon `H`:

1. Estimate `beta_i,t` from trailing four-hour symbol and BTC returns using the
   latest 30 calendar days of hourly as-of observations, with at least 240
   observations. The current trailing return may be used because it is known at
   decision time. Clip beta to `[-1, 3]` only for robustness.
2. Compute realized hedged return
   `residual_i,t+H = future_ret_i,H - beta_i,t * future_ret_BTC,H`.
3. Demean residual return inside the same dynamic Top30 timestamp. The fitted
   label is this relative residual, not raw future return.

BTC future return is used only to construct the realized target and portfolio
ledger. It is never an input feature.

## Frozen feature and model ladder

The v9.7 as-of price, volume, funding, OI, and BTC-regime inputs are retained.
Two derived as-of symbol features are added:

- cross-sectional rank of trailing 30-day BTC beta;
- cross-sectional rank of trailing four-hour beta-neutral return.

Models:

1. `residual_momentum`: trailing residual-return rank, no fit.
2. `ridge_residual`: Ridge alpha 10 fitted to the residual target.
3. `xgb_residual`: the unchanged v9.7 shallow XGBoost specification fitted to
   the residual target.
4. `xgb_residual_shuffled`: identical model with labels permuted separately in
   each training month.
5. `ridge_direct_control` and `xgb_direct_control`: identical expanded inputs
   fitted to the old same-timestamp raw relative-return target.

No hyperparameter search, symbol identity, embeddings, stacking, neural model,
or post-result threshold change is allowed.

## Frozen portfolios and no-trade zone

Every model is evaluated in two normalized books:

- `long_top5`: equal-weight Top5 residual selections minus the timestamp mean
  residual return;
- `long_short_5x5`: 0.5 long Top5 and 0.5 short Bottom5 by model score.

Each book uses three policies:

- `refresh`: rebuild the relevant Top5/Bottom5 each anchor;
- `sticky_10`: retain a long while it remains in Top10 and a short while it
  remains in Bottom10, filling vacancies from the strongest candidates;
- `sticky_10_min2`: the same rank band plus a minimum two-anchor hold unless a
  name leaves the available cross-section.

The Top5-to-Top10 retention band is the frozen low-confidence no-trade zone.
No score-spread threshold will be selected from validation or holdout data.

Turnover is the replaced fraction of the long book for `long_top5`, and the
average of long- and short-side replaced fractions for `long_short_5x5`.
Costs of 10, 20, 30, and 50 bps apply to normalized full-book turnover. Initial
entry is charged. The focal metric is additive net residual excess at 20 bps;
30 bps is the promotion stress.

## Frozen controls

- Direct-target models using identical features and splits.
- Monthly shuffled-label XGBoost.
- 500 random-score portfolio paths under every frozen book/policy combination,
  including the identical hysteresis mechanics.
- Monthly, validation, holdout, score-quintile, turnover, break-even-cost,
  symbol-contribution, and month-concentration summaries.
- The 4h experiment is read first. The 12h run is a preregistered robustness
  horizon, not an opportunity to tune the 4h result.

## Decision gates

A residual signal/policy is only a historical research candidate if all hold:

1. validation and May holdout net20 are both positive;
2. full-OOS net30 is positive;
3. at least three of five OOS months have positive net20;
4. no month supplies more than 35% of total positive net20;
5. average turnover is at most 35%;
6. the result exceeds the matched random-control 90th percentile;
7. it beats the corresponding direct-target control;
8. for XGBoost, it also beats residual Ridge and shuffled-label XGBoost in
   validation and holdout, and its five full-OOS score-bucket mean residuals
   are non-decreasing with the highest bucket strictly above the lowest.

Passing these gates still grants only `research_candidate_only`, because there
is one complete holdout month. Failure of the residual Ridge baseline means
there is no permission to escalate model complexity on the existing feature
block.

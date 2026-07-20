# v14.4 Bearish Graph-ML Convergence Preregistration

Date frozen: 2026-07-15, after inspecting v14.3 and before building or inspecting any v14.4 pair
return, model prediction, or portfolio.

## Adaptive status and hypothesis

This is a second-stage adaptive study. v14.3 found that 12-hour graph convergence was concentrated
after bearish community releases, but the frozen subset was too small and failed bootstrap and
holdout gates. v14.4 asks whether a walk-forward nonlinear ranker can identify the stronger pairs
inside a broader, causally defined bearish event pool. Historical success can grant provisional
forward-shadow status only.

## Frozen pair pool

- Reuse v14.3's monthly causal communities, BTC betas, volatility receiver graph, and 12-hour
  source-versus-receiver 50/50 spread construction.
- A bearish source requires return z at most -1.5, volatility z at least 0.5, negative breadth at
  least 60%, and at least five members.
- A graph receiver is eligible when absolute return z is at most 1.25, volatility z is at most 0.5,
  and at least five members are observed.
- Create one row per eligible source-edge-receiver pair. Target is the 12-hour BTC-residual spread:
  long the shocked source community and short the receiver community, 50/50 by leg.

## Frozen model

Features are source return z, source volatility z, source breadth, receiver return z, receiver
volatility z, receiver breadth, edge weight, volatility Spearman, magnitude advantage, and
source-minus-receiver return-z gap.

At each month start, fit `HistGradientBoostingRegressor` on all earlier pair rows whose 12-hour
target is fully realized. Require at least 200 training rows and 50 distinct training days. Freeze
`max_depth=2`, `learning_rate=0.05`, `max_iter=100`, `min_samples_leaf=20`, and
`l2_regularization=5`. Weight rows inversely by the number of pairs at their timestamp. No
hyperparameter search is allowed.

During the month, select the highest predicted pair at each timestamp only when predicted residual
gross return exceeds 40 bp, then apply a 12-hour cooldown. Candidate:
`GML1_BEARISH_VOL_CONVERGENCE_12H`.

Primary return is residual net40; naked net20/net30 are secondary.

## Controls and gates

- Static edge-weight-times-shock ranking, reversed edges, and a 24-hour delayed full node state are
  mandatory controls.
- Fifty training-label permutations preserve training-month membership and all features, graphs,
  pool rows, deployment months, threshold, and cooldown.
- Two-thousand entry-day block bootstrap draws use residual net40.
- Provisional passage requires at least 80 full, 20 validation, and 20 holdout observations;
  positive residual net40 and naked net20 in development, validation, and holdout; positive full
  naked net30; positive bootstrap lower bound; at least the 95th label-null percentile; superiority
  to static, reversed-edge, and delayed-state controls; positive month contribution concentration
  at most 35%; and worst period mean no worse than -40 bp.

No PaperLive, leverage, or live-order permission is granted, regardless of historical outcome.

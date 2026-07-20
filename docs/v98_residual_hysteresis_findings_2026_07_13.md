# v9.8 Market-Neutral Residual Alpha and Hysteresis - Findings

## Decision

The BTC-beta-neutral residual target and frozen rank-band hysteresis do not
produce a tradable alpha at either four or twelve hours.

Status: `reject_residual_hysteresis_no_complexity_case`. This result does not
change P2, shadow, paper-live, canary-live, or real-live permissions.

The experiment separates two conclusions:

1. The portfolio mechanism worked as designed. `sticky_10_min2` reduced average
   turnover from roughly 70-73% to 31-34%.
2. The residual signal did not survive validation, holdout, and costs. Lower
   turnover improved a losing strategy but did not create alpha.

All return totals below are additive OOS ledger sums rather than compounded
capital returns.

## Four-hour result

Focal book: normalized long Top5 / short Bottom5 residual portfolio with the
Top10 retention band and minimum two-anchor hold.

| model | gross residual excess | net20 | net30 | turnover | random percentile | validation net20 | May holdout net20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual Ridge | -19.10% | -81.20% | -112.25% | 34.27% | 25.8% | -39.63% | -18.66% |
| residual XGBoost | +17.62% | -44.42% | -75.44% | 34.24% | 96.2% | -8.37% | -16.21% |
| direct-target XGBoost control | -5.85% | -63.19% | -91.86% | 31.64% | n/a | -20.49% | -19.13% |

Residualizing the target materially improved shallow XGBoost relative to the
identical direct-target control. It also beat 96.2% of matched random-policy
paths. That is useful evidence that the target definition is cleaner.

It is not deployable evidence. The residual XGBoost break-even cost was only
5.68 bps, four of five OOS months were negative at 20 bps, and the sole May
holdout was negative even before cost. Full-OOS score buckets were not
monotonic: bucket two had the best mean residual while the highest bucket was
slightly negative.

The no-trade zone improved residual XGBoost net20 from -122.76% under full
refresh to -44.42%, while turnover fell from 72.79% to 34.24%. This is a large
execution improvement, but the absolute result still fails every economic
promotion gate.

## Twelve-hour result

The best real-label residual combination was Ridge, not XGBoost.

| model | gross residual excess | net20 | net30 | turnover | random percentile | validation net20 | May holdout net20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual Ridge, long-short sticky | +4.99% | -13.93% | -23.39% | 31.32% | 77.6% | -7.43% | -8.81% |
| residual XGBoost, long-short sticky | -33.30% | -54.30% | -64.80% | 34.77% | 3.4% | -35.58% | -17.83% |
| direct-target Ridge control | -15.43% | -32.45% | -40.96% | 28.18% | n/a | -22.65% | -3.90% |

Residual Ridge improved the full and validation results relative to the direct
control, but its 5.28-bp break-even cost remained far below the frozen cost
range and its May holdout deteriorated.

Residual XGBoost had negative mean IC (-0.0038) and strongly negative May IC
(-0.0819). Its score ordering was directionally inverted: the lowest full-OOS
bucket had positive mean residual and the highest bucket had negative mean
residual.

The strongest twelve-hour output came from shuffled-label XGBoost in the long
Top5 sticky book: +7.82% at 20 bps, but -2.92% at 30 bps and -10.30% in May.
This is a negative control, not a candidate, and demonstrates how easily a
favorable aggregate can arise from path luck.

## Assessment of model complexity

There is no current case for a more complex learner on this information block:

- residual Ridge did not establish a positive validation/holdout baseline;
- shallow XGBoost helped only at four hours and still failed absolute economics;
- at twelve hours XGBoost was materially worse than Ridge and the shuffled
  control;
- neither horizon produced monotonic score buckets;
- the best real-label break-even costs were 5-6 bps, below the 10-30-bp stress
  range.

A learning-to-rank tree, neural sequence model, symbol embedding, stacking, or
hyperparameter search would currently add capacity to an unstable relationship.
Complexity should remain closed until a new information block or untouched
forward sample first makes the frozen Ridge/shallow-tree baseline positive.

## What remains useful

The experiment did produce two reusable research results:

1. BTC-beta-neutral labels are cleaner than raw relative-return labels in some
   four-hour comparisons, so residualization should remain a standard control
   for future orthogonal features.
2. The Top10 retention band plus two-anchor minimum hold reliably cuts turnover
   by about half. It should remain an offline portfolio-policy baseline for new
   alpha sources, but it has no independent trading permission.

The next credible source of improvement is new information, especially the
preregistered synchronized cross-venue orderflow tape, rather than another
estimator fitted to the current bar/funding/OI panel.

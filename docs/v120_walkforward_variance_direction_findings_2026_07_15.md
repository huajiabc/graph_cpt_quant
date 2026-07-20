# v12.0 Walk-Forward Variance Direction Findings

Date: 2026-07-15

Verdict: `reject_walkforward_variance_direction_model`. This was a
preregistered second look and could not promote on reused history.

## Result

After the 30-day label purge and 24-month minimum training window, the ridge
model produced 27 genuine walk-forward predictions.

| Period | n | Long-vol share | Net 1 vol | Net 2 vol | Always-short net 1 vol | Spread-rule net 1 vol |
|---|---:|---:|---:|---:|---:|---:|
| Full | 27 | 22.22% | -7.67% | -11.85% | -3.19% | +0.61% |
| Development | 9 | 0.00% | +3.94% | +0.38% | +3.94% | +21.26% |
| Validation | 6 | 33.33% | -26.20% | -30.16% | -0.82% | -2.80% |
| Holdout | 12 | 33.33% | -7.12% | -11.86% | -9.73% | -13.18% |

The model's bootstrap interval was -26.30% to +10.87%, its random-direction
percentile was 36.35%, and normalized maximum drawdown was 526.85%.

## Interpretation

The model occasionally avoided part of the recent unconditional short-vol
loss, but its long/short timing was worse than always-short overall and much
worse than the simple spread-sign rule. Adding a small regression model to the
same DVOL and trailing-RV primitives does not create alpha. More flexible
boosting, neural, or graph models would raise selection variance while the
effective sample remains about five dozen non-overlapping volatility labels.

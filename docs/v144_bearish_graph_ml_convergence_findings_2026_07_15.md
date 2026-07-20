# v14.4 Bearish Graph-ML Convergence Findings

Verdict: `reject_bearish_graph_ml_candidate`.

This is an adaptive walk-forward study and cannot independently establish alpha.

## Audit

| candidate                        | eligible   | verdict                           |   full_residual_net40 |   development_residual_net40 |   validation_residual_net40 |   holdout_residual_net40 |   full_raw_net20 |   full_raw_net30 |   static_residual_net40 |   reversed_residual_net40 |   delayed_residual_net40 |   label_null_percentile |   bootstrap_ci_low |   bootstrap_ci_high |   max_positive_month_share |   worst_period_mean | failed_gates                                                                                                                                                                                                |
|:---------------------------------|:-----------|:----------------------------------|----------------------:|-----------------------------:|----------------------------:|-------------------------:|-----------------:|-----------------:|------------------------:|--------------------------:|-------------------------:|------------------------:|-------------------:|--------------------:|---------------------------:|--------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GML1_BEARISH_VOL_CONVERGENCE_12H | False      | reject_bearish_graph_ml_candidate |             -0.000935 |                     0.000661 |                    0.000868 |                -0.003672 |         0.000153 |        -0.000847 |               -0.003414 |                 -0.008482 |                -0.004476 |                0.980000 |          -0.004494 |            0.003054 |                   0.593257 |           -0.003672 | full_observations_80|validation_observations_20|holdout_observations_20|holdout_residual_net40_positive|holdout_raw_net20_positive|full_raw_net30_positive|bootstrap_lower_positive|month_share_below_35pct |

## Summary

| scope       | candidate                        |   observations |   active_days |   active_months |   mean_raw_gross_12h |   mean_raw_net_12h_20bp |   mean_raw_net_12h_30bp |   mean_residual_gross_12h |   mean_residual_net_12h_40bp |
|:------------|:---------------------------------|---------------:|--------------:|----------------:|---------------------:|------------------------:|------------------------:|--------------------------:|-----------------------------:|
| all         | GML1_BEARISH_VOL_CONVERGENCE_12H |             34 |            34 |               7 |             0.002153 |                0.000153 |               -0.000847 |                  0.003065 |                    -0.000935 |
| development | GML1_BEARISH_VOL_CONVERGENCE_12H |             11 |            11 |               2 |             0.002472 |                0.000472 |               -0.000528 |                  0.004661 |                     0.000661 |
| validation  | GML1_BEARISH_VOL_CONVERGENCE_12H |             10 |            10 |               2 |             0.004681 |                0.002681 |                0.001681 |                  0.004868 |                     0.000868 |
| holdout     | GML1_BEARISH_VOL_CONVERGENCE_12H |             13 |            13 |               3 |            -0.000062 |               -0.002062 |               -0.003062 |                  0.000328 |                    -0.003672 |

Eligible pair rows: `2329`; walk-forward predicted rows: `1930`.

No PaperLive, leverage, or live-order permission changed.

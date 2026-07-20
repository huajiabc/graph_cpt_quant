# v14.5 Bearish Convergence Horizon-Extension Findings

Verdict: `reject_bearish_horizon_family`.

This is an adaptive horizon extension and cannot independently establish alpha.

## Audit

| candidate                    | eligible   | verdict                          |   full_residual_net40 |   development_residual_net40 |   validation_residual_net40 |   holdout_residual_net40 |   full_raw_net20 |   full_raw_net30 |   delayed_residual_net40 |   reversed_residual_net40 |   random_family_percentile |   bootstrap_ci_low |   bootstrap_ci_high |   max_positive_month_share |   worst_period_mean | failed_gates                                                                                                                                                                                                                                  | family_verdict                |
|:-----------------------------|:-----------|:---------------------------------|----------------------:|-----------------------------:|----------------------------:|-------------------------:|-----------------:|-----------------:|-------------------------:|--------------------------:|---------------------------:|-------------------:|--------------------:|---------------------------:|--------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------|
| BCH1_BEARISH_CONVERGENCE_18H | False      | reject_bearish_horizon_candidate |             -0.003105 |                     0.001365 |                   -0.009787 |                -0.001806 |        -0.001018 |        -0.002018 |                -0.004213 |                 -0.005019 |                   0.520000 |          -0.006358 |            0.000347 |                   0.613347 |           -0.009787 | full_observations_50|validation_residual_net40_positive|holdout_residual_net40_positive|validation_raw_net20_positive|full_raw_net30_positive|bootstrap_lower_positive|random_family_p95|month_share_below_35pct|worst_period_above_minus40bp | reject_bearish_horizon_family |
| BCH2_BEARISH_CONVERGENCE_24H | False      | reject_bearish_horizon_candidate |             -0.002913 |                     0.000995 |                   -0.011346 |                 0.000448 |        -0.001347 |        -0.002347 |                -0.002876 |                 -0.003775 |                   0.580000 |          -0.007103 |            0.001409 |                   0.715690 |           -0.011346 | full_observations_50|validation_residual_net40_positive|validation_raw_net20_positive|full_raw_net30_positive|bootstrap_lower_positive|random_family_p95|beats_delayed|month_share_below_35pct|worst_period_above_minus40bp                   | reject_bearish_horizon_family |

## Summary

| scope       | candidate                    |   observations |   active_days |   active_months |   mean_raw_gross |   mean_raw_net20 |   mean_raw_net30 |   mean_residual_gross |   mean_residual_net40 |
|:------------|:-----------------------------|---------------:|--------------:|----------------:|-----------------:|-----------------:|-----------------:|----------------------:|----------------------:|
| all         | BCH1_BEARISH_CONVERGENCE_18H |             42 |            42 |              12 |         0.000982 |        -0.001018 |        -0.002018 |              0.000895 |             -0.003105 |
| all         | BCH2_BEARISH_CONVERGENCE_24H |             40 |            40 |              12 |         0.000653 |        -0.001347 |        -0.002347 |              0.001087 |             -0.002913 |
| development | BCH1_BEARISH_CONVERGENCE_18H |             13 |            13 |               5 |         0.006654 |         0.004654 |         0.003654 |              0.005365 |              0.001365 |
| development | BCH2_BEARISH_CONVERGENCE_24H |             13 |            13 |               5 |         0.004813 |         0.002813 |         0.001813 |              0.004995 |              0.000995 |
| validation  | BCH1_BEARISH_CONVERGENCE_18H |             12 |            12 |               3 |        -0.006741 |        -0.008741 |        -0.009741 |             -0.005787 |             -0.009787 |
| validation  | BCH2_BEARISH_CONVERGENCE_24H |             12 |            12 |               3 |        -0.008232 |        -0.010232 |        -0.011232 |             -0.007346 |             -0.011346 |
| holdout     | BCH1_BEARISH_CONVERGENCE_18H |             17 |            17 |               4 |         0.002096 |         0.000096 |        -0.000904 |              0.002194 |             -0.001806 |
| holdout     | BCH2_BEARISH_CONVERGENCE_24H |             15 |            15 |               4 |         0.004154 |         0.002154 |         0.001154 |              0.004448 |              0.000448 |

Graph months: `12`; observations: `82`.

No PaperLive, leverage, or live-order permission changed.

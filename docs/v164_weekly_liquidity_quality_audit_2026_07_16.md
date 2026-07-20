# v16.4 Independent Audit of v16.3 LQ1

Verdict: `rejection_confirmed`.

|   feature_audit_pass |   portfolio_audit_pass |   rejection_confirmed |
|---------------------:|-----------------------:|----------------------:|
|                    1 |                      1 |                     1 |

|   feature_rows |   max_abs_depth_hours_difference |   max_abs_volume_hours_difference |   max_abs_depth_1pct_difference |   max_abs_depth_5pct_difference |   max_abs_mean_volume_difference |   max_abs_quality_1pct_difference |   max_abs_quality_5pct_difference |
|---------------:|---------------------------------:|----------------------------------:|--------------------------------:|--------------------------------:|---------------------------------:|----------------------------------:|----------------------------------:|
|      8.160e+02 |                        0.000e+00 |                         0.000e+00 |                       0.000e+00 |                       0.000e+00 |                        0.000e+00 |                         0.000e+00 |                         0.000e+00 |

|   portfolio_weeks | long_pairs_exact   | short_pairs_exact   |   max_abs_turnover_difference |   max_abs_gross_return_difference |   max_abs_primary_return_difference |   max_abs_stress_return_difference |   max_abs_residual_btc_beta |   max_abs_gross_notional_drift |
|------------------:|:-------------------|:--------------------|------------------------------:|----------------------------------:|------------------------------------:|-----------------------------------:|----------------------------:|-------------------------------:|
|                51 | True               | True                |                     2.220e-16 |                         1.388e-17 |                           1.388e-17 |                          1.388e-17 |                   2.151e-16 |                      4.441e-16 |

The audit independently rebuilt all 784 weekly depth/volume quality
features, pair directions and 51 portfolio weeks. The rejection is
confirmed; positive stale and 5% diagnostics are temporally unstable.
PaperLive and remote state are unchanged.

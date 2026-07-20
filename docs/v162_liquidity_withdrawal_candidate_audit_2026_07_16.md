# v16.2 Independent Audit of v16.1 LW1

Verdict: `rejection_confirmed`.

|   raw_depth_audit_pass |   score_audit_pass |   portfolio_audit_pass |   rejection_confirmed |
|-----------------------:|-------------------:|-----------------------:|----------------------:|
|                      1 |                  1 |                      1 |                     1 |

|   raw_depth_samples | snapshot_counts_match   |   max_abs_total_depth_difference |
|--------------------:|:------------------------|---------------------------------:|
|                  80 | True                    |                        0.000e+00 |

|   score_rows |   max_abs_withdrawal_percentile_difference |   max_abs_prior_residual_difference |   max_abs_score_difference |
|-------------:|-------------------------------------------:|------------------------------------:|---------------------------:|
|    1.448e+05 |                                  0.000e+00 |                           0.000e+00 |                  0.000e+00 |

|   portfolio_hours |   max_abs_turnover_difference |   max_abs_gross_return_difference |   max_abs_primary_return_difference |   max_abs_stress_return_difference |   max_abs_residual_btc_beta |   max_abs_gross_notional_drift |
|------------------:|------------------------------:|----------------------------------:|------------------------------------:|-----------------------------------:|----------------------------:|-------------------------------:|
|         9.047e+03 |                     6.661e-16 |                         4.770e-18 |                           5.204e-18 |                          5.204e-18 |                   3.331e-16 |                      4.441e-16 |

The audit independently replayed total-depth windows, rebuilt every
withdrawal rank and residual-price product, and recomputed all portfolio
returns and costs. The zero-gross-edge rejection is confirmed.
PaperLive and remote state are unchanged.

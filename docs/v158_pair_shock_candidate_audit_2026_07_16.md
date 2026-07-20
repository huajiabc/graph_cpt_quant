# v15.8 Independent Audit of v15.7 VT4

Verdict: `rejection_confirmed`.

|   signal_audit_pass |   portfolio_audit_pass |   rejection_confirmed |
|--------------------:|-----------------------:|----------------------:|
|                   1 |                      1 |                     1 |

|   signal_rows | source_exact   | receiver_exact   | source_lag_exact   |   max_abs_source_residual_difference |   max_abs_receiver_fragility_difference |   max_abs_propagation_strength_difference |
|--------------:|:---------------|:-----------------|:-------------------|-------------------------------------:|----------------------------------------:|------------------------------------------:|
|          3000 | True           | True             | True               |                            0.000e+00 |                               0.000e+00 |                                 0.000e+00 |

|   portfolio_days |   max_abs_turnover_difference |   max_abs_gross_return_difference |   max_abs_primary_return_difference |   max_abs_stress_return_difference |   max_abs_residual_btc_beta |   max_abs_gross_notional_drift |
|-----------------:|------------------------------:|----------------------------------:|------------------------------------:|-----------------------------------:|----------------------------:|-------------------------------:|
|        3.750e+02 |                     4.441e-16 |                         6.939e-18 |                           6.939e-18 |                          6.939e-18 |                   1.804e-16 |                      2.220e-16 |

The audit independently rebuilt every prior-day residual, source/receiver
assignment and fragility product, then recomputed all portfolio returns,
turnover costs and beta constraints. The rejection is confirmed.
PaperLive and remote state are unchanged.

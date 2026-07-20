# v16.0 Independent Audit of v15.9 BD3

Verdict: `rejection_confirmed`.

|   raw_hour_audit_pass |   portfolio_audit_pass |   rejection_confirmed |
|----------------------:|-----------------------:|----------------------:|
|                     1 |                      1 |                     1 |

|   raw_hour_samples | all_hourly_daily_hashes_match   | snapshot_counts_match   |   max_abs_feature_difference |
|-------------------:|:--------------------------------|:------------------------|-----------------------------:|
|                 80 | True                            | True                    |                    0.000e+00 |

|   portfolio_hours |   max_abs_turnover_difference |   max_abs_gross_return_difference |   max_abs_primary_return_difference |   max_abs_stress_return_difference |   max_abs_residual_btc_beta |   max_abs_gross_notional_drift |
|------------------:|------------------------------:|----------------------------------:|------------------------------------:|-----------------------------------:|----------------------------:|-------------------------------:|
|         9.055e+03 |                     4.441e-16 |                         4.337e-18 |                           4.337e-18 |                          4.337e-18 |                   3.053e-16 |                      4.441e-16 |

The audit independently re-read 80 raw one-hour windows using strict
half-open timestamps and recomputed all 9,055 portfolio hours. It
confirms that the negative result is not a timing or cost bug.
PaperLive and remote state are unchanged.

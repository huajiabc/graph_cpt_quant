# v16.6 Independent Audit of v16.5 CM2

Verdict: `forward_shadow_confirmed_with_caveat`.

|   formula_audit_pass |   risk_audit_pass |   main_promotion_reproduced |   forward_shadow_only |   validation_stress_caveat |
|---------------------:|------------------:|----------------------------:|----------------------:|---------------------------:|
|                    1 |                 1 |                           1 |                     1 |                          1 |

|   weeks | calendar_exact   |   max_abs_price_difference |   max_abs_funding_difference |   max_abs_primary_difference |   max_abs_stress_difference |   max_abs_fss3_weight_difference |   max_abs_tg1_weight_difference |
|--------:|:-----------------|---------------------------:|-----------------------------:|-----------------------------:|----------------------------:|---------------------------------:|--------------------------------:|
|      49 | True             |                  0.000e+00 |                    0.000e+00 |                    0.000e+00 |                   0.000e+00 |                        0.000e+00 |                       0.000e+00 |

|   alternate_bootstrap_95_low_bp |   alternate_bootstrap_95_high_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   validation_stress_net_bp |   drawdown_reduction_vs_fss3 |   downside_semideviation_reduction_vs_fss3 |   mean_retention_vs_fss3 |
|--------------------------------:|---------------------------------:|-----------------------------:|----------------------------:|-------------------------:|---------------------------:|-----------------------------:|-------------------------------------------:|-------------------------:|
|                         27.2826 |                         131.3805 |                     138.6848 |                      4.9307 |                  53.9063 |                    -6.3227 |                       0.2535 |                                     0.2241 |                   0.8387 |

The audit independently reloaded both raw sleeve portfolios, aligned
their calendars, recomputed all 49 fixed-weight returns, and repeated
risk tests with 20,000 alternate bootstrap draws. The candidate remains
forward-shadow only because validation stress return is negative and the
20% satellite cap was chosen after sleeve-level evidence review.
PaperLive and remote state are unchanged.

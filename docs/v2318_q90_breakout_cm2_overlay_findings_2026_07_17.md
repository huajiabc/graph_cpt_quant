# v23.18 q90 Breakout + CM2 Overlay Findings

Verdict: `q90_cm2_portfolio_confirmation_rejected`.

| scope       |   weeks |   active_weeks |   events |   mean_core_primary_bp |   mean_satellite_primary_bp |   mean_primary_increment_bp |   mean_stress_increment_bp |   mean_combined_primary_bp |   mean_combined_stress_bp |   core_annualized_sharpe |   combined_annualized_sharpe |   satellite_core_correlation |   active_satellite_core_correlation |   core_downside_semideviation_bp |   combined_downside_semideviation_bp |   core_additive_max_drawdown_bp |   combined_additive_max_drawdown_bp |   core_negative_active_weeks |   downside_active_satellite_mean_bp |   mean_reversed_increment_bp |
|:------------|--------:|---------------:|---------:|-----------------------:|----------------------------:|----------------------------:|---------------------------:|---------------------------:|--------------------------:|-------------------------:|-----------------------------:|-----------------------------:|------------------------------------:|---------------------------------:|-------------------------------------:|--------------------------------:|------------------------------------:|-----------------------------:|------------------------------------:|-----------------------------:|
| all         |      49 |             22 |       53 |                78.9766 |                     21.5743 |                      2.1574 |                     1.0702 |                    81.1340 |                   68.0814 |                   2.4906 |                       2.5843 |                      -0.2247 |                             -0.2995 |                          85.9994 |                              81.8396 |                       -532.2283 |                           -530.2043 |                           12 |                             89.1719 |                      -4.2466 |
| development |      22 |              7 |       20 |               138.6848 |                      9.9509 |                      0.9951 |                     0.0807 |                   139.6799 |                  126.9392 |                   3.5332 |                       3.6003 |                      -0.2748 |                             -0.5216 |                          77.0909 |                              68.9009 |                       -306.4991 |                           -247.6147 |                            4 |                            141.9184 |                      -2.7037 |
| validation  |      13 |              8 |       15 |                 4.9307 |                     44.1430 |                      4.4143 |                     3.2464 |                     9.3450 |                   -3.0763 |                   0.2222 |                       0.4243 |                      -0.1332 |                              0.0481 |                         100.5854 |                              98.4745 |                       -532.2283 |                           -530.2043 |                            5 |                             86.4598 |                      -6.6340 |
| holdout     |      14 |              7 |       18 |                53.9063 |                     18.8832 |                      1.8883 |                     0.6044 |                    55.7946 |                   41.6656 |                   2.3315 |                       2.4093 |                       0.0407 |                              0.1523 |                          84.5688 |                              83.5304 |                       -269.3030 |                           -264.4947 |                            3 |                             23.3635 |                      -4.4545 |

| gate                                               | passed   |   observed |
|:---------------------------------------------------|:---------|-----------:|
| feature_hash_exact                                 | True     |     0.0000 |
| all_53_events_matched_and_triggered                | True     |    53.0000 |
| primary_increment_positive_all_scopes              | True     |     0.9951 |
| stress_increment_positive_all_scopes               | True     |     0.0807 |
| absolute_month_bootstrap_lower_above_zero          | False    |    -0.6930 |
| leave_one_month_out_minimum_above_zero             | True     |     1.1576 |
| satellite_core_correlation_abs_at_most_030         | True     |    -0.2247 |
| active_satellite_core_correlation_abs_at_most_050  | True     |    -0.2995 |
| combined_sharpe_improves_all_scopes                | True     |     0.0671 |
| full_downside_semideviation_not_worse              | True     |     4.1598 |
| full_additive_drawdown_not_worse                   | True     |     2.0240 |
| observed_overlay_beats_sign_reversed               | True     |     6.4041 |
| fixed_sensitivities_positive_all_scopes            | True     |     0.4975 |
| positive_month_increment_concentration_at_most_050 | True     |     0.3654 |

The fixed 10% overlay is temporary notional during four-hour events;
CM2 itself is unchanged. The 5% and 20% rows are frozen scale
sensitivities, not an allocation search. The ancestor remains post-selected.

No live, PaperLive, leverage, remote, application, or order state changed.

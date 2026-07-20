# v23.8 Positive-Pressure Narrow-Breakout Robustness

Verdict: `forward_shadow_candidate_not_statistically_confirmed`.

This is explicitly post-selection robustness, not a new untouched
holdout test. The 0.625-sigma width and positive-pressure filter
were identified after inspecting v23.4--v23.7 outcomes.

| variant            | scope       |   events |   active_months |   triggered_trades |   ambiguous_trades |   mean_primary_net_return_bp |   mean_stress_net_return_bp |
|:-------------------|:------------|---------:|----------------:|-------------------:|-------------------:|-----------------------------:|----------------------------:|
| 0.625sigma_primary | all         |       53 |              11 |                 53 |                  0 |                      19.6905 |                      9.6905 |
| 0.625sigma_primary | development |       20 |               4 |                 20 |                  0 |                      10.4394 |                      0.4394 |
| 0.625sigma_primary | validation  |       15 |               3 |                 15 |                  0 |                      38.0249 |                     28.0249 |
| 0.625sigma_primary | holdout     |       18 |               4 |                 18 |                  0 |                      14.6908 |                      4.6908 |
| 0.75sigma_adjacent | all         |       53 |              11 |                 53 |                  0 |                      17.4530 |                      7.4530 |
| 0.75sigma_adjacent | development |       20 |               4 |                 20 |                  0 |                       4.8343 |                     -5.1657 |
| 0.75sigma_adjacent | validation  |       15 |               3 |                 15 |                  0 |                      44.4331 |                     34.4331 |
| 0.75sigma_adjacent | holdout     |       18 |               4 |                 18 |                  0 |                       8.9904 |                     -1.0096 |

| scope       |   events |   event_mean_bp |   matched_random_percentile |   random_median_bp |
|:------------|---------:|----------------:|----------------------------:|-------------------:|
| all         |       53 |         19.6905 |                     99.4000 |           -13.5190 |
| development |       20 |         10.4394 |                     94.2000 |           -13.1695 |
| validation  |       15 |         38.0249 |                     96.6000 |           -14.3793 |
| holdout     |       18 |         14.6908 |                     91.1000 |           -10.5545 |

| gate                                               | passed   |   observed |
|:---------------------------------------------------|:---------|-----------:|
| primary_positive_all_temporal_scopes               | True     |    10.4394 |
| adjacent_width_positive_all_temporal_scopes        | True     |     4.8343 |
| primary_stress_positive_full_sample                | True     |     9.6905 |
| fifteen_minute_latency_primary_positive_all_scopes | True     |     1.9752 |
| three_and_four_hour_primary_positive_all_scopes    | True     |     3.8772 |
| matched_random_percentile_at_least_90_all_scopes   | True     |    91.1000 |
| within_month_sign_permutation_upper_p_at_most_1pct | True     |     0.0072 |
| month_bootstrap_sign_difference_lower_above_zero   | True     |    25.4703 |
| leave_one_month_out_minimum_above_zero             | True     |    10.4782 |
| absolute_month_bootstrap_lower_above_zero          | False    |    -9.0780 |

Within-month sign permutation upper-tail p: 0.007200.
Month-bootstrap sign-difference 2.5% lower: 25.4703 bp.
Absolute strategy month-bootstrap 2.5% lower: -9.0780 bp.

The candidate is suitable only for new forward shadow observation.
It is not statistically confirmed and has no PaperLive/live permission.

No live, PaperLive, leverage, remote, application, or order state changed.

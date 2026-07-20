# v23.24 q90 Broad-Taker Confirmation Findings

Verdict: `broad_taker_confirmation_rejected`.

| variant   | scope       |   events |   active_months |   mean_gross_return_bp |   mean_primary_net_return_bp |   mean_stress_net_return_bp |   win_rate_primary |
|:----------|:------------|---------:|----------------:|-----------------------:|-----------------------------:|----------------------------:|-------------------:|
| confirmed | all         |       26 |              10 |                -6.0649 |                     -16.0649 |                    -26.0649 |             0.4231 |
| confirmed | development |        8 |               4 |                10.2416 |                       0.2416 |                     -9.7584 |             0.5000 |
| confirmed | validation  |        7 |               2 |               -22.9597 |                     -32.9597 |                    -42.9597 |             0.5714 |
| confirmed | holdout     |       11 |               4 |                -7.1729 |                     -17.1729 |                    -27.1729 |             0.2727 |

| scope       |   events |   matched_events |   unmatched_events |   event_mean_bp |   matched_random_percentile |   random_median_bp |
|:------------|---------:|-----------------:|-------------------:|----------------:|----------------------------:|-------------------:|
| all         |       26 |               24 |                  2 |        -16.7123 |                     41.2000 |           -12.7112 |
| development |        8 |                8 |                  0 |          0.2416 |                     67.9000 |           -12.9755 |
| validation  |        7 |                7 |                  0 |        -32.9597 |                     29.3000 |           -15.6144 |
| holdout     |       11 |                9 |                  2 |        -19.1456 |                     42.4000 |           -11.8448 |

| gate                                             | passed   |   observed |
|:-------------------------------------------------|:---------|-----------:|
| minimum_events_and_period_coverage               | True     |     7.0000 |
| primary_positive_all_scopes                      | False    |   -32.9597 |
| stress_positive_all_scopes                       | False    |   -42.9597 |
| absolute_month_bootstrap_lower_above_zero        | False    |   -36.8225 |
| leave_one_month_out_minimum_above_zero           | False    |   -24.7889 |
| matched_random_percentile_at_least_90_all_scopes | False    |    29.3000 |
| every_event_has_at_least_five_controls           | False    |     2.0000 |
| confirmed_beats_unconfirmed_all_scopes           | False    |   -41.6238 |
| within_month_label_permutation_p_at_most_010     | False    |     0.6812 |
| fifteen_minute_delay_positive_all_scopes         | False    |   -33.5637 |
| long_beats_sign_reversed_short                   | False    |   -12.1298 |
| positive_month_concentration_at_most_050         | False    |     0.8153 |

This is a second-stage attribution test of a post-selected q90
ancestor and cannot independently authorize deployment.

No live, PaperLive, leverage, remote, application, or order state changed.

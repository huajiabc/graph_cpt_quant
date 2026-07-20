# v23.15 Positive-q85 Vacuum Breakout Findings

Verdict: `positive_q85_interpolation_rejected`.

| variant        | scope       |   events |   active_months |   triggered_trades |   ambiguous_trades |   mean_primary_net_return_bp |   mean_stress_net_return_bp |
|:---------------|:------------|---------:|----------------:|-------------------:|-------------------:|-----------------------------:|----------------------------:|
| q85_0.625sigma | all         |       75 |              12 |                 74 |                  0 |                      13.8674 |                      4.0007 |
| q85_0.625sigma | development |       28 |               5 |                 28 |                  0 |                       3.7173 |                     -6.2827 |
| q85_0.625sigma | validation  |       21 |               3 |                 21 |                  0 |                      36.1855 |                     26.1855 |
| q85_0.625sigma | holdout     |       26 |               4 |                 25 |                  0 |                       6.7721 |                     -2.8433 |

| scope       |   events |   matched_events |   unmatched_events |   event_mean_bp |   matched_random_percentile |   random_median_bp |
|:------------|---------:|-----------------:|-------------------:|----------------:|----------------------------:|-------------------:|
| all         |       75 |               75 |                  0 |         13.8674 |                    100.0000 |           -13.2023 |
| development |       28 |               28 |                  0 |          3.7173 |                     96.7000 |           -16.6731 |
| validation  |       21 |               21 |                  0 |         36.1855 |                     98.0000 |           -12.3980 |
| holdout     |       26 |               26 |                  0 |          6.7721 |                     87.5000 |           -10.6820 |

| gate                                             | passed   |   observed |
|:-------------------------------------------------|:---------|-----------:|
| minimum_total_and_each_period_triggers           | True     |     1.0000 |
| primary_positive_all_temporal_scopes             | True     |     3.7173 |
| stress_positive_full_sample                      | True     |     4.0007 |
| absolute_month_bootstrap_lower_above_zero        | False    |   -11.9088 |
| matched_random_percentile_at_least_90_all_scopes | False    |    87.5000 |
| every_event_has_at_least_five_matched_controls   | True     |     0.0000 |
| same_bar_ambiguity_at_most_10pct                 | True     |     0.0000 |
| leave_one_month_out_minimum_above_zero           | True     |     6.0515 |
| adjacent_width_positive_all_temporal_scopes      | False    |    -1.6208 |

q85 was the sole frozen interpolation between q80 and q90.
No further pressure-quantile search is authorized by this round.

No live, PaperLive, leverage, remote, application, or order state changed.

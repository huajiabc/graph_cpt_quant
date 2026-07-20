# v23.12 Positive-q80 Vacuum Breakout Findings

Verdict: `positive_q80_density_extension_rejected`.

| variant        | scope       |   events |   active_months |   triggered_trades |   ambiguous_trades |   mean_primary_net_return_bp |   mean_stress_net_return_bp |
|:---------------|:------------|---------:|----------------:|-------------------:|-------------------:|-----------------------------:|----------------------------:|
| q80_0.625sigma | all         |       89 |              12 |                 88 |                  2 |                       0.1112 |                     -9.7765 |
| q80_0.625sigma | development |       32 |               5 |                 32 |                  1 |                      -9.9469 |                    -19.9469 |
| q80_0.625sigma | validation  |       24 |               3 |                 24 |                  1 |                      20.8534 |                     10.8534 |
| q80_0.625sigma | holdout     |       33 |               4 |                 32 |                  0 |                      -5.2208 |                    -14.9178 |

| scope       |   events |   matched_events |   unmatched_events |   event_mean_bp |   matched_random_percentile |   random_median_bp |
|:------------|---------:|-----------------:|-------------------:|----------------:|----------------------------:|-------------------:|
| all         |       89 |               87 |                  2 |          1.7511 |                     93.3000 |           -10.8867 |
| development |       32 |               32 |                  0 |         -9.9469 |                     66.6000 |           -13.9065 |
| validation  |       24 |               24 |                  0 |         20.8534 |                     95.2000 |           -14.3177 |
| holdout     |       33 |               31 |                  2 |         -0.9624 |                     66.2000 |            -6.3262 |

| gate                                             | passed   |   observed |
|:-------------------------------------------------|:---------|-----------:|
| minimum_total_and_each_period_triggers           | True     |     4.0000 |
| primary_positive_all_temporal_scopes             | False    |    -9.9469 |
| stress_positive_full_sample                      | False    |    -9.7765 |
| absolute_month_bootstrap_lower_above_zero        | False    |   -23.6194 |
| matched_random_percentile_at_least_90_all_scopes | False    |    66.2000 |
| every_event_has_at_least_five_matched_controls   | False    |     2.0000 |
| same_bar_ambiguity_at_most_10pct                 | True     |     0.0227 |
| leave_one_month_out_minimum_above_zero           | False    |    -6.8283 |
| adjacent_width_positive_all_temporal_scopes      | False    |   -13.9037 |

The q80 outcomes were revealed only after the v23.11 feature hash
and v23.12 gates were frozen. This remains a research extension
of a post-selected ancestor, not live authorization.

No live, PaperLive, leverage, remote, application, or order state changed.

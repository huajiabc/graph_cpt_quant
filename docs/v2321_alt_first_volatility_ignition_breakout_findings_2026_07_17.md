# v23.21 Alt-First Volatility Ignition Breakout Findings

Verdict: `alt_first_ignition_breakout_rejected`.

| variant   | scope       |   events |   active_months |   triggered_trades |   ambiguous_trades |   mean_gross_return_bp |   mean_primary_net_return_bp |   mean_stress_net_return_bp |
|:----------|:------------|---------:|----------------:|-------------------:|-------------------:|-----------------------:|-----------------------------:|----------------------------:|
| 0.75sigma | all         |      100 |              12 |                 99 |                  4 |                -7.5319 |                     -17.4319 |                    -27.3319 |
| 0.75sigma | development |       58 |               6 |                 57 |                  3 |               -10.1199 |                     -19.9474 |                    -29.7750 |
| 0.75sigma | validation  |       22 |               3 |                 22 |                  1 |                -1.7626 |                     -11.7626 |                    -21.7626 |
| 0.75sigma | holdout     |       20 |               3 |                 20 |                  0 |                -6.3728 |                     -16.3728 |                    -26.3728 |

| scope       |   events |   matched_events |   unmatched_events |   event_mean_bp |   matched_random_percentile |   random_median_bp |
|:------------|---------:|-----------------:|-------------------:|----------------:|----------------------------:|-------------------:|
| all         |      100 |              100 |                  0 |        -17.4319 |                     13.8000 |           -10.2759 |
| development |       58 |               58 |                  0 |        -19.9474 |                     15.8000 |           -10.9372 |
| validation  |       22 |               22 |                  0 |        -11.7626 |                     24.8000 |             0.2860 |
| holdout     |       20 |               20 |                  0 |        -16.3728 |                     52.1000 |           -17.1960 |

| gate                                             | passed   |   observed |
|:-------------------------------------------------|:---------|-----------:|
| minimum_trigger_count                            | True     |     5.0000 |
| primary_positive_all_scopes                      | False    |   -19.9474 |
| stress_positive_all_scopes                       | False    |   -29.7750 |
| absolute_month_bootstrap_lower_above_zero        | False    |   -38.3404 |
| matched_random_percentile_at_least_90_all_scopes | False    |    13.8000 |
| every_event_has_at_least_four_controls           | True     |     0.0000 |
| same_bar_ambiguity_at_most_10pct                 | True     |     0.0404 |
| leave_one_month_out_minimum_above_zero           | False    |   -24.5244 |
| adjacent_widths_positive_all_scopes              | False    |   -22.9566 |

This result uses the frozen price-only alt-first state and a quiet-BTC
matched control. It does not authorize live or PaperLive execution.

No live, PaperLive, leverage, remote, application, or order state changed.

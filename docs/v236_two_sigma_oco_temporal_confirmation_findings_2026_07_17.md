# v23.6 Two-Sigma OCO Temporal Confirmation Findings

Verdict: `two_sigma_oco_rejected`.

| candidate                       | scope             |   events |   active_months |   triggered_trades |   trade_rate |   long_trades |   short_trades |   ambiguous_trades |   ambiguous_trade_fraction |   mean_gross_return_per_event_bp |   mean_primary_net_return_per_event_bp |   mean_stress_net_return_per_event_bp |   mean_primary_net_return_per_trade_bp |   mean_reversed_primary_net_return_per_event_bp |   median_trigger_delay_minutes |
|:--------------------------------|:------------------|---------:|----------------:|-------------------:|-------------:|--------------:|---------------:|-------------------:|---------------------------:|---------------------------------:|---------------------------------------:|--------------------------------------:|---------------------------------------:|------------------------------------------------:|-------------------------------:|
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | all               |      159 |              11 |                 92 |       0.5786 |            39 |             53 |                  0 |                     0.0000 |                           7.8387 |                                 2.0526 |                               -3.7336 |                                 3.5474 |                                        -13.6249 |                        75.0000 |
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | development       |       63 |               4 |                 33 |       0.5238 |            14 |             19 |                  0 |                     0.0000 |                           6.5425 |                                 1.3044 |                               -3.9337 |                                 2.4902 |                                        -11.7806 |                        75.0000 |
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | validation        |       47 |               3 |                 34 |       0.7234 |            18 |             16 |                  0 |                     0.0000 |                          22.3508 |                                15.1167 |                                7.8827 |                                20.8967 |                                        -29.5848 |                        75.0000 |
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | holdout           |       49 |               4 |                 25 |       0.5102 |             7 |             18 |                  0 |                     0.0000 |                          -4.4144 |                                -9.5165 |                              -14.6185 |                               -18.6523 |                                         -0.6876 |                        75.0000 |
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | positive_pressure |       53 |              11 |                 26 |       0.4906 |            10 |             16 |                  0 |                     0.0000 |                          10.0656 |                                 5.1599 |                                0.2543 |                                10.5183 |                                        -14.9712 |                        82.5000 |
| DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT | negative_pressure |      106 |              11 |                 66 |       0.6226 |            29 |             37 |                  0 |                     0.0000 |                           6.7253 |                                 0.4989 |                               -5.7275 |                                 0.8012 |                                        -12.9517 |                        75.0000 |

| gate                                          | passed   |   observed |
|:----------------------------------------------|:---------|-----------:|
| minimum_20_holdout_triggers                   | True     |    25.0000 |
| holdout_primary_net_positive                  | False    |    -9.5165 |
| full_sample_primary_net_positive              | True     |     2.0526 |
| month_block_bootstrap_lower_above_zero        | False    |    -7.0235 |
| holdout_matched_random_percentile_at_least_90 | False    |    34.7000 |
| full_matched_random_percentile_at_least_90    | True     |    97.8000 |
| ambiguous_trigger_fraction_at_most_10pct      | True     |     0.0000 |
| primary_beats_reversed_direction              | True     |    15.6774 |

Development and validation are selection-period diagnostics; the
2.00-sigma holdout is the new temporal confirmation evidence.
The result remains a bar-based research simulation.

No live, PaperLive, leverage, remote, application, or order state changed.

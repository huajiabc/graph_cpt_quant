# v23.4 Book-Vacuum BTC OCO Breakout Findings

Verdict: `oco_breakout_rejected`.

| candidate                         | scope             |   events |   active_months |   triggered_trades |   trade_rate |   long_trades |   short_trades |   ambiguous_trades |   ambiguous_trade_fraction |   mean_gross_return_per_event_bp |   mean_primary_net_return_per_event_bp |   mean_stress_net_return_per_event_bp |   mean_primary_net_return_per_trade_bp |   mean_reversed_primary_net_return_per_event_bp |   median_trigger_delay_minutes |
|:----------------------------------|:------------------|---------:|----------------:|-------------------:|-------------:|--------------:|---------------:|-------------------:|---------------------------:|---------------------------------:|---------------------------------------:|--------------------------------------:|---------------------------------------:|------------------------------------------------:|-------------------------------:|
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | all               |      159 |              11 |                149 |       0.9371 |            70 |             79 |                  2 |                     0.0134 |                           1.4133 |                                -7.9578 |                              -17.3289 |                                -8.4919 |                                        -12.9613 |                        30.0000 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | development       |       63 |               4 |                 59 |       0.9365 |            29 |             30 |                  1 |                     0.0169 |                           8.1025 |                                -1.2626 |                              -10.6276 |                                -1.3481 |                                        -20.7098 |                        45.0000 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | validation        |       47 |               3 |                 44 |       0.9362 |            22 |             22 |                  1 |                     0.0227 |                           7.2374 |                                -2.1243 |                              -11.4860 |                                -2.2692 |                                        -19.6176 |                        30.0000 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | holdout           |       49 |               4 |                 46 |       0.9388 |            19 |             27 |                  0 |                     0.0000 |                         -12.7736 |                               -22.1613 |                              -31.5491 |                               -23.6066 |                                          3.3858 |                        52.5000 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | positive_pressure |       53 |              11 |                 49 |       0.9245 |            21 |             28 |                  0 |                     0.0000 |                          13.1038 |                                 3.8585 |                               -5.3868 |                                 4.1734 |                                        -22.3490 |                        60.0000 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | negative_pressure |      106 |              11 |                100 |       0.9434 |            49 |             51 |                  2 |                     0.0200 |                          -4.4320 |                               -13.8659 |                              -23.2999 |                               -14.6979 |                                         -8.2674 |                        30.0000 |

Barrier variants:

|   sigma_multiple | scope       |   events |   triggered_trades |   mean_primary_net_return_per_event_bp |
|-----------------:|:------------|---------:|-------------------:|---------------------------------------:|
|           0.7500 | all         |      159 |                157 |                                -9.2651 |
|           0.7500 | development |       63 |                 62 |                                -6.0247 |
|           0.7500 | validation  |       47 |                 46 |                                -6.2300 |
|           0.7500 | holdout     |       49 |                 49 |                               -16.3427 |
|           1.0000 | all         |      159 |                149 |                                -7.9578 |
|           1.0000 | development |       63 |                 59 |                                -1.2626 |
|           1.0000 | validation  |       47 |                 44 |                                -2.1243 |
|           1.0000 | holdout     |       49 |                 46 |                               -22.1613 |
|           1.2500 | all         |      159 |                139 |                                -0.8924 |
|           1.2500 | development |       63 |                 58 |                                -7.2766 |
|           1.2500 | validation  |       47 |                 41 |                                10.7879 |
|           1.2500 | holdout     |       49 |                 40 |                                -3.8875 |

Matched random-time percentile: 71.40.
Month-block bootstrap 2.5% lower bound: -21.1829 bp.

The result uses pessimistic same-bar ambiguity and gap fills.
It remains a historical 15-minute bar simulation, not a guarantee
of stop execution or queue position.

No live, PaperLive, leverage, remote, application, or order state changed.

# v21.6 DEX Community Relative-Spread Findings

Verdict: `rejected`.

| candidate                                  |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   random_control_percentile |   day_bootstrap_lower_95_primary_net_bp | economically_interesting   | promotion_eligible   | status   |
|:-------------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------------:|----------------------------------------:|:---------------------------|:---------------------|:---------|
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD |      267 |          9.8349 |              -10.1651 |             -30.1651 |                      0.9280 |                                -21.6665 | False                      | False                | rejected |

Chronological results:

| candidate                                  | scope       |   events |   active_days |   active_months |   source_symbols |   mean_leg_size |   mean_laggard_bp |   mean_leader_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |
|:-------------------------------------------|:------------|---------:|--------------:|----------------:|-----------------:|----------------:|------------------:|-----------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | all         |      267 |           145 |               8 |               15 |          2.7378 |            9.4288 |          -1.4562 |              1.8623 |          9.8349 |              -10.1651 |             -30.1651 |                       -29.8349 |                      0.3783 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | development |      203 |            95 |               4 |               12 |          2.6847 |            9.2083 |          -1.2148 |              2.0922 |         10.0858 |               -9.9142 |             -29.9142 |                       -30.0858 |                      0.3793 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | validation  |       31 |            28 |               2 |                3 |          2.8065 |           -6.5562 |          18.0332 |              3.4007 |         14.8778 |               -5.1222 |             -25.1222 |                       -34.8778 |                      0.4516 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | holdout     |       33 |            22 |               2 |                7 |          3.0000 |           25.8014 |         -21.2490 |             -0.9978 |          3.5547 |              -16.4453 |             -36.4453 |                       -23.5547 |                      0.3030 |

Timing controls:

|   delayed_gross_bp |   delayed_net20_bp |   placebo_24h_gross_bp |   placebo_24h_net20_bp |
|-------------------:|-------------------:|-----------------------:|-----------------------:|
|             8.5097 |           -11.4903 |               -10.1875 |               -30.1875 |

Alternate horizons:

| candidate                                  | scope       |   events |   active_days |   active_months |   source_symbols |   mean_leg_size |   mean_laggard_bp |   mean_leader_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |   holding_bars |
|:-------------------------------------------|:------------|---------:|--------------:|----------------:|-----------------:|----------------:|------------------:|-----------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|---------------:|
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | all         |      269 |           146 |               8 |               15 |          2.7398 |            0.1746 |           1.1850 |             -0.1289 |          1.2307 |              -18.7693 |             -38.7693 |                       -21.2307 |                      0.3011 |             16 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | development |      203 |            95 |               4 |               12 |          2.6847 |           -3.7086 |           5.0888 |             -0.3924 |          0.9878 |              -19.0122 |             -39.0122 |                       -20.9878 |                      0.3103 |             16 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | validation  |       31 |            28 |               2 |                3 |          2.8065 |            6.7375 |          -8.9841 |              1.4394 |         -0.8072 |              -20.8072 |             -40.8072 |                       -19.1928 |                      0.1613 |             16 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | holdout     |       35 |            23 |               2 |                7 |          3.0000 |           16.8842 |         -12.4502 |              0.0101 |          4.4442 |              -15.5558 |             -35.5558 |                       -24.4442 |                      0.3714 |             16 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | all         |      267 |           145 |               8 |               15 |          2.7378 |           -7.4010 |           8.9092 |             -0.6535 |          0.8547 |              -19.1453 |             -39.1453 |                       -20.8547 |                      0.3708 |             96 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | development |      203 |            95 |               4 |               12 |          2.6847 |           -7.1232 |           8.5707 |             -0.3807 |          1.0668 |              -18.9332 |             -38.9332 |                       -21.0668 |                      0.3793 |             96 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | validation  |       31 |            28 |               2 |                3 |          2.8065 |          -50.6295 |          47.2649 |             -0.8226 |         -4.1872 |              -24.1872 |             -44.1872 |                       -15.8128 |                      0.3548 |             96 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | holdout     |       33 |            22 |               2 |                7 |          3.0000 |           31.4989 |         -25.0398 |             -2.1724 |          4.2867 |              -15.7133 |             -35.7133 |                       -24.2867 |                      0.3333 |             96 |

Concentration:

| candidate                                  |   maximum_month_positive_pnl_share |   maximum_source_positive_pnl_share |
|:-------------------------------------------|-----------------------------------:|------------------------------------:|
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD |                             0.9571 |                              0.5010 |

This is a second-stage same-history diagnostic. Even a passing economic result is not promotion evidence and requires genuinely new natural-forward data.

No live, PaperLive, application, leverage, remote, or order state changed.

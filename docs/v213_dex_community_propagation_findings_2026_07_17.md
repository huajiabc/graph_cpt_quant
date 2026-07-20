# v21.3 DEX Community-Propagation Findings

Verdict: `reject_dex_community_propagation_candidates`.

| candidate                                |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   random_control_percentile |   day_bootstrap_lower_95_primary_net_bp | eligible   | status   |
|:-----------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------------:|----------------------------------------:|:-----------|:---------|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION |      267 |         -1.8487 |              -21.8487 |             -41.8487 |                      0.6620 |                                -31.4170 | False      | rejected |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    |      267 |          1.4058 |              -18.5942 |             -38.5942 |                      0.9080 |                                -28.8206 | False      | rejected |

Chronological primary results:

| candidate                                | scope       |   events |   active_days |   active_months |   source_symbols |   mean_selection_count |   mean_alt_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |
|:-----------------------------------------|:------------|---------:|--------------:|----------------:|-----------------:|-----------------------:|--------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | all         |      267 |           145 |               8 |               15 |                 5.9513 |        4.4007 |             -6.2494 |         -1.8487 |              -21.8487 |             -41.8487 |                       -18.1513 |                      0.3820 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | development |      203 |            95 |               4 |               12 |                 5.8424 |        3.1802 |             -8.3566 |         -5.1764 |              -25.1764 |             -45.1764 |                       -14.8236 |                      0.3695 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | validation  |       31 |            28 |               2 |                3 |                 6.0645 |       -9.0072 |             10.8619 |          1.8547 |              -18.1453 |             -38.1453 |                       -21.8547 |                      0.4516 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | holdout     |       33 |            22 |               2 |                7 |                 6.5152 |       24.5035 |             -9.3610 |         15.1425 |               -4.8575 |             -24.8575 |                       -35.1425 |                      0.3939 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | all         |      267 |           145 |               8 |               15 |                 3.2921 |        7.5650 |             -6.1593 |          1.4058 |              -18.5942 |             -38.5942 |                       -21.4058 |                      0.3708 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | development |      203 |            95 |               4 |               12 |                 3.2611 |        5.6831 |             -8.2954 |         -2.6123 |              -22.6123 |             -42.6123 |                       -17.3877 |                      0.3498 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | validation  |       31 |            28 |               2 |                3 |                 3.2581 |       -1.9234 |             11.4488 |          9.5254 |              -10.4746 |             -30.4746 |                       -29.5254 |                      0.4516 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | holdout     |       33 |            22 |               2 |                7 |                 3.5152 |       28.0553 |             -9.5599 |         18.4954 |               -1.5046 |             -21.5046 |                       -38.4954 |                      0.4242 |

Timing and attribution controls:

| candidate                                |   delayed_gross_bp |   delayed_net20_bp |   placebo_24h_gross_bp |   placebo_24h_net20_bp |   source_only_gross_bp |   source_only_net20_bp |
|:-----------------------------------------|-------------------:|-------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION |             0.1691 |           -19.8309 |                -3.8130 |               -23.8130 |               -22.1494 |               -42.1494 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    |             3.0829 |           -16.9171 |                -5.7553 |               -25.7553 |               -22.1494 |               -42.1494 |

Alternate horizons:

| candidate                                | scope       |   events |   active_days |   active_months |   source_symbols |   mean_selection_count |   mean_alt_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |   holding_bars |
|:-----------------------------------------|:------------|---------:|--------------:|----------------:|-----------------:|-----------------------:|--------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|---------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | all         |      269 |           146 |               8 |               15 |                 5.9554 |       -0.3809 |             -4.5621 |         -4.9430 |              -24.9430 |             -44.9430 |                       -15.0570 |                      0.2788 |             16 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | development |      203 |            95 |               4 |               12 |                 5.8424 |       -4.5551 |             -4.1686 |         -8.7237 |              -28.7237 |             -48.7237 |                       -11.2763 |                      0.2759 |             16 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | validation  |       31 |            28 |               2 |                3 |                 6.0645 |        9.8941 |             -5.0778 |          4.8163 |              -15.1837 |             -35.1837 |                       -24.8163 |                      0.3548 |             16 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | holdout     |       35 |            23 |               2 |                7 |                 6.5143 |       14.7288 |             -6.3880 |          8.3408 |              -11.6592 |             -31.6592 |                       -28.3408 |                      0.2286 |             16 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | all         |      269 |           146 |               8 |               15 |                 3.2937 |        0.0158 |             -4.6554 |         -4.6396 |              -24.6396 |             -44.6396 |                       -15.3604 |                      0.2602 |             16 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | development |      203 |            95 |               4 |               12 |                 3.2611 |       -4.6453 |             -4.3327 |         -8.9780 |              -28.9780 |             -48.9780 |                       -11.0220 |                      0.2365 |             16 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | validation  |       31 |            28 |               2 |                3 |                 3.2581 |       10.4742 |             -4.7821 |          5.6921 |              -14.3079 |             -34.3079 |                       -25.6921 |                      0.3548 |             16 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | holdout     |       35 |            23 |               2 |                7 |                 3.5143 |       17.7871 |             -6.4151 |         11.3720 |               -8.6280 |             -28.6280 |                       -31.3720 |                      0.3143 |             16 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | all         |      267 |           145 |               8 |               15 |                 5.9513 |       -3.8670 |              5.0707 |          1.2037 |              -18.7963 |             -38.7963 |                       -21.2037 |                      0.4007 |             96 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | development |      203 |            95 |               4 |               12 |                 5.8424 |       -3.1711 |              1.4714 |         -1.6997 |              -21.6997 |             -41.6997 |                       -18.3003 |                      0.4089 |             96 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | validation  |       31 |            28 |               2 |                3 |                 6.0645 |      -42.1400 |             43.2110 |          1.0710 |              -18.9290 |             -38.9290 |                       -21.0710 |                      0.3226 |             96 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | holdout     |       33 |            22 |               2 |                7 |                 6.5152 |       27.8056 |             -8.6174 |         19.1882 |               -0.8118 |             -20.8118 |                       -39.1882 |                      0.4242 |             96 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | all         |      267 |           145 |               8 |               15 |                 3.2921 |       -2.8988 |              4.8755 |          1.9767 |              -18.0233 |             -38.0233 |                       -21.9767 |                      0.3933 |             96 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | development |      203 |            95 |               4 |               12 |                 3.2611 |       -2.5003 |              1.3114 |         -1.1890 |              -21.1890 |             -41.1890 |                       -18.8110 |                      0.4039 |             96 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | validation  |       31 |            28 |               2 |                3 |                 3.2581 |      -41.8789 |             42.8897 |          1.0108 |              -18.9892 |             -38.9892 |                       -21.0108 |                      0.3226 |             96 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | holdout     |       33 |            22 |               2 |                7 |                 3.5152 |       31.2677 |             -8.9100 |         22.3577 |                2.3577 |             -17.6423 |                       -42.3577 |                      0.3939 |             96 |

Concentration:

| candidate                                |   maximum_month_positive_pnl_share |   maximum_source_positive_pnl_share |
|:-----------------------------------------|-----------------------------------:|------------------------------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION |                             1.0000 |                              0.4212 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    |                             1.0000 |                              0.3989 |

The reveal follows the frozen v21.3 preregistration. The four-event DEX-vendor transition interval is excluded from eligibility statistics. Primary/stress book costs are 20/40 bp round trip on unit gross.

No live, PaperLive, application, leverage, remote, or order state changed.

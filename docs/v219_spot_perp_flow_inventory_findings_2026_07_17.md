# v21.9 Spot-Perpetual Flow-Inventory Findings

Verdict: `reject_spot_perp_flow_inventory_candidates`.

| candidate                                |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   random_control_percentile |   day_bootstrap_lower_95_primary_net_bp | eligible   | status   |
|:-----------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------------:|----------------------------------------:|:-----------|:---------|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    |      357 |          4.3989 |              -15.6011 |             -35.6011 |                      0.8820 |                                -21.7662 | False      | rejected |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE |      393 |          4.9984 |              -15.0016 |             -35.0016 |                      0.9560 |                                -21.1940 | False      | rejected |

Chronological results:

| candidate                                | scope       |   events |   active_days |   active_months |   mean_long_count |   mean_short_count |   mean_long_bp |   mean_short_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |
|:-----------------------------------------|:------------|---------:|--------------:|----------------:|------------------:|-------------------:|---------------:|----------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | all         |      357 |           237 |              10 |            7.3697 |             7.2885 |        -3.4460 |          6.5350 |              1.3100 |          4.3989 |              -15.6011 |             -35.6011 |                       -24.3989 |                      0.3305 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | development |      154 |           103 |               4 |            7.2662 |             7.2857 |       -11.4424 |         12.3906 |              0.0104 |          0.9586 |              -19.0414 |             -39.0414 |                       -20.9586 |                      0.2532 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | validation  |       69 |            47 |               2 |            7.3478 |             7.0870 |       -14.7578 |         16.0333 |              2.9439 |          4.2194 |              -15.7806 |             -35.7806 |                       -24.2194 |                      0.3768 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | holdout     |      134 |            87 |               4 |            7.5000 |             7.3955 |        11.5687 |         -5.0857 |              1.9622 |          8.4452 |              -11.5548 |             -31.5548 |                       -28.4452 |                      0.3955 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | all         |      393 |           240 |              10 |            5.7226 |             5.7226 |        -0.9014 |          5.7755 |              0.1243 |          4.9984 |              -15.0016 |             -35.0016 |                       -24.9984 |                      0.3639 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | development |      166 |           103 |               4 |            5.8434 |             5.8434 |       -11.3880 |         12.4647 |             -0.7247 |          0.3520 |              -19.6480 |             -39.6480 |                       -20.3520 |                      0.3193 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | validation  |       81 |            49 |               2 |            5.6543 |             5.6543 |         0.3505 |          7.9011 |             -1.4909 |          6.7607 |              -13.2393 |             -33.2393 |                       -26.7607 |                      0.4444 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | holdout     |      146 |            88 |               4 |            5.6233 |             5.6233 |        10.3273 |         -3.0093 |              1.9856 |          9.3036 |              -10.6964 |             -30.6964 |                       -29.3036 |                      0.3699 |

Timing controls:

| candidate                                |   delayed_gross_bp |   delayed_net20_bp |   placebo_24h_gross_bp |   placebo_24h_net20_bp |
|:-----------------------------------------|-------------------:|-------------------:|-----------------------:|-----------------------:|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    |             5.1658 |           -14.8342 |                -0.4156 |               -20.4156 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE |             5.9227 |           -14.0773 |                -2.3877 |               -22.3877 |

Alternate horizons:

| candidate                                | scope       |   events |   active_days |   active_months |   mean_long_count |   mean_short_count |   mean_long_bp |   mean_short_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |   holding_hours |
|:-----------------------------------------|:------------|---------:|--------------:|----------------:|------------------:|-------------------:|---------------:|----------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|----------------:|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | all         |      358 |           238 |              10 |            7.3715 |             7.2905 |        -0.7501 |         -0.0963 |              1.0966 |          0.2501 |              -19.7499 |             -39.7499 |                       -20.2501 |                      0.2318 |               4 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | development |      154 |           103 |               4 |            7.2662 |             7.2857 |        -1.7805 |         -1.2323 |             -0.1276 |         -3.1404 |              -23.1404 |             -43.1404 |                       -16.8596 |                      0.1753 |               4 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | validation  |       69 |            47 |               2 |            7.3478 |             7.0870 |       -12.8843 |          8.8252 |              2.3873 |         -1.6718 |              -21.6718 |             -41.6718 |                       -18.3282 |                      0.2319 |               4 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | holdout     |      135 |            88 |               4 |            7.5037 |             7.4000 |         6.6271 |         -3.3603 |              1.8333 |          5.1002 |              -14.8998 |             -34.8998 |                       -25.1002 |                      0.2963 |               4 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | all         |      394 |           240 |              10 |            5.7234 |             5.7234 |         1.8696 |         -2.6101 |              0.7087 |         -0.0319 |              -20.0319 |             -40.0319 |                       -19.9681 |                      0.2411 |               4 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | development |      166 |           103 |               4 |            5.8434 |             5.8434 |        -0.2162 |         -4.5265 |              0.0175 |         -4.7252 |              -24.7252 |             -44.7252 |                       -15.2748 |                      0.1747 |               4 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | validation  |       81 |            49 |               2 |            5.6543 |             5.6543 |         1.1945 |         -3.5901 |              1.1390 |         -1.2565 |              -21.2565 |             -41.2565 |                       -18.7435 |                      0.3333 |               4 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | holdout     |      147 |            88 |               4 |            5.6259 |             5.6259 |         4.5970 |          0.0940 |              1.2520 |          5.9430 |              -14.0570 |             -34.0570 |                       -25.9430 |                      0.2653 |               4 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | all         |      357 |           237 |              10 |            7.3697 |             7.2885 |        -5.4365 |          9.6814 |              1.5143 |          5.7593 |              -14.2407 |             -34.2407 |                       -25.7593 |                      0.3810 |              24 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | development |      154 |           103 |               4 |            7.2662 |             7.2857 |       -19.0762 |         17.3221 |             -0.7733 |         -2.5274 |              -22.5274 |             -42.5274 |                       -17.4726 |                      0.3247 |              24 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | validation  |       69 |            47 |               2 |            7.3478 |             7.0870 |       -13.1435 |         25.9800 |              5.1690 |         18.0056 |               -1.9944 |             -21.9944 |                       -38.0056 |                      0.4638 |              24 |
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | holdout     |      134 |            87 |               4 |            7.5000 |             7.3955 |        14.2074 |         -7.4922 |              2.2616 |          8.9768 |              -11.0232 |             -31.0232 |                       -28.9768 |                      0.4030 |              24 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | all         |      392 |           239 |              10 |            5.7245 |             5.7245 |        -2.1003 |          7.5975 |              0.8137 |          6.3110 |              -13.6890 |             -33.6890 |                       -26.3110 |                      0.4005 |              24 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | development |      166 |           103 |               4 |            5.8434 |             5.8434 |       -15.1139 |         16.2778 |             -0.7174 |          0.4465 |              -19.5535 |             -39.5535 |                       -20.4465 |                      0.3494 |              24 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | validation  |       81 |            49 |               2 |            5.6543 |             5.6543 |        -7.1691 |         19.7193 |              1.8979 |         14.4481 |               -5.5519 |             -25.5519 |                       -34.4481 |                      0.5062 |              24 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | holdout     |      145 |            87 |               4 |            5.6276 |             5.6276 |        15.6296 |         -9.1114 |              1.9610 |          8.4793 |              -11.5207 |             -31.5207 |                       -28.4793 |                      0.4000 |              24 |

Concentration:

| candidate                                |   maximum_month_positive_pnl_share |   maximum_symbol_selection_share |
|:-----------------------------------------|-----------------------------------:|---------------------------------:|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    |                             1.0000 |                           0.3137 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE |                             1.0000 |                           0.2545 |

The reveal follows the frozen v21.9 preregistration. Spot is information only; all realized returns are Binance USD-M perpetual returns. Primary and stress costs are 20/40 bp round trip on unit gross.

No live, PaperLive, application, leverage, remote, or order state changed.

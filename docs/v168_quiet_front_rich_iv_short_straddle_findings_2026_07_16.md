# v16.8 Quiet Alt Front + Rich IV Short-Straddle Findings

Verdict: `reject_rich_iv_quiet_front_short_straddle`.

| candidate                               |   eligible_daily_straddles |   rich_iv_days |   candidate_trades |   candidate_primary_net_bp |   candidate_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   circular_percentile |   iv_only_primary_net_bp |   delayed_primary_net_bp |   identical_long_primary_net_bp |   worst_trade_bp |   expected_shortfall_5_bp |   positive_month_concentration |   positive_trade_concentration | promote_research_followup   | failed_gates                                                                                                                                                                  |
|:----------------------------------------|---------------------------:|---------------:|-------------------:|---------------------------:|--------------------------:|----------------------:|-----------------------:|----------------------:|-------------------------:|-------------------------:|--------------------------------:|-----------------:|--------------------------:|-------------------------------:|-------------------------------:|:----------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OVS1_RICH_IV_QUIET_FRONT_SHORT_STRADDLE |                        118 |             62 |                 30 |                   -46.7661 |                  -59.4780 |              -79.4462 |               -18.0531 |                0.6080 |                 -49.4333 |                 -47.5827 |                        -28.0340 |        -323.2466 |                 -275.6197 |                         0.5970 |                         0.2387 | False                       | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|circular_percentile_95|month_concentration_50 |

| scope       |   trades |   active_months |   mean_short_gross_bp |   mean_short_primary_net_bp |   mean_short_stress_net_bp |   mean_identical_long_primary_net_bp |   win_rate |   mean_iv_rv_spread_vol_points |
|:------------|---------:|----------------:|----------------------:|----------------------------:|---------------------------:|-------------------------------------:|-----------:|-------------------------------:|
| all         |       30 |               6 |              -34.0542 |                    -46.7661 |                   -59.4780 |                             -28.0340 |     0.3000 |                        18.0617 |
| development |       19 |               3 |              -25.3381 |                    -38.1088 |                   -50.8796 |                             -45.6731 |     0.3158 |                        18.0619 |
| validation  |        5 |               2 |              -47.8854 |                    -60.3945 |                   -72.9037 |                              -2.2122 |     0.2000 |                        17.6665 |
| holdout     |        6 |               2 |              -50.1292 |                    -62.8236 |                   -75.5181 |                               6.3049 |     0.3333 |                        18.3903 |

Short entries use archived bids and exits use archived asks; the result is not
the negative of v16.7 long PnL. Static BTC delta hedging and option/hedge fees
are included. The study is adaptive and the 2023 archive is short, so no
PaperLive, leverage, remote-change, or real-order permission is granted.

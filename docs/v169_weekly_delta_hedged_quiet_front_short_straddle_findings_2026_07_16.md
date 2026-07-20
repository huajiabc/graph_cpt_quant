# v16.9 Weekly Delta-Hedged Quiet-Front Short-Straddle Findings

Verdict: `reject_weekly_quiet_front_short_straddle`.

| candidate                                      |   eligible_weekly_paths |   candidate_trades |   candidate_primary_net_bp |   candidate_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   circular_percentile |   iv_only_primary_net_bp |   delayed_primary_net_bp |   identical_long_primary_net_bp |   worst_trade_bp |   expected_shortfall_5_bp |   positive_month_concentration |   positive_trade_concentration | promote_research_followup   | failed_gates                                                                                                                                                                  |
|:-----------------------------------------------|------------------------:|-------------------:|---------------------------:|--------------------------:|----------------------:|-----------------------:|----------------------:|-------------------------:|-------------------------:|--------------------------------:|-----------------:|--------------------------:|-------------------------------:|-------------------------------:|:----------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OVS2_WEEKLY_RICH_IV_QUIET_FRONT_SHORT_STRADDLE |                     101 |                 11 |                   -79.9084 |                  -95.5998 |             -232.5827 |                56.7853 |                0.1300 |                 -26.1182 |                   6.0027 |                        -47.9890 |        -593.1536 |                 -593.1536 |                         0.5045 |                         0.3202 | False                       | development_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|circular_percentile_95|beats_iv_only|beats_delayed|month_concentration_50 |

| scope       |   trades |   mean_short_gross_bp |   mean_short_primary_net_bp |   mean_short_stress_net_bp |   mean_identical_long_primary_net_bp |   win_rate |   mean_hedge_turnover_x_btc |
|:------------|---------:|----------------------:|----------------------------:|---------------------------:|-------------------------------------:|-----------:|----------------------------:|
| all         |       11 |              -64.2170 |                    -79.9084 |                   -95.5998 |                             -47.9890 |     0.5455 |                      0.8991 |
| development |        7 |              -31.6544 |                    -47.1178 |                   -62.5813 |                             -71.4845 |     0.7143 |                      0.8584 |
| validation  |        2 |               44.5196 |                     29.6243 |                    14.7290 |                             -94.7787 |     0.5000 |                      0.7397 |
| holdout     |        2 |             -286.9228 |                   -304.2083 |                  -321.4938 |                              81.0349 |     0.0000 |                      1.2007 |

All eight option snapshots are exact archived quotes. Entry sells at bids,
day-seven exit buys at asks, and every daily BTC delta adjustment pays turnover
cost. This adaptive short 2023 study changes no PaperLive, leverage, remote,
or real-order permission.

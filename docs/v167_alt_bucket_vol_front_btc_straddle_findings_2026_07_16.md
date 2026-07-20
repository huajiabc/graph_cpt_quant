# v16.7 Alt-Bucket Volatility Front -> BTC Straddle Findings

Verdict: `reject_frozen_alt_front_straddle`.

| candidate                            |   eligible_daily_straddles |   candidate_trades |   candidate_primary_net_bp |   candidate_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   circular_percentile |   delayed_primary_net_bp |   btc_compression_primary_net_bp |   positive_month_concentration |   positive_trade_concentration | promote_research_followup   | failed_gates                                                                                                                                                                                                                                        |
|:-------------------------------------|---------------------------:|-------------------:|---------------------------:|--------------------------:|----------------------:|-----------------------:|----------------------:|-------------------------:|---------------------------------:|-------------------------------:|-------------------------------:|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OVT1_ALT_VOL_FRONT_LONG_BTC_STRADDLE |                        118 |                  1 |                   -20.5620 |                  -33.8098 |              -20.5620 |               -20.5620 |                0.7635 |                -106.6986 |                         -48.9943 |                            inf |                            inf | False                       | trades_20|validation_trades_5|holdout_trades_5|development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|circular_percentile_95|month_concentration_50|trade_concentration_35 |

| scope       |   trades |   active_days |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_premium_return_pct |   win_rate |
|:------------|---------:|--------------:|----------------:|----------------------:|---------------------:|--------------------------:|-----------:|
| all         |        1 |             1 |         -7.3143 |              -20.5620 |             -33.8098 |                   -3.0588 |     0.0000 |
| development |        0 |             0 |        nan      |              nan      |             nan      |                  nan      |   nan      |
| validation  |        0 |             0 |        nan      |              nan      |             nan      |                  nan      |   nan      |
| holdout     |        1 |             1 |         -7.3143 |              -20.5620 |             -33.8098 |                   -3.0588 |     0.0000 |

Every option entry uses the archived ask and every exit uses the archived bid.
The static BTC delta hedge and frozen primary/stress fee schedules are included.
Missing option days were not filled. This short 2023 archive cannot authorize
PaperLive, leverage, remote changes, or real orders.

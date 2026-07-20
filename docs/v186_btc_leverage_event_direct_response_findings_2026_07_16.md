# v18.6 BTC Leverage-Event Direct Response Findings

Verdict: `reject_btc_leverage_event_direct_response`.

| candidate                   |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   delayed_primary_net_bp |   reversed_primary_net_bp |   positive_profit_concentration | eligible   | failed_gates                                                                                                                                                                                                                                       | verdict                                   |
|:----------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------:|--------------------------:|--------------------------------:|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|
| LDR1_BTC_BUILD_CONTINUATION |      199 |          2.6311 |               -7.3689 |             -12.3689 |              -13.0086 |                -1.5700 |                     0.7880 |                  -8.8589 |                  -12.6311 |                          0.8923 | False      | development_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|return_q85_positive|return_q95_positive|holding_15m_positive|holding_60m_positive|positive_profit_concentration_35 | reject_btc_leverage_event_direct_response |
| LDR2_BTC_UNWIND_REVERSAL    |      256 |          5.3676 |               -4.6324 |              -9.6324 |               -9.7403 |                 0.0943 |                     0.9780 |                  -4.6852 |                  -15.3676 |                          0.5856 | False      | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|return_q85_positive|return_q95_positive|holding_15m_positive|holding_60m_positive|positive_profit_concentration_35 | reject_btc_leverage_event_direct_response |

| candidate                   | scope       |   events |   active_days |   active_months |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |
|:----------------------------|:------------|---------:|--------------:|----------------:|----------------:|----------------------:|---------------------:|-------------------:|
| LDR1_BTC_BUILD_CONTINUATION | all         |      199 |           141 |              12 |          2.6311 |               -7.3689 |             -12.3689 |             0.3266 |
| LDR1_BTC_BUILD_CONTINUATION | development |      118 |            80 |               6 |          0.1253 |               -9.8747 |             -14.8747 |             0.3305 |
| LDR1_BTC_BUILD_CONTINUATION | validation  |       32 |            25 |               2 |         13.2669 |                3.2669 |              -1.7331 |             0.3125 |
| LDR1_BTC_BUILD_CONTINUATION | holdout     |       49 |            36 |               4 |          1.7196 |               -8.2804 |             -13.2804 |             0.3265 |
| LDR2_BTC_UNWIND_REVERSAL    | all         |      256 |           166 |              12 |          5.3676 |               -4.6324 |              -9.6324 |             0.4805 |
| LDR2_BTC_UNWIND_REVERSAL    | development |      154 |            95 |               6 |          8.5834 |               -1.4166 |              -6.4166 |             0.5000 |
| LDR2_BTC_UNWIND_REVERSAL    | validation  |       37 |            28 |               2 |         -6.0281 |              -16.0281 |             -21.0281 |             0.4595 |
| LDR2_BTC_UNWIND_REVERSAL    | holdout     |       65 |            43 |               4 |          4.2354 |               -5.7646 |             -10.7646 |             0.4462 |

The v18.5 source construction was reused without graph membership.
No live, PaperLive, application, leverage, remote, or order scope changed.

# v19.2 Premium-Innovation Unwind Reversal Findings

Verdict: `reject_premium_innovation_unwind_reversal`.

| candidate                             |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   reversed_primary_net_bp |   delayed_primary_net_bp |   ranking_control_primary_net_bp |   positive_profit_concentration | eligible   | failed_gates                                                                                                                                                                                                                                                                                                                                                                  | verdict                                   |
|:--------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------:|-----------------------:|---------------------------:|--------------------------:|-------------------------:|---------------------------------:|--------------------------------:|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       |      103 |          7.3275 |               -2.6725 |              -7.6725 |               -9.5970 |                 4.1919 |                     0.7900 |                  -17.3275 |                  -3.8756 |                          -6.3994 |                          0.5821 | False      | validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|source_q90_positive|broad_premium_positive|holding_15m_positive|holding_60m_positive|short_cover_positive|positive_profit_concentration_35                                                                                                     | reject_premium_innovation_unwind_reversal |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET |       81 |         -1.8931 |              -31.8931 |             -41.8931 |              -35.2911 |               -28.4797 |                     0.0000 |                  -28.1069 |                 -29.2066 |                         -32.0219 |                        inf      | False      | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|beats_reversed_direction|beats_one_bar_delay|source_q90_positive|broad_premium_positive|holding_15m_positive|holding_60m_positive|long_liquidation_positive|short_cover_positive|positive_profit_concentration_35 | reject_premium_innovation_unwind_reversal |

| candidate                             | scope       |   events |   active_days |   active_months |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |
|:--------------------------------------|:------------|---------:|--------------:|----------------:|----------------:|----------------------:|---------------------:|-------------------:|
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | all         |      103 |            82 |              12 |          7.3275 |               -2.6725 |              -7.6725 |             0.4660 |
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | development |       68 |            53 |               6 |         12.7281 |                2.7281 |              -2.2719 |             0.5000 |
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | validation  |       10 |             9 |               2 |         -7.7823 |              -17.7823 |             -22.7823 |             0.4000 |
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | holdout     |       25 |            20 |               4 |         -1.3182 |              -11.3182 |             -16.3182 |             0.4000 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | all         |       81 |            69 |              11 |         -1.8931 |              -31.8931 |             -41.8931 |             0.0123 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | development |       50 |            42 |               5 |         -1.4069 |              -31.4069 |             -41.4069 |             0.0200 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | validation  |        8 |             7 |               2 |         -1.2311 |              -31.2311 |             -41.2311 |             0.0000 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | holdout     |       23 |            20 |               4 |         -3.1805 |              -33.1805 |             -43.1805 |             0.0000 |

| candidate                             | source_side      |   events |   mean_gross_bp |   mean_primary_net_bp |
|:--------------------------------------|:-----------------|---------:|----------------:|----------------------:|
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | long_liquidation |       59 |         10.1196 |                0.1196 |
| PIR1_BTC_PREMIUM_SHOCK_REVERSAL       | short_cover      |       44 |          3.5834 |               -6.4166 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | long_liquidation |       48 |         -2.8221 |              -32.8221 |
| PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET | short_cover      |       33 |         -0.5419 |              -30.5419 |

Premium values are exact completed closes with no forward filling.
No live, PaperLive, application, leverage, remote, or order scope changed.

# v19.4 Opposing-Premium Absorption Continuation Findings

Verdict: `reject_opposing_premium_absorption_continuation`.

| candidate                                          |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   reversed_primary_net_bp |   delayed_primary_net_bp |   control_primary_net_bp |   positive_profit_concentration | eligible   | failed_gates                                                                                                                                                                                                                                                                                                                               | verdict                                         |
|:---------------------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------:|-----------------------:|---------------------------:|--------------------------:|-------------------------:|-------------------------:|--------------------------------:|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------|
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  |      178 |         -4.5872 |              -14.5872 |             -19.5872 |              -22.5163 |                -6.5592 |                     0.2020 |                   -5.4128 |                 -16.6317 |                 -19.2888 |                          0.6836 | False      | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|beats_reversed_direction|source_q90_positive|range_z15_positive|holding_15m_positive|holding_60m_positive|up_move_positive|down_move_positive|positive_profit_concentration_35 | reject_opposing_premium_absorption_continuation |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET |       69 |          7.6302 |              -22.3698 |             -32.3698 |              -30.6128 |               -11.5311 |                     0.0000 |                  -37.6302 |                 -20.4207 |                 -26.1441 |                          1.0000 | False      | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|beats_one_bar_delay|source_q90_positive|range_z15_positive|holding_15m_positive|holding_60m_positive|up_move_positive|down_move_positive|positive_profit_concentration_35      | reject_opposing_premium_absorption_continuation |

| candidate                                          | scope       |   events |   active_days |   active_months |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |
|:---------------------------------------------------|:------------|---------:|--------------:|----------------:|----------------:|----------------------:|---------------------:|-------------------:|
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | all         |      178 |           116 |              12 |         -4.5872 |              -14.5872 |             -19.5872 |             0.3427 |
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | development |       73 |            51 |               6 |          2.8602 |               -7.1398 |             -12.1398 |             0.4247 |
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | validation  |       42 |            20 |               2 |        -12.6079 |              -22.6079 |             -27.6079 |             0.2857 |
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | holdout     |       63 |            45 |               4 |         -7.8696 |              -17.8696 |             -22.8696 |             0.2857 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | all         |       69 |            54 |              11 |          7.6302 |              -22.3698 |             -32.3698 |             0.1159 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | development |       18 |            16 |               5 |         17.7435 |              -12.2565 |             -22.2565 |             0.1111 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | validation  |       21 |            12 |               2 |          1.1559 |              -28.8441 |             -38.8441 |             0.0476 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | holdout     |       30 |            26 |               4 |          6.0942 |              -23.9058 |             -33.9058 |             0.1667 |

| candidate                                          | source_side   |   events |   mean_gross_bp |   mean_primary_net_bp |
|:---------------------------------------------------|:--------------|---------:|----------------:|----------------------:|
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | up_move       |       83 |         -7.6421 |              -17.6421 |
| PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION  | down_move     |       95 |         -1.9183 |              -11.9183 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | up_move       |       19 |          5.3290 |              -24.6710 |
| PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET | down_move     |       50 |          8.5046 |              -21.4954 |

Premium OHLC values are exact completed bars with shifted historical
scales and no forward filling. No live, PaperLive, application, leverage,
remote, or order scope changed.

# v17.8 BTC Confirmed-Flow Laggard Findings

Verdict: `reject_btc_confirmed_flow_laggard`.

| candidate                          |   events |   mean_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   delayed_primary_net_bp |   reversed_primary_net_bp |   positive_month_share | eligible   | failed_gates                                                                                                                                                                                                                                                                            | verdict                           |
|:-----------------------------------|---------:|----------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------:|--------------------------:|-----------------------:|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------|
| BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP |      139 |              -28.9606 |              -45.7061 |                -9.9940 |                     0.3600 |                 -31.9985 |                  -11.0394 |                 1.0000 | False      | holdout_events_25|development_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|beats_reversed_direction|source_q95_positive|source_q99_positive|holding_15m_positive|holding_60m_positive|positive_month_share_35    | reject_btc_confirmed_flow_laggard |
| BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP   |      139 |              -28.3660 |              -32.6407 |               -24.4546 |                     0.6340 |                 -30.8225 |                  -31.6340 |               inf      | False      | holdout_events_25|development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_95|source_q95_positive|source_q99_positive|holding_15m_positive|holding_60m_positive|positive_month_share_35 | reject_btc_confirmed_flow_laggard |

| candidate                          | scope       |   events |   active_days |   active_months |   mean_laggards |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |
|:-----------------------------------|:------------|---------:|--------------:|----------------:|----------------:|----------------:|----------------------:|---------------------:|-------------------:|
| BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP | all         |      139 |           103 |              11 |          4.7338 |         -8.9606 |              -28.9606 |             -38.9606 |             0.2950 |
| BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP | development |       94 |            68 |               6 |          4.8936 |        -21.0379 |              -41.0379 |             -51.0379 |             0.2872 |
| BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP | validation  |       22 |            17 |               2 |          4.7727 |         55.5595 |               35.5595 |              25.5595 |             0.4545 |
| BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP | holdout     |       23 |            18 |               3 |          4.0435 |        -21.3160 |              -41.3160 |             -51.3160 |             0.1739 |
| BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP   | all         |      139 |           103 |              11 |          4.7338 |          1.6340 |              -28.3660 |             -38.3660 |             0.0863 |
| BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP   | development |       94 |            68 |               6 |          4.8936 |          0.4627 |              -29.5373 |             -39.5373 |             0.0957 |
| BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP   | validation  |       22 |            17 |               2 |          4.7727 |          4.9722 |              -25.0278 |             -35.0278 |             0.0909 |
| BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP   | holdout     |       23 |            18 |               3 |          4.0435 |          3.2279 |              -26.7721 |             -36.7721 |             0.0435 |

Signals use only completed Binance USD-M 15m bars. Graphs are frozen from
prior-month history and laggards are selected at the closed signal bar.
No PaperLive, application, leverage, remote, or real-order permission changes.

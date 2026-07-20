# v17.5 Deribit Skew Receiver-vs-Insulator Spread Findings

Verdict: `reject_receiver_insulator_spread`.

| candidate                                 |   events |   mean_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   delayed_primary_net_bp |   positive_year_share | eligible   | failed_gates                                                                                                                                                                                                        | verdict                          |
|:------------------------------------------|---------:|----------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------:|----------------------:|:-----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR |       31 |              -40.0408 |              -80.9578 |                -1.4206 |                     0.2420 |                   5.8210 |                1.0000 | False      | full_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_90|beats_one_day_delay|holding_8h_positive|positive_year_share_50 | reject_receiver_insulator_spread |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR |       27 |               22.8102 |              -48.0596 |               111.3352 |                     0.9840 |                 -49.6415 |                0.6034 | False      | holdout_events_8|holdout_primary_positive|bootstrap_lower_positive|holding_8h_positive|positive_year_share_50                                                                                                       | reject_receiver_insulator_spread |

| candidate                                 | scope       |   events |   active_years |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |   sum_primary_net |
|:------------------------------------------|:------------|---------:|---------------:|----------------:|----------------------:|---------------------:|-------------------:|------------------:|
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | all         |       31 |              6 |         -0.0408 |              -40.0408 |             -60.0408 |             0.2903 |           -0.1241 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | development |       14 |              3 |         -0.8834 |              -40.8834 |             -60.8834 |             0.2857 |           -0.0572 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | validation  |        6 |              1 |        -39.7209 |              -79.7209 |             -99.7209 |             0.1667 |           -0.0478 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | holdout     |       11 |              2 |         22.6754 |              -17.3246 |             -37.3246 |             0.3636 |           -0.0191 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | all         |       27 |              6 |         62.8102 |               22.8102 |               2.8102 |             0.5185 |            0.0616 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | development |       13 |              3 |         42.2233 |                2.2233 |             -17.7767 |             0.6154 |            0.0029 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | validation  |        9 |              1 |        109.3416 |               69.3416 |              49.3416 |             0.4444 |            0.0624 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | holdout     |        5 |              2 |         32.5795 |               -7.4205 |             -27.4205 |             0.4000 |           -0.0037 |

Both legs use closed Binance USD-M hourly prices and unit gross exposure.
Deribit option trades are signal-only. No live permission changes.

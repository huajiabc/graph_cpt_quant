# v17.3 Deribit Skew-to-Receiver-Bucket Findings

Verdict: `reject_deribit_skew_receiver_alpha`.

| candidate                        |   events |   mean_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   delayed_primary_net_bp |   positive_year_share | eligible   | failed_gates                                                                                                                                                                                  | verdict                            |
|:---------------------------------|---------:|----------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------:|----------------------:|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------|
| DSR1_STRESS_RECEIVER_SHORT       |       31 |              -37.7042 |             -181.8937 |               112.9488 |                     0.4220 |                -118.2257 |                0.4282 | False      | full_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_90|threshold_075_positive|threshold_125_positive                                                 | reject_deribit_skew_receiver_alpha |
| DSR2_STRESS_RECEIVER_BTC_NEUTRAL |       31 |              -47.2356 |              -95.6211 |                 2.6151 |                     0.2200 |                 -50.1027 |                0.6557 | False      | full_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_90|threshold_075_positive|threshold_125_positive|positive_year_share_50 | reject_deribit_skew_receiver_alpha |

| candidate                        | scope       |   events |   active_years |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |   sum_primary_net |
|:---------------------------------|:------------|---------:|---------------:|----------------:|----------------------:|---------------------:|-------------------:|------------------:|
| DSR1_STRESS_RECEIVER_SHORT       | all         |       31 |              6 |        -17.7042 |              -37.7042 |             -57.7042 |             0.5161 |           -0.1169 |
| DSR1_STRESS_RECEIVER_SHORT       | development |       14 |              3 |       -109.5078 |             -129.5078 |            -149.5078 |             0.2857 |           -0.1813 |
| DSR1_STRESS_RECEIVER_SHORT       | validation  |        6 |              1 |        110.4687 |               90.4687 |              70.4687 |             0.6667 |            0.0543 |
| DSR1_STRESS_RECEIVER_SHORT       | holdout     |       11 |              2 |         29.2243 |                9.2243 |             -10.7757 |             0.7273 |            0.0101 |
| DSR2_STRESS_RECEIVER_BTC_NEUTRAL | all         |       31 |              6 |        -17.2356 |              -47.2356 |             -67.2356 |             0.1935 |           -0.1464 |
| DSR2_STRESS_RECEIVER_BTC_NEUTRAL | development |       14 |              3 |        -59.5008 |              -89.5008 |            -109.5008 |             0.1429 |           -0.1253 |
| DSR2_STRESS_RECEIVER_BTC_NEUTRAL | validation  |        6 |              1 |         57.6471 |               27.6471 |               7.6471 |             0.1667 |            0.0166 |
| DSR2_STRESS_RECEIVER_BTC_NEUTRAL | holdout     |       11 |              2 |         -4.2887 |              -34.2887 |             -54.2887 |             0.2727 |           -0.0377 |
| DSR3_RELIEF_RECEIVER_LONG        | all         |       27 |              6 |        179.0828 |              159.0828 |             139.0828 |             0.5926 |            0.4295 |
| DSR3_RELIEF_RECEIVER_LONG        | development |       13 |              3 |         66.8448 |               46.8448 |              26.8448 |             0.5385 |            0.0609 |
| DSR3_RELIEF_RECEIVER_LONG        | validation  |        9 |              1 |        444.7235 |              424.7235 |             404.7235 |             0.8889 |            0.3823 |
| DSR3_RELIEF_RECEIVER_LONG        | holdout     |        5 |              2 |         -7.2517 |              -27.2517 |             -47.2517 |             0.2000 |           -0.0136 |

BTC option trade surfaces are used only as completed-day signals. The
traded legs are Binance USD-M closed-bar returns; option execution is not
simulated. The BTC-neutral result is normalized to unit gross exposure.
No PaperLive, application, remote, leverage, or real-order permission changes.

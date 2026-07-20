# v20.7 Unhedged Flow-Exhaustion Diagnostic Findings

Verdict: `reject_unhedged_flow_exhaustion_diagnostic`.

| candidate                              |   events |   mean_receiver_gross_bp |   mean_receiver_primary_net_bp |   mean_btc_control_gross_bp |   mean_receiver_minus_btc_bp |   random_control_percentile | eligible_for_natural_forward_observation   | status            |
|:---------------------------------------|---------:|-------------------------:|-------------------------------:|----------------------------:|-----------------------------:|----------------------------:|:-------------------------------------------|:------------------|
| RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE |       53 |                  50.8856 |                        30.8856 |                     14.1159 |                      36.7697 |                      0.7620 | False                                      | diagnostic_reject |

| candidate                              | scope       |   events |   active_days |   active_months |   mean_receivers |   mean_receiver_gross_bp |   mean_receiver_primary_net_bp |   mean_receiver_stress_net_bp |   mean_btc_control_gross_bp |   mean_btc_control_primary_net_bp |   mean_receiver_minus_btc_bp |   mean_reversed_primary_net_bp |   positive_receiver_primary_fraction |
|:---------------------------------------|:------------|---------:|--------------:|----------------:|-----------------:|-------------------------:|-------------------------------:|------------------------------:|----------------------------:|----------------------------------:|-----------------------------:|-------------------------------:|-------------------------------------:|
| RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE | all         |       53 |            28 |              11 |           3.8679 |                  50.8856 |                        30.8856 |                       10.8856 |                     14.1159 |                           -5.8841 |                      36.7697 |                       -70.8856 |                               0.5094 |
| RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE | development |       30 |            15 |               5 |           3.5667 |                  64.2593 |                        44.2593 |                       24.2593 |                     18.4357 |                           -1.5643 |                      45.8236 |                       -84.2593 |                               0.4667 |
| RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE | validation  |       15 |             5 |               2 |           4.6000 |                  51.1588 |                        31.1588 |                       11.1588 |                     20.0453 |                            0.0453 |                      31.1135 |                       -71.1588 |                               0.6667 |
| RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE | holdout     |        8 |             8 |               4 |           3.6250 |                   0.2217 |                       -19.7783 |                      -39.7783 |                    -13.2014 |                          -33.2014 |                      13.4232 |                       -20.2217 |                               0.3750 |

This is a post-hoc decomposition diagnostic, not independent alpha evidence. Passing could only justify untouched natural-forward observation.

No live, PaperLive, application, leverage, remote, or order state changed.

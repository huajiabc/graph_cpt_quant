# v15.3 Crowded-Source Exhaustion Reversal Findings

Verdict: `reject_candidate`.

| candidate                              |   events |   event_days |   months |   validation_events |   holdout_events |   long_only_events |   short_only_events |   two_sided_events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_source_percentile |   positive_month_concentration |   worst_period_bp |   worst_event_bp |   reversed_control_mean_bp |   shifted_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:---------------------------------------|---------:|-------------:|---------:|--------------------:|-----------------:|-------------------:|--------------------:|-------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------------:|------------------:|-----------------:|---------------------------:|--------------------------:|----------------------------:|---------------------------:|:----------|
| LR1_CROWDED_SOURCE_EXHAUSTION_REVERSAL |      412 |          213 |       11 |                 134 |              115 |                389 |                  23 |                  0 |          3.0393 |              -36.9607 |             -76.9607 |                     -33.4008 |                    -39.7060 |                 -38.8075 |              -51.7082 |               -22.7355 |                    59.0000 |                            inf |          -39.7060 |       -1528.5100 |                   -43.0393 |                  -52.8922 |                      0.0000 |                     0.0000 | False     |

## Controls

| control                          |   events |   mean_primary_net_bp |
|:---------------------------------|---------:|----------------------:|
| LR1_REVERSED_SOURCE_CONTINUATION |      412 |              -43.0393 |
| LR1_CROWDING_SHIFTED_24H         |      326 |              -52.8922 |

The v11.8 event thresholds and causal feature timing were reused
unchanged. This study trades source coins, not community followers.
PaperLive and remote state are unchanged.

# v15.9 Hourly Cross-Venue Depth Imbalance Findings

Verdict: `reject_candidate`.

| candidate                                 |   hours |   months |   validation_hours |   holdout_hours |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_ranking_percentile |   positive_month_concentration |   mean_one_way_turnover |   reversed_control_mean_bp |   stale_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:------------------------------------------|--------:|---------:|-------------------:|----------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------------:|-------------------------------:|------------------------:|---------------------------:|------------------------:|----------------------------:|---------------------------:|:----------|
| BD3_HOURLY_CROSS_VENUE_DEPTH_CONTINUATION |    9055 |       13 |               2135 |            2519 |         -0.1756 |               -4.4186 |              -8.6617 |                      -4.5273 |                     -3.0131 |                  -5.4200 |               -4.8538 |                -3.9773 |                    100.0000 |                            inf |                  0.2122 |                    -4.0674 |                 -4.2909 |                      0.0000 |                     0.0000 | False     |

## Frozen controls

| control                                 |   hours |   mean_primary_net_bp |
|:----------------------------------------|--------:|----------------------:|
| BD3_REVERSED_HOURLY_DEPTH_REVERSAL      |    9055 |               -4.0674 |
| BD3_ONE_HOUR_STALE_DEPTH                |    9055 |               -4.2909 |
| BD3_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE |    9055 |               -1.7460 |

Every feature uses only snapshots strictly before the Bybit entry
hour. The 5% row is diagnostic-only. PaperLive and remote state are
unchanged.

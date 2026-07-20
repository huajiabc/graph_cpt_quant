# v16.3 Within-Bucket Liquidity Quality Findings

Verdict: `reject_candidate`.

| candidate                           |   weeks |   months |   validation_weeks |   holdout_weeks |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_quality_pairing_percentile |   positive_month_concentration |   mean_one_way_turnover |   reversed_control_mean_bp |   primary_stale_overlap_mean_bp |   stale_control_mean_bp |   raw_depth_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:------------------------------------|--------:|---------:|-------------------:|----------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------------------------:|-------------------------------:|------------------------:|---------------------------:|--------------------------------:|------------------------:|----------------------------:|----------------------------:|---------------------------:|:----------|
| LQ1_WITHIN_BUCKET_LIQUIDITY_QUALITY |      51 |       13 |                 12 |              14 |          2.7031 |               -5.8870 |             -14.4771 |                      -6.6954 |                    -22.6803 |                   9.9508 |              -46.3135 |                32.2907 |                             73.4000 |                         0.3399 |                  0.4295 |                   -11.2932 |                         -3.0040 |                 16.0094 |                     -6.9304 |                      0.0000 |                     0.0000 | False     |

## Frozen controls

| control                                  |   weeks |   mean_primary_net_bp |
|:-----------------------------------------|--------:|----------------------:|
| LQ1_REVERSED_LIQUIDITY_FRAGILITY_PREMIUM |      51 |              -11.2932 |
| LQ1_ONE_WEEK_STALE_QUALITY               |      49 |               16.0094 |
| LQ1_RAW_DEPTH_ONLY                       |      51 |               -6.9304 |
| LQ1_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE  |      51 |               23.0291 |

The graph pairs, seven-day depth/volume ratio, direction, beta hedge,
costs and gates were frozen before returns. The 5% row is diagnostic-only.
PaperLive and remote state are unchanged.

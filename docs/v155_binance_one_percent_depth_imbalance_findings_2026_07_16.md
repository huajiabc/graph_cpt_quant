# v15.5 Binance One-Percent Depth Imbalance Findings

Verdict: `reject_candidate`.

| candidate                                    |   days |   months |   validation_days |   holdout_days |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_ranking_percentile |   positive_month_concentration |   mean_one_way_turnover |   reversed_control_mean_bp |   stale_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:---------------------------------------------|-------:|---------:|------------------:|---------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------------:|-------------------------------:|------------------------:|---------------------------:|------------------------:|----------------------------:|---------------------------:|:----------|
| BD2_PRIOR_DAY_ONE_PERCENT_DEPTH_CONTINUATION |    375 |       13 |                88 |            105 |         -6.7952 |              -10.1230 |             -13.4508 |                      -6.2766 |                      8.8272 |                 -32.6723 |              -19.7146 |                -1.3064 |                     98.4000 |                         0.4940 |                  0.1664 |                     3.4674 |                -11.0272 |                      0.0000 |                     0.0000 | False     |

## Frozen controls

| control                                 |   days |   mean_primary_net_bp |
|:----------------------------------------|-------:|----------------------:|
| BD2_REVERSED_ONE_PERCENT_DEPTH_REVERSAL |    375 |                3.4674 |
| BD2_STALE_TWO_DAY_ONE_PERCENT_DEPTH     |    375 |              -11.0272 |
| BD2_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE |    375 |               -2.2158 |

The 1% continuation direction, 4/4 bucket, top/bottom-eight holding
band, 24-hour horizon, beta hedge, cost model, sample splits and gates
were frozen before inspecting any portfolio return. The 5% row is
diagnostic-only and cannot be promoted from this study. PaperLive and
remote state are unchanged.

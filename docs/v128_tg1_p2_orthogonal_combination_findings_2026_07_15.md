# v12.8 TG1 + Frozen P2 Orthogonal Combination Findings

Verdict: `reject_combination`.

| candidate             |   weeks |   months |   validation_weeks |   holdout_weeks |   p2_trades |   active_p2_weeks |   sleeve_correlation |   mean_tg1_bp |   mean_p2_bp |   mean_combined_bp |   development_combined_bp |   validation_combined_bp |   holdout_combined_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   positive_month_concentration |   worst_period_bp |   shifted_p2_mean_bp | promote   |
|:----------------------|--------:|---------:|-------------------:|----------------:|------------:|------------------:|---------------------:|--------------:|-------------:|-------------------:|--------------------------:|-------------------------:|----------------------:|----------------------:|-----------------------:|-------------------------------:|------------------:|---------------------:|:----------|
| CM1_50_50_TG1_P2_MAX8 |      43 |       10 |                 13 |               8 |         115 |                22 |               0.1848 |       21.1963 |      15.2958 |            18.2461 |                   33.3247 |                  -1.0500 |                8.1360 |                0.4118 |                38.1587 |                         0.2439 |           -1.0500 |              18.2461 | False     |

The 50/50 capital weight and exact frozen sleeves were registered before alignment. No existing PaperLive strategy was changed.

# v12.9 Causal Risk-Parity Combination Findings

Verdict: `reject_combination`.

| candidate            |   weeks |   months |   validation_weeks |   holdout_weeks |   sleeve_correlation |   mean_tg1_bp |   mean_p2_bp |   mean_tg1_weight |   mean_allocation_turnover |   mean_allocation_cost_bp |   mean_combined_bp |   development_combined_bp |   validation_combined_bp |   holdout_combined_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   positive_month_concentration |   worst_period_bp | promote   |
|:---------------------|--------:|---------:|-------------------:|----------------:|---------------------:|--------------:|-------------:|------------------:|---------------------------:|--------------------------:|-------------------:|--------------------------:|-------------------------:|----------------------:|----------------------:|-----------------------:|-------------------------------:|------------------:|:----------|
| RP1_CAUSAL_8W_TG1_P2 |      43 |       10 |                 13 |               8 |               0.1848 |       21.1963 |      15.2958 |            0.5679 |                     0.0306 |                    0.6115 |            19.1653 |                   33.1312 |                  -3.0598 |               16.8750 |                2.9758 |                37.8306 |                         0.2318 |           -3.0598 | False     |

The eight-week causal weighting rule, bounds, and allocation cost were frozen before inspection. PaperLive was not changed.

# v13.6 Severity-Weighted Negative-Funding Basket Findings

Verdict: `reject_as_tradable_alpha`.

| candidate                                      |   weeks |   months |   validation_weeks |   holdout_weeks |   contracted_weeks |   median_negative_breadth |   mean_maximum_base_weight |   mean_turnover |   mean_price_bp |   mean_funding_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   contracted_primary_net_bp |   broad_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   max_abs_residual_btc_beta | promote   |
|:-----------------------------------------------|--------:|---------:|-------------------:|----------------:|-------------------:|--------------------------:|---------------------------:|----------------:|----------------:|------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|----------------------------:|:----------|
| NF3_ALL_NEGATIVE_Q75_SEVERITY_BTC_BETA_NEUTRAL |      49 |       12 |                 13 |              14 |                 10 |                   25.0000 |                     0.1475 |          0.4515 |         34.2107 |           69.1706 |        103.3813 |               94.3522 |              85.3231 |                     143.8468 |                     76.1448 |                  33.4818 |                   -125.9264 |               150.8339 |                3.9725 |               175.9549 |           38.3000 |                         0.3198 |           33.4818 |                      0.0000 | False     |

The severity cap, continuous allocation, beta hedge, costs, states,
and permutation controls were frozen before return inspection. No
PaperLive, leverage, or strategy-status permission changed.

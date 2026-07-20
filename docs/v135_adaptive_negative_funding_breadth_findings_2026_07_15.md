# v13.5 Adaptive-Breadth Negative-Funding Rebound Findings

Verdict: `reject_as_tradable_alpha`.

| candidate                                |   weeks |   months |   validation_weeks |   holdout_weeks |   contracted_weeks |   median_selected_breadth |   mean_turnover |   mean_price_bp |   mean_funding_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   full9_primary_net_bp |   contracted_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   max_abs_residual_btc_beta | promote   |
|:-----------------------------------------|--------:|---------:|-------------------:|----------------:|-------------------:|--------------------------:|----------------:|----------------:|------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|-----------------------:|----------------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|----------------------------:|:----------|
| NF2_ADAPTIVE4TO9_HOLD18_BTC_BETA_NEUTRAL |      49 |       12 |                 13 |              14 |                 10 |                    9.0000 |          0.3252 |         79.1447 |           74.8636 |        154.0083 |              147.5044 |             141.0005 |                     244.4382 |                     33.4705 |                 101.0686 |               159.7889 |                     99.5951 |               52.4187 |               249.5638 |           89.4000 |                         0.3044 |           33.4705 |                      0.0000 | False     |

The breadth rule, beta hedge, costs, states, and controls were frozen
before this return was inspected. PaperLive and leverage permissions
remain unchanged.

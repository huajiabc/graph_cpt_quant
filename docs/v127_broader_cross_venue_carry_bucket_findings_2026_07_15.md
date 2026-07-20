# v12.7 Broader Cross-Venue Carry Bucket Findings

Verdict: `reject_as_tradable_alpha`.

| candidate            |   weeks |   months |   validation_weeks |   holdout_weeks |   mean_turnover |   mean_retained_names |   mean_price_basis_bp |   mean_funding_spread_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp | promote   |
|:---------------------|--------:|---------:|-------------------:|----------------:|----------------:|----------------------:|----------------------:|-------------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|:----------|
| BD1_30D_TOP12_HOLD24 |      42 |       10 |                 13 |               8 |          0.2897 |               10.2619 |                1.1038 |                  21.7435 |         22.8473 |               17.0536 |              11.2600 |                      27.6229 |                      3.3925 |                  11.5087 |                3.7677 |                31.7300 |           99.9000 |                         0.4333 |            3.3925 | False     |

The top-12/hold-24 breadth rule and all gates were frozen before this return was inspected. No existing PaperLive strategy was changed.

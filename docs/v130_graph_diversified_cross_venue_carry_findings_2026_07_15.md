# v13.0 Graph-Diversified Cross-Venue Carry Findings

Verdict: `reject_candidate`.

| candidate                    |   weeks |   months |   validation_weeks |   holdout_weeks |   mean_invested_exposure |   mean_turnover |   mean_price_basis_bp |   mean_funding_spread_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_partition_percentile |   positive_month_concentration |   worst_period_bp | promote   |
|:-----------------------------|--------:|---------:|-------------------:|----------------:|-------------------------:|----------------:|----------------------:|-------------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------------------:|-------------------------------:|------------------:|:----------|
| GC1_30D_COMMUNITY_TOP1_HOLD2 |      42 |       10 |                 13 |               8 |                   0.9524 |          0.5417 |                2.8602 |                  22.2957 |         25.1558 |               14.3225 |               3.4892 |                      29.9254 |                     -1.1744 |                  -1.4525 |               -1.2863 |                36.4287 |                       39.0000 |                         0.6864 |           -1.4525 | False     |

The frozen communities and random-partition controls use the same top-1/hold-2 implementation. PaperLive was not changed.

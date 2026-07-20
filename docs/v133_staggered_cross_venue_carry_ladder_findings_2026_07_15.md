# v13.3 Staggered Cross-Venue Carry Ladder Findings

Verdict: `reject_as_tradable_alpha`.

| candidate                   |   weeks |   months |   validation_weeks |   holdout_weeks |   median_coverage |   mean_turnover |   mean_price_basis_bp |   mean_funding_spread_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   monday_tg1_correlation |   monday_overlap_weeks | promote   |
|:----------------------------|--------:|---------:|-------------------:|----------------:|------------------:|----------------:|----------------------:|-------------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|-------------------------:|-----------------------:|:----------|
| SL1_7COHORT_30D_TOP9_HOLD18 |      48 |       12 |                 13 |              14 |           70.0000 |          0.3122 |                0.9359 |                  21.4961 |         22.4320 |               16.1886 |               9.9452 |                      39.9884 |                    -18.7743 |                  12.9544 |               -0.3090 |                35.0476 |          100.0000 |                         0.3924 |          -18.7743 |                   0.8005 |                     48 | False     |

The seven weekday cohorts use only prior settled funding. Returns are
non-overlapping calendar-week sums of daily marked pair PnL. The first
week is a burn-in whose entry cost is carried into evaluation; terminal
close cost is charged to the last complete week. PaperLive was unchanged.

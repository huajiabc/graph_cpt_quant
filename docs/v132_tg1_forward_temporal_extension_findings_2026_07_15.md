# v13.2 Exact TG1 Forward Temporal-Extension Findings

Verdict: `reject_as_tradable_alpha`.

| candidate                       |   weeks |   months |   validation_weeks |   holdout_weeks |   median_coverage |   mean_turnover |   mean_retained_names |   mean_price_basis_bp |   mean_funding_spread_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp | promote   |
|:--------------------------------|--------:|---------:|-------------------:|----------------:|------------------:|----------------:|----------------------:|----------------------:|-------------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|:----------|
| TG1_FORWARD_EXTENDED_TO_2026_07 |      49 |       12 |                 13 |              14 |           70.0000 |          0.3175 |                7.5714 |                1.0508 |                  23.4996 |         24.5504 |               18.2012 |              11.8520 |                      35.6214 |                      1.5152 |                   6.3208 |                3.7160 |                34.6827 |          100.0000 |                         0.4152 |            1.5152 | False     |

## Newly added weeks

| entry_time                | selected_symbols                                                                   |   funding_spread_return |   price_basis_return |   realized_turnover |   primary_net_return |
|:--------------------------|:-----------------------------------------------------------------------------------|------------------------:|---------------------:|--------------------:|---------------------:|
| 2026-06-01 00:00:00+00:00 | MERLUSDT|ETHFIUSDT|PYTHUSDT|PIPPINUSDT|BRETTUSDT|DRIFTUSDT|ZENUSDT|AXSUSDT|TWTUSDT |                0.000826 |            -0.000128 |            0.666667 |            -0.000635 |
| 2026-06-08 00:00:00+00:00 | MERLUSDT|ETHFIUSDT|PYTHUSDT|PIPPINUSDT|BRETTUSDT|DRIFTUSDT|ZENUSDT|AXSUSDT|TWTUSDT |                0.000600 |            -0.000616 |            0.000000 |            -0.000016 |
| 2026-06-15 00:00:00+00:00 | ETHFIUSDT|PYTHUSDT|PIPPINUSDT|BRETTUSDT|DRIFTUSDT|ZENUSDT|AXSUSDT|TWTUSDT|NMRUSDT  |               -0.000105 |            -0.000273 |            0.222222 |            -0.000822 |
| 2026-06-22 00:00:00+00:00 | ETHFIUSDT|PYTHUSDT|PIPPINUSDT|BRETTUSDT|DRIFTUSDT|ZENUSDT|AXSUSDT|TWTUSDT|NMRUSDT  |                0.000608 |             0.000128 |            0.000000 |             0.000736 |
| 2026-06-29 00:00:00+00:00 | ETHFIUSDT|PYTHUSDT|PIPPINUSDT|ZENUSDT|TWTUSDT|NMRUSDT|TRUMPUSDT|MAGICUSDT|UNIUSDT  |                0.000197 |            -0.000913 |            0.666667 |            -0.002049 |
| 2026-07-06 00:00:00+00:00 | ETHFIUSDT|PYTHUSDT|PIPPINUSDT|ZENUSDT|TWTUSDT|NMRUSDT|TRUMPUSDT|UNIUSDT|MERLUSDT   |                0.000892 |             0.000381 |            1.222222 |            -0.001172 |

Historical overlap maximum absolute return difference: `0.000e+00`. PaperLive was not changed.

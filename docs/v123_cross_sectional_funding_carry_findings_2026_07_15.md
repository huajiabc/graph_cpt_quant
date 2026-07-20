# v12.3 Cross-Sectional Funding Carry Findings

Verdict: `reject_all_as_tradable_alpha`.

| candidate                   |   weeks |   months |   validation_weeks |   holdout_weeks |   median_coverage |   mean_price_bp |   mean_funding_bp |   mean_gross_bp |   mean_net_40bp_bp |   mean_net_60bp_bp |   mean_turnover_net_bp |   mean_residual_net_40bp_bp |   development_net_40bp_bp |   validation_net_40bp_bp |   holdout_net_40bp_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp | promote   |
|:----------------------------|--------:|---------:|-------------------:|----------------:|------------------:|----------------:|------------------:|----------------:|-------------------:|-------------------:|-----------------------:|----------------------------:|--------------------------:|-------------------------:|----------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|:----------|
| FC1_7D_FUNDING_CARRY        |      43 |       10 |                 13 |               8 |           72.0000 |         52.4537 |           82.3517 |        134.8053 |            94.8053 |            74.8053 |               117.1309 |                     99.3648 |                  170.5897 |                  19.1967 |                9.2624 |              -52.1057 |               247.6951 |           98.8000 |                         0.3429 |            9.2624 | False     |
| FC2_30D_FUNDING_CARRY       |      43 |       10 |                 13 |               8 |           72.0000 |        -31.2581 |           92.7711 |         61.5130 |            21.5130 |             1.5130 |                52.9859 |                     15.0372 |                   46.5223 |                  23.3436 |              -50.2369 |             -115.7823 |               172.5044 |           87.6000 |                         0.4384 |          -50.2369 | False     |
| FC3_COMMUNITY_NEUTRAL_CARRY |      43 |       10 |                 13 |               8 |           72.0000 |        -33.2403 |           58.8776 |         25.6372 |           -14.3628 |           -34.3628 |                 2.3814 |                    -20.6934 |                   68.2362 |                -239.1916 |              123.8370 |             -157.3228 |               119.4043 |            1.5000 |                         0.3387 |         -239.1916 | False     |

Funding, price, and BTC-residual components use only as-of settlements and exact seven-day closes. No existing PaperLive strategy was changed.

## Interpretation

`FC1_7D_FUNDING_CARRY` is a near-candidate, not a promoted strategy. It passed
sample breadth, all three chronological period signs, stress cost, BTC residual,
null percentile, month concentration, and worst-period gates. Its only formal
failure was the bootstrap 95% lower bound (`-52.11 bp/week`). Sleeve attribution
showed that the funding cash flow came mainly from the long extremely-negative
funding contracts, not from shorting high-positive funding. That observation
motivated the separately preregistered same-coin hedge studies v12.4-v13.2; it
does not authorize a post-hoc v12.3 overlay.

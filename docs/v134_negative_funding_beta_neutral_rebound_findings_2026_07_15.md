# v13.4 Negative-Funding Beta-Neutral Rebound Findings

Verdict: `reject_as_tradable_alpha`.

| candidate                        |   weeks |   months |   validation_weeks |   holdout_weeks |   median_coverage |   median_eligible_negative_names |   mean_turnover |   mean_long_notional |   mean_btc_short_notional |   mean_price_bp |   mean_funding_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   max_abs_residual_btc_beta | promote   |
|:---------------------------------|--------:|---------:|-------------------:|----------------:|------------------:|---------------------------------:|----------------:|---------------------:|--------------------------:|----------------:|------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|----------------------------:|:----------|
| NF1_LOW9_HOLD18_BTC_BETA_NEUTRAL |      39 |       10 |                 13 |              12 |           72.0000 |                          27.0000 |          0.4405 |               0.4594 |                    0.5406 |         82.5468 |           79.7475 |        162.2943 |              153.4834 |             144.6724 |                     236.0598 |                     33.4705 |                 187.1581 |               36.6627 |               276.5832 |           90.2000 |                         0.3480 |           33.4705 |                      0.0000 | False     |

Signal, beta, prices, and funding are strictly causal. Gross notional
is one and the estimated BTC beta is algebraically neutralized. No
PaperLive, leverage, or status permission changed.

# v16.1 Hourly Liquidity-Withdrawal Amplification Findings

Verdict: `reject_candidate`.

| candidate                                     |   hours |   months |   validation_hours |   holdout_hours |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_depth_pairing_percentile |   positive_month_concentration |   mean_one_way_turnover |   price_only_mean_bp |   reversed_control_mean_bp |   stale_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:----------------------------------------------|--------:|---------:|-------------------:|----------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------------------:|-------------------------------:|------------------------:|---------------------:|---------------------------:|------------------------:|----------------------------:|---------------------------:|:----------|
| LW1_HOURLY_LIQUIDITY_WITHDRAWAL_AMPLIFICATION |    9047 |       13 |               2134 |            2518 |         -0.0382 |              -21.3422 |             -42.6462 |                     -21.5269 |                    -20.9075 |                 -21.3882 |              -21.8824 |               -20.7920 |                           56.5000 |                            inf |                  1.0652 |             -21.5932 |                   -21.2657 |                -21.4074 |                      0.0000 |                     0.0000 | False     |

## Frozen controls

| control                                         |   hours |   mean_primary_net_bp |
|:------------------------------------------------|--------:|----------------------:|
| LW1_PRICE_ONLY_RESIDUAL_MOMENTUM                |    9047 |              -21.5932 |
| LW1_REVERSED_WITHDRAWAL_AMPLIFICATION           |    9047 |              -21.2657 |
| LW1_ONE_HOUR_STALE_WITHDRAWAL_SIGNAL            |    9047 |              -21.4074 |
| LW1_FIVE_PERCENT_DIAGNOSTIC_ONLY_NON_PROMOTABLE |    9047 |              -21.4199 |

Total-depth change, residual-price direction, score multiplication,
costs and gates were frozen before inspecting returns. The 5% row is
diagnostic-only. PaperLive and remote state are unchanged.

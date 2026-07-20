# v12.1 Top-Trader Divergence and Community Rotation Findings

Verdict: `reject_all_as_tradable_alpha`.

The Binance metrics panel contains 570,813 admissible hourly rows across 71 symbols. After joining frozen memberships, prices, and BTC betas, 129,268 symbol-time observations remained.

## Candidate results

| candidate                      |   decisions |   months |   median_coverage | chosen_direction   |   mean_gross_bp |   mean_net_40bp_bp |   mean_net_60bp_bp |   mean_turnover_net_bp |   mean_residual_net_40bp_bp |   mean_signed_ic |   development_net_40bp_bp |   validation_net_40bp_bp |   holdout_net_40bp_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   max_drawdown | promote   |
|:-------------------------------|------------:|---------:|------------------:|:-------------------|----------------:|-------------------:|-------------------:|-----------------------:|----------------------------:|-----------------:|--------------------------:|-------------------------:|----------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|---------------:|:----------|
| TD1_POSITION_VS_CROWD          |        1838 |       11 |           70.0000 | reversal           |          1.3226 |           -38.6774 |           -58.6774 |                -5.8374 |                    -38.6492 |           0.0082 |                  -38.1137 |                 -43.0907 |              -33.7677 |              -41.3665 |               -36.0704 |           77.5000 |                            nan |          -43.0907 |        -0.9992 | False     |
| TD2_POSITION_VS_TOP_ACCOUNTS   |        1838 |       11 |           70.0000 | reversal           |         -1.4762 |           -41.4762 |           -61.4762 |                -8.4584 |                    -41.4018 |           0.0079 |                  -39.6649 |                 -48.8940 |              -35.3107 |              -44.9170 |               -38.3148 |            4.0000 |                            nan |          -48.8940 |        -0.9995 | False     |
| TD3_DIVERGENCE_PLUS_TAKER_FLOW |        1838 |       11 |           70.0000 | reversal           |          1.2834 |           -38.7166 |           -58.7166 |               -25.7617 |                    -38.3689 |           0.0071 |                  -38.1646 |                 -41.9604 |              -35.4403 |              -40.8434 |               -36.2936 |           74.5000 |                            nan |          -41.9604 |        -0.9992 | False     |
| TD4_FROZEN_COMMUNITY_ROTATION  |        1838 |       11 |           70.0000 | continuation       |          0.5623 |           -39.4377 |           -59.4377 |               -11.8564 |                    -39.3079 |          -0.0034 |                  -39.2072 |                 -37.8936 |              -42.1887 |              -42.7054 |               -36.2658 |           46.0000 |                            nan |          -42.1887 |        -0.9993 | False     |

## Development-only direction freeze

| candidate                      |   development_continuation_mean |   chosen_sign | chosen_direction   |
|:-------------------------------|--------------------------------:|--------------:|:-------------------|
| TD1_POSITION_VS_CROWD          |                       -0.000189 |            -1 | reversal           |
| TD2_POSITION_VS_TOP_ACCOUNTS   |                       -0.000034 |            -1 | reversal           |
| TD3_DIVERGENCE_PLUS_TAKER_FLOW |                       -0.000184 |            -1 | reversal           |
| TD4_FROZEN_COMMUNITY_ROTATION  |                        0.000079 |             1 | continuation       |

## Interpretation

- The best realized-turnover result was `TD1_POSITION_VS_CROWD` at -5.84 bp
  per decision. Even the low-turnover accounting remained negative; the
  conservative full-replacement result was much worse.
- `TD1_POSITION_VS_CROWD` went from -38.11 bp net in development to -43.09 bp
  in validation. In gross terms that is +1.89 bp followed by -3.09 bp, so the
  development-only direction did not transfer chronologically.
- The frozen-community candidate reached only the 46th percentile of random
  nine-symbol partitions. Price-community topology added no positioning-flow
  attribution.
- BTC residualization did not rescue any family. The small gross effects were
  not hidden market beta, but they were far below executable costs and unstable
  by period.

The large-trader metrics remain useful state variables, but this preregistered
level-based rotation family is not a tradable alpha. A distinct follow-up must
test positioning impulses or absorption, not tune these bucket sizes or costs
after the fact.

No existing PaperLive strategy was changed.

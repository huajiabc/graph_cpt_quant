# v15.2 Negative-Funding OI-State Spread Findings

Verdict: `reject_candidate`.

| candidate                          |   weeks |   months |   validation_weeks |   holdout_weeks |   thin_weeks |   median_eligible_breadth |   mean_turnover |   cap_binding_weeks |   mean_executed_target_fraction |   max_capped_transition_turnover |   max_cap_breach |   mean_oi_state_gap |   mean_price_bp |   mean_funding_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   thin_primary_net_bp |   broad_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   null_percentile |   positive_month_concentration |   worst_period_bp |   reversed_control_mean_bp |   correlation_with_fss3 |   correlation_with_tg1 |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:-----------------------------------|--------:|---------:|-------------------:|----------------:|-------------:|--------------------------:|----------------:|--------------------:|--------------------------------:|---------------------------------:|-----------------:|--------------------:|----------------:|------------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------:|-----------------------:|------------------:|-------------------------------:|------------------:|---------------------------:|------------------------:|-----------------------:|----------------------------:|---------------------------:|:----------|
| OI1_NEG_FUNDING_BUILD_MINUS_UNWIND |      49 |       12 |                 13 |              14 |            9 |                   24.0000 |          0.7265 |                  48 |                          0.4539 |                           0.7000 |           0.0000 |              0.0601 |         67.9727 |           -5.1719 |               48.2702 |              33.7396 |                     106.9357 |                    -69.5248 |                  65.4625 |              392.1383 |               -29.1002 |              -75.6910 |               196.6924 |           78.6000 |                         0.3630 |          -69.5248 |                   -77.3314 |                  0.3280 |                -0.1347 |                      0.0000 |                     0.0000 | False     |

## Reversed control

| candidate                       |   weeks |   mean_primary_net_bp |   mean_stress_net_bp |
|:--------------------------------|--------:|----------------------:|---------------------:|
| OI1_REVERSED_UNWIND_MINUS_BUILD |      49 |              -77.3314 |             -91.8620 |

The OI direction, within-negative-funding split, execution cap and all
gates were frozen before return inspection. PaperLive is unchanged.

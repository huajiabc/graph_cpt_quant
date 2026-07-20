# v17.0 BTC Option-Skew Directional Alpha Findings

Verdict: `reject_option_skew_directional_alpha`.

| candidate                |   surface_days_with_zscore |   candidate_trades |   primary_net_bp |   stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   circular_percentile |   delayed_primary_net_bp |   reversed_primary_net_bp |   positive_month_concentration |   positive_trade_concentration |   worst_trade_bp | promote_research_followup   | failed_gates                                                                                                                                                                     |
|:-------------------------|---------------------------:|-------------------:|-----------------:|----------------:|----------------------:|-----------------------:|----------------------:|-------------------------:|--------------------------:|-------------------------------:|-------------------------------:|-----------------:|:----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OSD1_25D_SKEW_FOLLOW_BTC |                         73 |                 29 |          -7.5099 |        -17.5099 |              -78.8738 |                67.0024 |                0.5405 |                 -29.4456 |                  -12.4901 |                         0.4395 |                         0.3514 |        -577.9348 | False                       | trades_30|holdout_trades_8|validation_primary_positive|full_stress_positive|bootstrap_lower_positive|circular_percentile_95|trade_concentration_35|worst_trade_above_minus_500bp |

| scope       |   trades |   short_fraction |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate |   mean_abs_skew_zscore |
|:------------|---------:|-----------------:|----------------:|----------------------:|---------------------:|-----------:|-----------------------:|
| all         |       29 |           0.5517 |          2.4901 |               -7.5099 |             -17.5099 |     0.4828 |                 1.9500 |
| development |       12 |           0.0000 |         27.8378 |               17.8378 |               7.8378 |     0.4167 |                 1.7151 |
| validation  |       16 |           1.0000 |        -38.1614 |              -48.1614 |             -58.1614 |     0.5000 |                 2.1832 |
| holdout     |        1 |           0.0000 |        348.7428 |              338.7428 |             328.7428 |     1.0000 |                 1.0361 |

The option surface is information only; the traded leg is BTC perpetual at
10/20 bp total round-trip cost. The short 2023 archive grants no PaperLive,
leverage, remote-change, or real-order permission.

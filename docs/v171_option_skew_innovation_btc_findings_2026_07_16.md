# v17.1 BTC Option-Skew Innovation Alpha Findings

Verdict: `reject_option_skew_innovation_alpha`.

| candidate                           |   innovation_days_with_zscore |   candidate_trades |   primary_net_bp |   stress_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   circular_percentile |   delayed_primary_net_bp |   reversed_primary_net_bp |   positive_month_concentration |   positive_trade_concentration |   worst_trade_bp | promote_research_followup   | failed_gates                                                                                                                                                                                                                                                                                      |
|:------------------------------------|------------------------------:|-------------------:|-----------------:|----------------:|----------------------:|-----------------------:|----------------------:|-------------------------:|--------------------------:|-------------------------------:|-------------------------------:|-----------------:|:----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OSD2_25D_SKEW_INNOVATION_FOLLOW_BTC |                            61 |                 10 |         -63.9571 |        -73.9571 |             -201.2001 |                38.6857 |                0.2100 |                 -37.2958 |                   43.9571 |                         0.6309 |                         0.3116 |        -577.9348 | False                       | trades_25|holdout_trades_5|development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|circular_percentile_95|beats_delayed|beats_reversed|development_direction_balance|month_concentration_50|worst_trade_above_minus_500bp |

| scope       |   trades |   short_fraction |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate |
|:------------|---------:|-----------------:|----------------:|----------------------:|---------------------:|-----------:|
| all         |       10 |           0.7000 |        -53.9571 |              -63.9571 |             -73.9571 |     0.6000 |
| development |        5 |           0.8000 |          2.8136 |               -7.1864 |             -17.1864 |     0.6000 |
| validation  |        5 |           0.6000 |       -110.7279 |             -120.7279 |            -130.7279 |     0.6000 |
| holdout     |        0 |         nan      |        nan      |              nan      |             nan      |   nan      |

The option surface is used only as a causal signal; BTC perpetual is the
traded leg at 10/20 bp total cost. This adaptive short-history study changes
no PaperLive, leverage, remote, or real-order permission.

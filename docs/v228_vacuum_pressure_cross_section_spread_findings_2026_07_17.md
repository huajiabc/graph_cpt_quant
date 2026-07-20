# v22.8 Vacuum-Pressure Cross-Section Spread Findings

Verdict: `reject_vacuum_pressure_cross_section_spread`.

| candidate                               |   events |   active_days |   active_months |   development_events |   validation_events |   holdout_events |   mean_gross_1h_bp |   mean_gross_4h_bp |   mean_gross_8h_bp |   mean_primary_net_4h_bp |   mean_stress_net_4h_bp |   development_primary_net_4h_bp |   validation_primary_net_4h_bp |   holdout_primary_net_4h_bp |   raw_dollar_neutral_gross_4h_bp |   raw_dollar_neutral_net_4h_bp |   long_alt_component_4h_bp |   short_alt_component_4h_bp |   btc_hedge_component_4h_bp |   bootstrap_95_low_primary_bp |   bootstrap_95_high_primary_bp |   random_rank_percentile |   reversed_primary_net_4h_bp |   delayed_primary_net_4h_bp |   positive_month_concentration |   positive_day_concentration |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:----------------------------------------|---------:|--------------:|----------------:|---------------------:|--------------------:|-----------------:|-------------------:|-------------------:|-------------------:|-------------------------:|------------------------:|--------------------------------:|-------------------------------:|----------------------------:|---------------------------------:|-------------------------------:|---------------------------:|----------------------------:|----------------------------:|------------------------------:|-------------------------------:|-------------------------:|-----------------------------:|----------------------------:|-------------------------------:|-----------------------------:|----------------------------:|---------------------------:|:----------|
| DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4 |      159 |           120 |              11 |                   63 |                  47 |               49 |            -2.4274 |            -7.6325 |           -10.3294 |                 -37.6325 |                -47.6325 |                        -35.0619 |                       -41.5721 |                    -37.1587 |                          -6.1170 |                       -26.1170 |                   -10.0954 |                      4.3288 |                     -1.8660 |                      -45.0871 |                       -30.3565 |                   1.7000 |                     -22.3675 |                    -34.1561 |                            inf |                       0.2402 |                      0.0000 |                     0.0000 | False     |

## Chronological periods

| period      |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |
|:------------|---------:|----------------:|----------------------:|---------------------:|
| development |       63 |         -5.0619 |              -35.0619 |             -45.0619 |
| holdout     |       49 |         -7.1587 |              -37.1587 |             -47.1587 |
| validation  |       47 |        -11.5721 |              -41.5721 |             -51.5721 |

## Holding horizons

|   horizon_hours |   mean_gross_bp |
|----------------:|----------------:|
|          1.0000 |         -2.4274 |
|          4.0000 |         -7.6325 |
|          8.0000 |        -10.3294 |

Only the preregistered Top4-minus-Bottom4 rank spread was evaluated.
The one/eight-hour and raw-dollar-neutral views cannot rescue the primary.

No live, PaperLive, leverage, remote, application, or order state changed.

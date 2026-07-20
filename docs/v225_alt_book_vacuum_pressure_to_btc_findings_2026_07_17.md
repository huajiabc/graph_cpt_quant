# v22.5 Alt-Book Vacuum Pressure to BTC Findings

Verdict: `reject_alt_book_vacuum_pressure_to_btc`.

| candidate                            |   events |   active_days |   active_months |   development_events |   validation_events |   holdout_events |   minimum_direction_period_events |   mean_btc_gross_1h_bp |   mean_btc_primary_net_1h_bp |   mean_btc_gross_4h_bp |   mean_btc_primary_net_4h_bp |   mean_btc_stress_net_4h_bp |   development_primary_net_4h_bp |   validation_primary_net_4h_bp |   holdout_primary_net_4h_bp |   long_primary_net_4h_bp |   short_primary_net_4h_bp |   mean_alt_bucket_gross_4h_bp |   mean_alt_bucket_primary_net_4h_bp |   mean_variance_ratio |   validation_variance_ratio |   holdout_variance_ratio |   bootstrap_95_low_primary_bp |   bootstrap_95_high_primary_bp |   random_time_percentile |   reversed_primary_net_4h_bp |   delayed_primary_net_4h_bp |   no_vacuum_primary_net_4h_bp |   positive_month_concentration |   positive_day_concentration | promote   |
|:-------------------------------------|---------:|--------------:|----------------:|---------------------:|--------------------:|-----------------:|----------------------------------:|-----------------------:|-----------------------------:|-----------------------:|-----------------------------:|----------------------------:|--------------------------------:|-------------------------------:|----------------------------:|-------------------------:|--------------------------:|------------------------------:|------------------------------------:|----------------------:|----------------------------:|-------------------------:|------------------------------:|-------------------------------:|-------------------------:|-----------------------------:|----------------------------:|------------------------------:|-------------------------------:|-----------------------------:|:----------|
| DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC |      159 |           120 |              11 |                   63 |                  47 |               49 |                                15 |               -11.8033 |                     -21.8033 |                 2.8179 |                      -7.1821 |                    -17.1821 |                          6.5438 |                       -15.3911 |                    -16.9556 |                 -14.6820 |                   -3.4321 |                       -3.3717 |                            -23.3717 |                1.7711 |                      2.3379 |                   1.4663 |                      -22.0358 |                         9.8940 |                  39.0000 |                     -12.8179 |                     -1.0746 |                       -5.2808 |                         0.6597 |                       0.0893 | False     |

## Chronological periods

| period      |   events |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   variance_ratio |
|:------------|---------:|----------------:|----------------------:|---------------------:|-----------------:|
| development |       63 |         16.5438 |                6.5438 |              -3.4562 |           1.5854 |
| holdout     |       49 |         -6.9556 |              -16.9556 |             -26.9556 |           1.4663 |
| validation  |       47 |         -5.3911 |              -15.3911 |             -25.3911 |           2.3379 |

## Signal direction

|   signal_direction |   events |   mean_gross_bp |   mean_primary_net_bp |   variance_ratio |
|-------------------:|---------:|----------------:|----------------------:|-----------------:|
|            -1.0000 | 106.0000 |          6.5679 |               -3.4321 |           1.5667 |
|             1.0000 |  53.0000 |         -4.6820 |              -14.6820 |           2.1801 |

The single preregistered four-hour BTC endpoint was evaluated.
The one-hour and alt-bucket outcomes are secondary and cannot rescue it.

No live, PaperLive, leverage, remote, application, or order state changed.

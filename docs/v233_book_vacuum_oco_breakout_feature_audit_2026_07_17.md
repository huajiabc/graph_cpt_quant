# v23.3 Book-Vacuum OCO Breakout Feature Audit

Verdict: `feature_viable_freeze_one_sigma_oco_breakout`.

| candidate                         | scope       |   events |   active_months |   long_pressure_events |   short_pressure_events |   median_hourly_sigma_bp |   median_barrier_distance_bp |
|:----------------------------------|:------------|---------:|----------------:|-----------------------:|------------------------:|-------------------------:|-----------------------------:|
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | all         |      159 |              11 |                     53 |                     106 |                  43.4435 |                      43.4435 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | development |       63 |               4 |                     20 |                      43 |                  47.9335 |                      47.9335 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | validation  |       47 |               3 |                     15 |                      32 |                  46.2107 |                      46.2107 |
| DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT | holdout     |       49 |               4 |                     18 |                      31 |                  39.8864 |                      39.8864 |

At each frozen v22.4 event, the completed-hour BTC close and the
trailing 24 completed hourly log moves define a causal one-hour
sigma. Symmetric stops are frozen at plus/minus one sigma. All 16
subsequent 15-minute timestamps must exist, but their highs, lows,
trigger states, direction, fills, and returns were not loaded.

No live, PaperLive, leverage, remote, application, or order state changed.

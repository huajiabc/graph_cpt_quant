# v23.11 Positive-q80 Vacuum Breakout Feature Audit

Verdict: `feature_viable_freeze_positive_q80_0625sigma_breakout`.

| candidate                                   | scope       |   events |   active_months |   median_pressure_ratio |   median_withdrawal_breadth |   median_barrier_distance_bp |
|:--------------------------------------------|:------------|---------:|----------------:|------------------------:|----------------------------:|-----------------------------:|
| DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT | all         |       89 |              12 |                  1.3445 |                      0.4375 |                      24.9915 |
| DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT | development |       32 |               5 |                  1.3617 |                      0.4375 |                      20.0967 |
| DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT | validation  |       24 |               3 |                  1.4595 |                      0.3750 |                      32.1090 |
| DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT | holdout     |       33 |               4 |                  1.2669 |                      0.4375 |                      25.7213 |

This denser mechanism-preserving candidate keeps positive bucket
pressure, 11/16 directional agreement, 5/16 depth withdrawal,
false transitions, and four-hour cooldown. Only the causal pressure
threshold changes from q90 to the predeclared q80. BTC OCO barriers
are frozen at plus/minus 0.625 trailing hourly sigma.

No post-entry high, low, trigger, fill, direction, or return was used.
No live, PaperLive, leverage, remote, application, or order state changed.

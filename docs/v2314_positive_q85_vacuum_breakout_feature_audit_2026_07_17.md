# v23.14 Positive-q85 Vacuum Breakout Feature Audit

Verdict: `feature_viable_freeze_single_q85_interpolation`.

| candidate                                   | scope       |   events |   active_months |   median_pressure_ratio |   median_withdrawal_breadth |   median_barrier_distance_bp |
|:--------------------------------------------|:------------|---------:|----------------:|------------------------:|----------------------------:|-----------------------------:|
| DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT | all         |       75 |              12 |                  1.2963 |                      0.4375 |                      25.5071 |
| DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT | development |       28 |               5 |                  1.2246 |                      0.4375 |                      20.6710 |
| DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT | validation  |       21 |               3 |                  1.3112 |                      0.3750 |                      31.8720 |
| DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT | holdout     |       26 |               4 |                  1.3149 |                      0.4062 |                      26.0329 |

q85 is the sole predeclared interpolation between the rejected
q80 density extension and the post-selected q90 tail candidate.
All other breadth, withdrawal, transition, cooldown, BTC sigma,
barrier, and path-coverage rules are unchanged.

No post-entry trigger, fill, high/low path, or return was used.
No live, PaperLive, leverage, remote, application, or order state changed.

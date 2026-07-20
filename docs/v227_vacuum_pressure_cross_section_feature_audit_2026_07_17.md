# v22.7 Vacuum-Pressure Cross-Section Feature Audit

Verdict: `feature_viable_freeze_vacuum_pressure_spread`.

| candidate                               | scope       |   events |   active_months |   mean_score_gap |   long_withdrawal_rate |   short_withdrawal_rate |
|:----------------------------------------|:------------|---------:|----------------:|-----------------:|-----------------------:|------------------------:|
| DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4 | all         |      159 |              11 |           5.2943 |                 0.5362 |                  0.5094 |
| DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4 | development |       63 |               4 |           9.6438 |                 0.6032 |                  0.5595 |
| DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4 | validation  |       47 |               3 |           2.4017 |                 0.4574 |                  0.4894 |
| DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4 | holdout     |       49 |               4 |           2.4768 |                 0.5255 |                  0.4643 |

At each frozen v22.4 broad book-vacuum event, the sole candidate
is long the four highest one-percent imbalance z-scores and short
the four lowest, with 0.5 raw notional per side. No depth severity
multiplier, threshold grid, or outcome-conditioned rank was used.

No future price, return, PnL, beta, turnover, or outcome was loaded.

No live, PaperLive, leverage, remote, application, or order state changed.

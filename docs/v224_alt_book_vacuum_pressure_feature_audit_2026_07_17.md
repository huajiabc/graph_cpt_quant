# v22.4 Alt-Book Vacuum Pressure Feature Audit

Verdict: `feature_viable_freeze_alt_book_vacuum_pressure`.

| candidate                            | scope       |   events |   active_months |   long_events |   short_events |   median_abs_pressure |   mean_directional_breadth |   mean_withdrawal_breadth |
|:-------------------------------------|:------------|---------:|----------------:|--------------:|---------------:|----------------------:|---------------------------:|--------------------------:|
| DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC | all         |      159 |              11 |            53 |            106 |                0.7939 |                     0.8066 |                    0.5266 |
| DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC | development |       63 |               4 |            20 |             43 |                0.8069 |                     0.8283 |                    0.5601 |
| DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC | validation  |       47 |               3 |            15 |             32 |                0.8026 |                     0.7899 |                    0.5027 |
| DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC | holdout     |       49 |               4 |            18 |             31 |                0.7639 |                     0.7946 |                    0.5064 |

The feature standardizes each symbol's completed-hour one-percent
book imbalance against shifted trailing-720-hour history. Candidate
events require an aggregate absolute-pressure q90 breach, at least
11/16 symbols aligned with the direction, at least 5/16 symbols in
their own trailing q20 depth-withdrawal state, a false transition,
and a four-hour cooldown.

No future price, return, PnL, turnover, or outcome was loaded.

No live, PaperLive, leverage, remote, application, or order state changed.

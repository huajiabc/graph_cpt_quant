# v23.17 q90 Breakout + CM2 Feature Audit

Verdict: `feature_viable_freeze_q90_cm2_overlay`.

Feature hash: `CF79F9D42324BF5C930292F89D0186D86845B73561FA7E2B8EF6368E24C82046`.

| scope       |   events |   active_weeks |   calendar_weeks |   active_months |   mean_events_per_active_week |   maximum_events_per_week |
|:------------|---------:|---------------:|-----------------:|----------------:|------------------------------:|--------------------------:|
| all         |       53 |             22 |               49 |              11 |                        2.4091 |                         6 |
| development |       20 |              7 |               22 |               4 |                        2.8571 |                         5 |
| validation  |       15 |              8 |               13 |               3 |                        1.8750 |                         5 |
| holdout     |       18 |              7 |               14 |               4 |                        2.5714 |                         6 |

The frozen positive-q90 event times were mapped into the existing
49-week CM2 calendar using only calendar fields. Returns are assigned
to the week in which the four-hour event exits; signal and realization
weeks are both retained when an event crosses Monday 00:00 UTC.
The primary overlay
weight is fixed at 10%; 5% and 20% are sensitivity scales only.
No trigger, fill, BTC path, or return column was loaded.

No live, PaperLive, leverage, remote, application, or order state changed.

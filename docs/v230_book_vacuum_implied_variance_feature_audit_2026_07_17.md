# v23.0 Book-Vacuum Causal Implied-Variance Feature Audit

Verdict: `feature_viable_freeze_causal_implied_variance_test`.

| candidate                                | scope       |   events |   active_months |   median_surface_age_hours |   median_surface_dte |   median_causal_atm_iv |   median_prior_24h_squared_move |
|:-----------------------------------------|:------------|---------:|----------------:|---------------------------:|---------------------:|-----------------------:|--------------------------------:|
| DVB2_BOOK_VACUUM_CAUSAL_IMPLIED_VARIANCE | all         |      123 |              10 |                  15.000000 |            26.929167 |               0.392811 |                        0.000446 |
| DVB2_BOOK_VACUUM_CAUSAL_IMPLIED_VARIANCE | development |       46 |               4 |                  15.000000 |            26.463104 |               0.369198 |                        0.000504 |
| DVB2_BOOK_VACUUM_CAUSAL_IMPLIED_VARIANCE | validation  |       44 |               3 |                  15.000000 |            26.714813 |               0.482164 |                        0.000535 |
| DVB2_BOOK_VACUUM_CAUSAL_IMPLIED_VARIANCE | holdout     |       33 |               3 |                  18.000000 |            28.923057 |               0.378760 |                        0.000328 |

Each frozen v22.4 event uses only the latest completed Deribit
daily trade surface known at entry, capped at 72 hours of age.
Within the causal surface timestamp, the quality-passing 7--45 DTE
row closest to 21 DTE supplies ATM IV. Entry BTC and the trailing
24-hour sum of squared log moves are also known at the event time.

No post-entry spot, option value, return, PnL, or event outcome was
loaded. Historical option trade bars remain signal-only because no
historical bid/ask archive is available.

No live, PaperLive, leverage, remote, application, or order state changed.

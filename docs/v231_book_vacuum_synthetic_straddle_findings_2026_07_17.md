# v23.1 Book-Vacuum Synthetic Straddle Findings

Verdict: `movement_sufficiency_rejected`.

| candidate                               | scope             |   events |   active_months |   mean_gross_premium_return_1h_bp |   mean_gross_premium_return_4h_bp |   mean_primary_net_premium_return_4h_bp |   mean_stress_net_premium_return_4h_bp |   mean_gross_premium_return_8h_bp |   mean_realized_to_implied_variance_4h |   median_realized_to_implied_variance_4h |   mean_absolute_log_move_4h_bp |
|:----------------------------------------|:------------------|---------:|----------------:|----------------------------------:|----------------------------------:|----------------------------------------:|---------------------------------------:|----------------------------------:|---------------------------------------:|-----------------------------------------:|-------------------------------:|
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | all               |      123 |              10 |                           13.0172 |                           20.3532 |                                -79.6468 |                              -179.6468 |                           37.6035 |                                 1.4433 |                                   0.6263 |                        81.6498 |
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | development       |       46 |               4 |                            6.5278 |                           37.5936 |                                -62.4064 |                              -162.4064 |                           53.0418 |                                 1.4264 |                                   0.7439 |                        81.6529 |
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | validation        |       44 |               3 |                           20.6791 |                           31.0889 |                                -68.9111 |                              -168.9111 |                           57.2497 |                                 1.8438 |                                   0.7307 |                       103.9741 |
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | holdout           |       33 |               3 |                           11.8470 |                          -17.9930 |                               -117.9930 |                              -217.9930 |                          -10.1114 |                                 0.9329 |                                   0.3928 |                        51.8795 |
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | positive_pressure |       39 |               9 |                           -1.8856 |                           14.2018 |                                -85.7982 |                              -185.7982 |                           42.4355 |                                 0.9591 |                                   0.5358 |                        81.9545 |
| DVB2_BOOK_VACUUM_SYNTHETIC_ATM_STRADDLE | negative_pressure |       84 |              10 |                           19.9363 |                           23.2093 |                                -76.7907 |                              -176.7907 |                           35.3601 |                                 1.6681 |                                   0.6309 |                        81.5083 |

Matched-event coverage: 123/123.
Matched random-time percentile: 100.00.
Month-block bootstrap 2.5% lower bound: -103.5179 bp.

This is a constant-IV synthetic movement-sufficiency test, not
historical executable option PnL. The local Deribit archive has
trade OHLCV but no synchronized historical bid/ask, and the exact
preselected two-leg 4-hour trade coverage is too sparse for promotion.

No live, PaperLive, leverage, remote, application, or order state changed.

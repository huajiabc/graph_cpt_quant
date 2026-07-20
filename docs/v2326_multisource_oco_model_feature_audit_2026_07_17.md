# v23.26 Multisource OCO Model Feature Audit

Verdict: `feature_viable_freeze_multisource_model`.

Feature hash: `4B93B3F7A5D340776CF0CDAA5E16C37AFACF6A20AFC93ED25C74DC2BB393B081`.

| scope       |   events |   active_months |   minimum_alt_metric_symbols |   positive_pressure_fraction |   median_pressure_excess |   median_alt_taker_buy_breadth |   median_alt_oi_build_breadth |   median_causal_sigma_bp |
|:------------|---------:|----------------:|-----------------------------:|-----------------------------:|-------------------------:|-------------------------------:|------------------------------:|-------------------------:|
| all         |      159 |              11 |                           15 |                       0.3333 |                   0.2669 |                         0.5000 |                        0.5000 |                  43.4435 |
| development |       63 |               4 |                           16 |                       0.3175 |                   0.4029 |                         0.5000 |                        0.5000 |                  47.9335 |
| validation  |       47 |               3 |                           15 |                       0.3191 |                   0.1654 |                         0.5000 |                        0.5625 |                  46.2107 |
| holdout     |       49 |               4 |                           16 |                       0.3673 |                   0.1969 |                         0.5000 |                        0.5000 |                  39.8864 |

The 19 fixed features combine book state, causal BTC volatility and
return, same-timestamp 15-of-16-or-better taker/OI/top-position
state, BTC derivatives state, and UTC-hour cyclic terms. One
validation event uses 15 symbols because XLM lacks that exact 5-minute
record; no future or stale fill is used. No OCO return was loaded.

No live, PaperLive, leverage, remote, application, or order state changed.

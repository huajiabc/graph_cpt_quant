# v23.23 q90 Broad-Taker Confirmation Feature Audit

Verdict: `feature_viable_freeze_broad_taker_confirmation`.

Feature hash: `7C33473808AC4C2CFE3DBB81FD71642C2D6C3814D405D9925EE1D495B6A1C7DD`.

| scope       |   events |   active_months |   median_taker_buy_symbols |   median_taker_buy_breadth |   median_log_taker_ratio |   median_bucket_pressure |
|:------------|---------:|----------------:|---------------------------:|---------------------------:|-------------------------:|-------------------------:|
| all         |       26 |              10 |                    10.0000 |                     0.6250 |                   0.1971 |                   0.8639 |
| development |        8 |               4 |                    10.5000 |                     0.6562 |                   0.2506 |                   0.8601 |
| validation  |        7 |               2 |                    10.0000 |                     0.6250 |                   0.2064 |                   0.8529 |
| holdout     |       11 |               4 |                    10.0000 |                     0.6250 |                   0.1396 |                   0.8749 |

Every q90 event is joined to the exact completed Binance metrics
timestamp. Selection requires at least 9 of 16 taker long/short
volume ratios above one. No BTC exit price or return was loaded.

This is a second-stage diagnostic of a post-selected ancestor.
No live, PaperLive, leverage, remote, application, or order state changed.

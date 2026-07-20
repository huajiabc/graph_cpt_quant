# v23.29 Event Volatility-Transmission Feature Audit

Verdict: `feature_viable_freeze_direct_volatility_selector`.

Feature hash: `C7EFC21FA0B9FEC822BE86D4C7A986C352E6B3F2C5382E0EF6256CBCCD312FFF`.

| scope       |   events |   active_months |   minimum_history_hours |   median_alt_abs_z |   median_alt_btc_abs_z_gap |   median_rv_acceleration |   median_directed_edge_fraction |   median_btc_receiver_gap |
|:------------|---------:|----------------:|------------------------:|-------------------:|---------------------------:|-------------------------:|--------------------------------:|--------------------------:|
| all         |      159 |              11 |                     720 |             0.6984 |                    -0.0932 |                   0.9245 |                          0.1875 |                   -0.0656 |
| development |       63 |               4 |                     720 |             0.9414 |                    -0.0366 |                   1.0048 |                          0.1250 |                   -0.1878 |
| validation  |       47 |               3 |                     720 |             0.6526 |                    -0.1886 |                   0.8766 |                          0.2500 |                   -0.0401 |
| holdout     |       49 |               4 |                     720 |             0.6944 |                    -0.1030 |                   0.9696 |                          0.2500 |                    0.0508 |

The 17 features use the current completed hourly return and strictly
prior 30-day normalization/lead-lag history. No payoff was loaded.

No PaperLive, leverage, remote, application, or order state changed.

# v23.20 Alt-First Volatility Ignition Feature Audit

Verdict: `feature_viable_freeze_alt_first_ignition`.

Feature hash: `C4F814ADD57330518B98C6ABFA2CCC98A7A4BC7EC3814D2B8DB7F9118A478B6F`.

| scope       |   events |   active_months |   median_shocked_symbols |   median_alt_bucket_shock_z |   median_btc_abs_move_z |   median_barrier_width_bp |
|:------------|---------:|----------------:|-------------------------:|----------------------------:|------------------------:|--------------------------:|
| all         |      100 |              12 |                  10.0000 |                      1.1091 |                  0.2457 |                   28.0347 |
| development |       58 |               6 |                  10.0000 |                      1.1572 |                  0.2343 |                   26.9471 |
| validation  |       22 |               3 |                   9.0000 |                      1.0663 |                  0.2560 |                   33.8904 |
| holdout     |       20 |               3 |                   9.0000 |                      1.1161 |                  0.3335 |                   28.2258 |

The signal uses only completed hourly prices and prior rolling
normalization windows. It captures broad alt volatility while BTC
remains below its own causal median shock. No post-entry price or
return outcome was used to select the 100 events.

No live, PaperLive, leverage, remote, application, or order state changed.

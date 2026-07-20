# v20.2 Community Peer-Hedge Feature-Only Audit

| candidate                                 |   original_feature_events |   constructible_events |   excluded_events |   constructible_fraction |   development_events |   validation_events |   holdout_events |   active_days |   active_months |   median_selected_count |   median_peer_count |   minimum_peer_count |   median_abs_prior_btc_beta_exposure |   p90_abs_prior_btc_beta_exposure |   max_abs_net_dollar_exposure |   max_gross_notional_drift | feature_viable   |
|:------------------------------------------|--------------------------:|-----------------------:|------------------:|-------------------------:|---------------------:|--------------------:|-----------------:|--------------:|----------------:|------------------------:|--------------------:|---------------------:|-------------------------------------:|----------------------------------:|------------------------------:|---------------------------:|:-----------------|
| RPH1_COMMUNITY_TRADE_OVERSHOOT_PEER_HEDGE |                       360 |                    237 |               123 |                   0.6583 |                  108 |                  59 |               70 |           100 |              11 |                  3.0000 |              3.0000 |                    2 |                               0.0483 |                            0.1447 |                        0.0000 |                     0.0000 | True             |

Excluded-event reasons:

| reason                          |   events |
|:--------------------------------|---------:|
| fewer_than_two_unselected_peers |      123 |

The construction reuses only the frozen RPT4 community trade-overshoot events. Selected over-shooting receivers retain the preregistered fade direction; every other available member of the same frozen community forms the opposite peer sleeve. Each sleeve has 0.5 absolute notional, so the book is exactly dollar neutral without introducing a BTC leg.

Prior BTC beta is reported as a risk diagnostic, not neutralized. No future return or candidate PnL was calculated or inspected in this audit. The branch is explicitly posthoc because v20.1 sleeve attribution motivated the hedge redesign.

No live, PaperLive, application, leverage, remote, or order state changed.

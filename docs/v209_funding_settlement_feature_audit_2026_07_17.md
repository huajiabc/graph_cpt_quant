# v20.9 Funding-Settlement Feature Audit

Verdict: `feature_viable_freeze_two_settlement_candidates`.

| candidate                            | scope   |   events |   active_days |   active_months |   mean_selection_count |   median_selection_count |   mean_selected_funding_bp |
|:-------------------------------------|:--------|---------:|--------------:|----------------:|-----------------------:|-------------------------:|---------------------------:|
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND | all     |      813 |           294 |              11 |                18.3383 |                  18.0000 |                    -1.5779 |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      | all     |      505 |           262 |              11 |                 8.9703 |                   8.0000 |                    -0.7771 |

Frozen feature rules:

- FSE1: at a synchronized 00/08/16 UTC Binance USD-M funding settlement, select every alt with a just-settled negative funding rate; require at least five names.
- FSE2: select only alts whose just-settled rate is negative while their immediately prior settled rate was non-negative; require at least five names.

The signal is observed at settlement. The frozen entry is one full 15-minute bar later and the primary holding window is 60 minutes. No post-event return was calculated during this feature audit.

No live, PaperLive, application, leverage, remote, or order state was read or changed.

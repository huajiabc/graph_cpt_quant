# v21.5 DEX Community Relative-Spread Feature Audit

Verdict: `feature_viable_freeze_relative_spread_diagnostic`.

| candidate                                  | scope             |   events |   active_days |   active_months |   source_symbols |   communities |   mean_leg_size |   median_feature_rank_gap_bp |
|:-------------------------------------------|:------------------|---------:|--------------:|----------------:|-----------------:|--------------:|----------------:|-----------------------------:|
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | all               |      274 |           151 |              10 |               15 |            32 |          2.7409 |                      67.3703 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | development       |      203 |            95 |               4 |               12 |            20 |          2.6847 |                      66.8815 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | source_transition |        4 |             4 |               2 |                3 |             2 |          3.0000 |                      99.4536 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | validation        |       31 |            28 |               2 |                3 |             4 |          2.8065 |                      66.8689 |
| DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD | holdout           |       36 |            24 |               2 |                7 |             6 |          2.9722 |                      74.0853 |

At each already frozen v21.2 source event, peers are ranked using only their one-hour returns observed by the feature close. The slowest and fastest equal-sized halves form disjoint laggard and leader legs; an odd middle name is discarded. Each leg requires at least two names.

This candidate is explicitly second-stage and was motivated by the v21.3 reveal. Same-history results can assess economic magnitude but cannot provide independent promotion evidence.

Frozen v21.2 event-feature SHA256: `C17C550BB99EA22A74188FC01C00B50E5EB710029F5D73024ABAB1D4773DD23C`.

No post-event return was calculated or inspected in this feature audit. No live, PaperLive, application, leverage, remote, or order state changed.

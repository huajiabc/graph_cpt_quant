# v21.2 DEX Community-Propagation Feature Audit

Verdict: `feature_viable_freeze_dex_community_propagation`.

After 24-hour same-source de-overlap, 1,687 DEX attention events were available; 786 mapped to a causal 15-minute source feature and a forward-frozen monthly graph community.

| candidate                                | scope   |   events |   active_days |   active_months |   source_symbols |   communities |   mean_selection_count |   median_abs_source_z |
|:-----------------------------------------|:--------|---------:|--------------:|----------------:|-----------------:|--------------:|-----------------------:|----------------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | all     |      274 |           151 |              10 |               15 |            32 |                 5.9489 |                1.8219 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | all     |      274 |           151 |              10 |               15 |            32 |                 3.2883 |                1.8219 |

Frozen feature candidates:

- DAP1: a DEX volume-attention event is followed by an absolute CEX source return innovation of at least 1.0 prior sigma; after a four-hour community cooldown, select every other available member of the source's monthly graph community, requiring at least four peers.
- DAP2: apply the same source rule, rank peers by their observed one-hour return in the source direction, and select the slowest half (at least three names). This is a relative laggard rule, not an outcome filter.
- DEX data are available at the recorded event availability time. The source feature is the first 15-minute close strictly afterward; proposed execution waits one additional complete 15-minute bar.
- Source-return normalization uses only the preceding 20-30 days. No post-event portfolio return was calculated or inspected in this audit.
- Coverage-only chronology freezes Aug-Nov 2025 as development, Dec 2025-Feb 2026 as a visible but excluded vendor-transition interval, Mar-Apr 2026 as validation, and May 2026 onward as holdout.

Vendor/period coverage:

| candidate                                | period            | source_vendor            |   events |   source_symbols |   active_months |
|:-----------------------------------------|:------------------|:-------------------------|---------:|-----------------:|----------------:|
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | development       | dexpaprika_pool_ohlcv_1h |      203 |               12 |               4 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | holdout           | geckoterminal_pool_ohlcv |       36 |                7 |               2 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | source_transition | dexpaprika_pool_ohlcv_1h |        1 |                1 |               1 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | source_transition | geckoterminal_pool_ohlcv |        3 |                2 |               1 |
| DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION | validation        | geckoterminal_pool_ohlcv |       31 |                3 |               2 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | development       | dexpaprika_pool_ohlcv_1h |      203 |               12 |               4 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | holdout           | geckoterminal_pool_ohlcv |       36 |                7 |               2 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | source_transition | dexpaprika_pool_ohlcv_1h |        1 |                1 |               1 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | source_transition | geckoterminal_pool_ohlcv |        3 |                2 |               1 |
| DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS    | validation        | geckoterminal_pool_ohlcv |       31 |                3 |               2 |

Input hashes:

| input                                                                                 | sha256                                                           |
|:--------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| reports\v6_3_token_pool_dex_attention\token_pool_attention_events.csv                 | 261EAF0C11D514B71647C0D88F6ABA6810B04DE98475D2FFFBFC04BBFE133A4D |
| reports\v13_2_tg1_forward_temporal_extension\monthly_balanced_membership_extended.csv | 8B3B75C2E03F54635E6655583FEF482F3DB76798334D380517A91154622BBBA0 |

No live, PaperLive, application, leverage, remote, or order state was read or changed.

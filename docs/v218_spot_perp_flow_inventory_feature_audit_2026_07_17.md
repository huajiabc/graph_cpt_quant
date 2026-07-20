# v21.8 Spot-Perpetual Flow-Inventory Feature Audit

Verdict: `feature_viable_freeze_spot_perp_flow_inventory`.

The exact spot/perpetual/member intersection contains 61 symbols. The audit built 33,794 symbol-hour feature rows at 00/12 UTC using only then-available data.

| candidate                                | scope   |   events |   active_days |   active_months |   mean_eligible_symbols |   mean_long_count |   mean_short_count |   mean_score_spread |
|:-----------------------------------------|:--------|---------:|--------------:|----------------:|------------------------:|------------------:|-------------------:|--------------------:|
| SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE    | all     |      362 |           239 |              10 |                 48.9392 |            7.3757 |             7.2983 |              3.5669 |
| SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE | all     |      396 |           240 |              10 |                 50.6995 |            5.7298 |             5.7298 |              3.2315 |

Frozen feature candidates:

- SFI1 ranks the causally standardized spot-minus-perpetual taker imbalance gap globally, taking up to eight scores above +1 and eight below -1, with at least five names per leg.
- SFI2 takes the highest and lowest gap within each forward-frozen monthly graph community when both signs and a 1.5-sigma within-community gap are present; at least four community pairs are required.
- Both spot and perpetual hourly quote activity must be at least half their strictly prior seven-day median. Flow z-scores use only the prior 20-30 days.
- The feature is observed at the hourly close and proposed execution waits one additional complete hour. No post-event return was calculated.

Input manifest hashes:

| input                                                                                 | sha256                                                           |
|:--------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| data\external\binance_spot_1h\manifest.csv                                            | DE9FAE9F5C7BD7494E6232CFF58AF3CD2219D246E4D08A5A63E4382ECB875598 |
| data\external\binance_um_carry\manifest.csv                                           | 7145F2B53B0FE31306E8D20504CD3A7AFFECBDB38CD1105D083616CB066E6B25 |
| reports\v13_2_tg1_forward_temporal_extension\monthly_balanced_membership_extended.csv | 8B3B75C2E03F54635E6655583FEF482F3DB76798334D380517A91154622BBBA0 |

No live, PaperLive, application, leverage, remote, or order state was read or changed.

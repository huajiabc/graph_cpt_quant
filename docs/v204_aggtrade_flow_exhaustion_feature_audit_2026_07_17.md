# v20.4 AggTrade Flow-Exhaustion Feature Audit

Verdict: `feature_viable_freeze_two_candidates`.

All 936 frozen receiver windows are present across 216 events and 45 symbols. Every window passed the frozen trade-count and boundary-coverage quality gate.

The aggTrade first/last return agrees with the official 15-minute kline: correlation=0.99999823, p99 absolute difference=1.7204 bp.

| candidate                                | scope   |   events |   active_days |   active_months |   mean_candidate_receivers |   mean_late_opposes_fraction |   mean_strict_exhausted_count |
|:-----------------------------------------|:--------|---------:|--------------:|----------------:|---------------------------:|-----------------------------:|------------------------------:|
| RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE  | all     |       53 |            28 |              11 |                     3.8679 |                       0.7363 |                        2.0000 |
| RFX2_EXHAUSTED_VS_PERSISTENT_FLOW_SPREAD | all     |       52 |            22 |              10 |                     4.3654 |                       0.3630 |                        1.5962 |

Frozen feature rules:

- RFX1: at least half of the original extreme overshoot receivers have late-third taker flow opposite to the graph source direction; fade the full quality-screened receiver bucket.
- RFX2: at least one receiver flips from early-third source-aligned flow to late-third opposing flow and at least two retain source-aligned flow; test the exhausted-versus-persistent within-event spread.

The thresholds are structural zero/sign rules and were frozen without reading post-event returns. This is post-hoc offline discovery and any positive reveal remains natural-forward-only, not promotion evidence.

No future price, candidate PnL, live, PaperLive, application, leverage, remote, or order state was read or changed.

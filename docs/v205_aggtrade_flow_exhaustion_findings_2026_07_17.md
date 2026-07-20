# v20.5 AggTrade Flow-Exhaustion Findings

Verdict: `reject_aggtrade_flow_exhaustion_candidates`.

| candidate                                |   events |   mean_gross_bp |   mean_primary_net_bp |   break_even_cost_bp |   random_control_percentile |   bootstrap_lower_95_primary_net_bp | eligible   | status   |
|:-----------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------------:|------------------------------------:|:-----------|:---------|
| RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE  |       53 |          8.9569 |              -11.0431 |               8.9569 |                      0.9580 |                            -21.6210 | False      | rejected |
| RFX2_EXHAUSTED_VS_PERSISTENT_FLOW_SPREAD |       52 |         -1.8519 |              -21.8519 |              -1.8519 |                      0.2460 |                            -26.0027 | False      | rejected |

| candidate                                | scope   |   events |   active_days |   active_months |   mean_receivers |   mean_all_alt_bp |   mean_exhausted_bp |   mean_persistent_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |
|:-----------------------------------------|:--------|---------:|--------------:|----------------:|-----------------:|------------------:|--------------------:|---------------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|
| RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE  | all     |       53 |            28 |              11 |           3.8679 |           18.2166 |            nan      |             nan      |             -9.2597 |          8.9569 |              -11.0431 |             -31.0431 |                       -28.9569 |                      0.1509 |
| RFX2_EXHAUSTED_VS_PERSISTENT_FLOW_SPREAD | all     |       52 |            22 |              10 |           4.3654 |           -1.0603 |             12.3319 |             -13.3922 |             -0.7916 |         -1.8519 |              -21.8519 |             -41.8519 |                       -18.1481 |                      0.0769 |

The reveal follows the frozen v20.5 preregistration. The primary and stress results charge 20/40 bp round-trip book costs. A positive result would remain post-hoc and natural-forward-only.

No live, PaperLive, application, leverage, remote, or order state changed.

# v15.7 Pair-Shock to Fragile-Receiver Findings

Verdict: `reject_candidate`.

| candidate                          |   calendar_days |   active_days |   active_validation_days |   active_holdout_days |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   development_primary_net_bp |   validation_primary_net_bp |   holdout_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_pairing_percentile |   positive_month_concentration |   mean_calendar_turnover |   reversed_control_mean_bp |   stale_control_mean_bp |   source_only_control_mean_bp |   max_abs_residual_btc_beta |   max_gross_notional_drift | promote   |
|:-----------------------------------|----------------:|--------------:|-------------------------:|----------------------:|----------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|-------------------------:|----------------------:|-----------------------:|----------------------------:|-------------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------------:|----------------------------:|---------------------------:|:----------|
| VT4_PAIR_SHOCK_TO_FRAGILE_RECEIVER |             375 |           174 |                       39 |                    56 |          0.7435 |              -14.9247 |             -30.5929 |                     -17.6999 |                    -13.5329 |                 -11.2807 |              -22.0756 |                -7.8087 |                     60.8000 |                         1.0000 |                   0.7834 |                   -16.4118 |                -13.3260 |                      -12.6818 |                      0.0000 |                     0.0000 | False     |

## Frozen controls

| control                        |   mean_primary_net_bp |   active_days |
|:-------------------------------|----------------------:|--------------:|
| VT4_REVERSED_PROPAGATION       |              -16.4118 |           174 |
| VT4_ONE_DAY_STALE_SIGNAL       |              -13.3260 |           174 |
| VT4_SOURCE_SHOCK_ONLY_NO_DEPTH |              -12.6818 |           174 |

The graph pairs, source/receiver definition, one-percent fragility,
two-per-side holding band, cost model and gates were frozen before
inspecting propagation returns. PaperLive and remote state are unchanged.

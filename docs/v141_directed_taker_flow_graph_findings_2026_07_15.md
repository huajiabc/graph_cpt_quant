# v14.1 Directed Taker-Flow Graph Findings

Verdict: `reject_directed_flow_graph_family`.

The primary endpoint is one-hour BTC-residual return after 40 bp total cost. Four-hour values are diagnostic only.

## Candidate audit

| candidate                      | eligible   | verdict                              |   full_residual_net40 |   development_residual_net40 |   validation_residual_net40 |   holdout_residual_net40 |   full_raw_net20 |   full_raw_net30 |   delayed_residual_net40 |   reversed_residual_net40 |   random_family_percentile |   bootstrap_ci_low |   bootstrap_ci_high |   max_positive_month_share |   worst_period_mean | failed_gates                                                                                                                                                                                                                                                                                                                          | family_verdict                    |
|:-------------------------------|:-----------|:-------------------------------------|----------------------:|-----------------------------:|----------------------------:|-------------------------:|-----------------:|-----------------:|-------------------------:|--------------------------:|---------------------------:|-------------------:|--------------------:|---------------------------:|--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------|
| TFG1_POSITIVE_FLOW_PROPAGATION | False      | reject_directed_flow_graph_candidate |             -0.004064 |                    -0.005287 |                   -0.003402 |                -0.002887 |        -0.002690 |        -0.003690 |                -0.004417 |                 -0.003743 |                   0.220000 |          -0.005165 |           -0.003249 |                        inf |           -0.005287 | development_residual_net40_positive|validation_residual_net40_positive|holdout_residual_net40_positive|development_raw_net20_positive|validation_raw_net20_positive|holdout_raw_net20_positive|full_raw_net30_positive|bootstrap_lower_positive|random_family_p95|beats_reversed|month_share_below_35pct|worst_period_above_minus40bp | reject_directed_flow_graph_family |
| TFG2_NEGATIVE_FLOW_PROPAGATION | False      | reject_directed_flow_graph_candidate |             -0.003929 |                    -0.004042 |                   -0.003992 |                -0.003705 |        -0.002015 |        -0.003015 |                -0.004242 |                 -0.003726 |                   0.500000 |          -0.004274 |           -0.003566 |                        inf |           -0.004042 | development_residual_net40_positive|validation_residual_net40_positive|holdout_residual_net40_positive|development_raw_net20_positive|validation_raw_net20_positive|holdout_raw_net20_positive|full_raw_net30_positive|bootstrap_lower_positive|random_family_p95|beats_reversed|month_share_below_35pct|worst_period_above_minus40bp | reject_directed_flow_graph_family |

## Primary summary

| scope       | candidate                      |   portfolio_observations |   mean_raw_net_1h_20bp |   mean_raw_net_1h_30bp |   mean_residual_net_1h_40bp |   mean_residual_net_4h_40bp |
|:------------|:-------------------------------|-------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------------:|
| all         | TFG1_POSITIVE_FLOW_PROPAGATION |                      660 |              -0.002690 |              -0.003690 |                   -0.004064 |                   -0.003764 |
| all         | TFG2_NEGATIVE_FLOW_PROPAGATION |                      715 |              -0.002015 |              -0.003015 |                   -0.003929 |                   -0.003662 |
| development | TFG1_POSITIVE_FLOW_PROPAGATION |                      280 |              -0.003813 |              -0.004813 |                   -0.005287 |                   -0.005078 |
| development | TFG2_NEGATIVE_FLOW_PROPAGATION |                      296 |              -0.002542 |              -0.003542 |                   -0.004042 |                   -0.003695 |
| validation  | TFG1_POSITIVE_FLOW_PROPAGATION |                      204 |              -0.002066 |              -0.003066 |                   -0.003402 |                   -0.003304 |
| validation  | TFG2_NEGATIVE_FLOW_PROPAGATION |                      210 |              -0.001395 |              -0.002395 |                   -0.003992 |                   -0.003007 |
| holdout     | TFG1_POSITIVE_FLOW_PROPAGATION |                      176 |              -0.001627 |              -0.002627 |                   -0.002887 |                   -0.002206 |
| holdout     | TFG2_NEGATIVE_FLOW_PROPAGATION |                      209 |              -0.001892 |              -0.002892 |                   -0.003705 |                   -0.004273 |

Frozen graph months: `11`; edges: `2311`; real portfolio observations: `1375`.

This retrospective audit grants no PaperLive, leverage, or live-order permission.

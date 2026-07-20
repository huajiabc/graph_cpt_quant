# v22.1 SFI-on-FSS3 Overlay Feature Audit

Verdict: `feature_viable_freeze_sfi_fss3_overlay`.

| candidate                           | scope       |   weeks |   months |   mean_symbols |   mean_sfi_coverage |   mean_sfi_symbol_availability |
|:------------------------------------|:------------|--------:|---------:|---------------:|--------------------:|-------------------------------:|
| SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT | all         |      35 |        9 |        72.0000 |             46.6286 |                         0.6476 |
| SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT | development |      16 |        4 |        72.0000 |             48.6875 |                         0.6762 |
| SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT | validation  |      11 |        3 |        72.0000 |             45.4545 |                         0.6313 |
| SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT | holdout     |       8 |        2 |        72.0000 |             44.1250 |                         0.6128 |

The overlay uses the latest 00/12 UTC SFI snapshot with at least 30 eligible symbols between 12 and 36 hours before the unchanged Monday 00:00 FSS3 rebalance. Within each original funding-sign side, a frozen 0.50 rank tilt produces multipliers in [0.75, 1.25]; missing SFI names retain multiplier 1.0. Each side is renormalized to 0.5 raw notional, so no sign, name, or decision-time filter is introduced.

No future price, funding, return, PnL, or turnover outcome was calculated.

| input                                                                                 | sha256                                                           |
|:--------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| reports\v13_4_negative_funding_beta_neutral_rebound\weekly_symbol_panel.parquet       | FE2D7518E675EEE4D7A9EA6A5F0E97AA2A3D9CBE46EC3A9637B3F7AD1EBA8F99 |
| reports\v21_8_spot_perp_flow_inventory_feature_audit\decision_symbol_features.parquet | 25CFB10B95ED377130D33D1417C9588A488B9DE662A52F279B830BB366C9FD4F |

No live, PaperLive, application, leverage, remote, or order state changed.

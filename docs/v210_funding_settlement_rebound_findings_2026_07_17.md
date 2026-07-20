# v21.0 Funding-Settlement Rebound Findings

Verdict: `reject_funding_settlement_rebound_candidates`.

| candidate                            |   events |   mean_gross_bp |   mean_primary_net_bp |   break_even_cost_bp |   random_control_percentile |   day_bootstrap_lower_95_primary_net_bp | eligible   | status   |
|:-------------------------------------|---------:|----------------:|----------------------:|---------------------:|----------------------------:|----------------------------------------:|:-----------|:---------|
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND |      813 |         -0.1038 |              -20.1038 |              -0.1038 |                      0.8460 |                                -21.4601 | False      | rejected |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      |      505 |          0.8322 |              -19.1678 |               0.8322 |                      0.8380 |                                -21.0250 | False      | rejected |

| candidate                            | scope   |   events |   active_days |   active_months |   mean_selection_count |   mean_alt_bp |   mean_btc_hedge_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   mean_reversed_primary_net_bp |   positive_primary_fraction |
|:-------------------------------------|:--------|---------:|--------------:|----------------:|-----------------------:|--------------:|--------------------:|----------------:|----------------------:|---------------------:|-------------------------------:|----------------------------:|
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND | all     |      813 |           294 |              11 |                18.3383 |        0.4019 |             -0.5057 |         -0.1038 |              -20.1038 |             -40.1038 |                       -19.8962 |                      0.1292 |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      | all     |      505 |           262 |              11 |                 8.9703 |        1.2748 |             -0.4427 |          0.8322 |              -19.1678 |             -39.1678 |                       -20.8322 |                      0.1446 |

| candidate                            |   settlement_hour |   events |   mean_gross_bp |   mean_primary_net_bp |
|:-------------------------------------|------------------:|---------:|----------------:|----------------------:|
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND |                 0 |      270 |          0.8818 |              -19.1182 |
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND |                 8 |      267 |         -1.2622 |              -21.2622 |
| FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND |                16 |      276 |          0.0527 |              -19.9473 |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      |                 0 |      157 |          2.2319 |              -17.7681 |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      |                 8 |      161 |         -0.3664 |              -20.3664 |
| FSE2_NEW_NEGATIVE_ONSET_REBOUND      |                16 |      187 |          0.6889 |              -19.3111 |

The reveal follows the frozen v21.0 preregistration. Signal observation precedes entry by a full 15-minute bar; no funding payment is credited. Primary/stress results charge 20/40 bp round-trip book costs.

No live, PaperLive, application, leverage, remote, or order state changed.

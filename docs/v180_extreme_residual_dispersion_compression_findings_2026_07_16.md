# v18.0 Extreme Residual Dispersion Compression Findings

Verdict: `reject_extreme_residual_dispersion_compression`.

| candidate                                    |   events |   mean_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_rank_percentile |   delayed_primary_net_bp |   reversed_primary_net_bp |   positive_profit_concentration | eligible   | failed_gates                                                                                                                                                                                                                                               | verdict                                        |
|:---------------------------------------------|---------:|----------------------:|----------------------:|-----------------------:|-------------------------:|-------------------------:|--------------------------:|--------------------------------:|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|
| RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION |      484 |              -27.4560 |              -30.1703 |               -24.5813 |                   1.0000 |                 -28.8983 |                  -32.5440 |                             inf | False      | development_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|dispersion_q95_positive|dispersion_q99_positive|holding_30m_positive|holding_60m_positive|positive_profit_concentration_35 | reject_extreme_residual_dispersion_compression |

| candidate                                    | scope       |   events |   active_days |   active_months |   mean_dispersion_bp |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |
|:---------------------------------------------|:------------|---------:|--------------:|----------------:|---------------------:|----------------:|----------------------:|---------------------:|-------------------:|
| RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION | all         |      484 |           187 |              12 |             132.7717 |          2.5440 |              -27.4560 |             -37.4560 |             0.1364 |
| RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION | development |      203 |            90 |               6 |             146.5513 |          2.3190 |              -27.6810 |             -37.6810 |             0.1281 |
| RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION | validation  |      124 |            40 |               2 |             127.3345 |         -0.1845 |              -30.1845 |             -40.1845 |             0.1048 |
| RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION | holdout     |      157 |            57 |               4 |             119.2491 |          4.9898 |              -25.0102 |             -35.0102 |             0.1720 |

All beta estimates and dispersion thresholds use prior completed bars.
No live, PaperLive, application, leverage, remote, or order scope changed.

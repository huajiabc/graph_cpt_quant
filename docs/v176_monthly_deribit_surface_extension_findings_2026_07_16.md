# v17.6 Monthly Deribit Surface Extension Findings

Verdict: `reject_monthly_surface_extension`.

| candidate                                 |   events |   mean_primary_net_bp |   bootstrap_95_low_bp |   bootstrap_95_high_bp |   random_family_percentile |   delayed_primary_net_bp |   positive_year_share | eligible   | failed_gates                                                                                                                                                                                                                             | verdict                          | original_gate_pass   | strict_count_pass   | forward_watch   | lead_of_interest   |
|:------------------------------------------|---------:|----------------------:|----------------------:|-----------------------:|---------------------------:|-------------------------:|----------------------:|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|:---------------------|:--------------------|:----------------|:-------------------|
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR |       85 |              -34.9175 |              -58.4574 |               -11.6502 |                     0.4240 |                 -23.2012 |                   inf | False      | full_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_90|beats_one_day_delay|holding_8h_positive|holding_48h_positive|positive_year_share_50 | reject_monthly_surface_extension | False                | True                | False           | False              |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR |       73 |              -32.3560 |              -64.9942 |                 6.1216 |                     0.5080 |                 -51.0307 |                   inf | False      | full_primary_positive|validation_primary_positive|holdout_primary_positive|full_stress_positive|bootstrap_lower_positive|random_family_percentile_90|holding_8h_positive|holding_48h_positive|positive_year_share_50                     | reject_monthly_surface_extension | False                | False               | False           | True               |

| candidate                                 | scope       |   events |   active_years |   mean_gross_bp |   mean_primary_net_bp |   mean_stress_net_bp |   win_rate_primary |   sum_primary_net |
|:------------------------------------------|:------------|---------:|---------------:|----------------:|----------------------:|---------------------:|-------------------:|------------------:|
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | all         |       85 |              6 |          5.0825 |              -34.9175 |             -54.9175 |             0.2941 |           -0.2968 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | development |       37 |              3 |          8.0885 |              -31.9115 |             -51.9115 |             0.3514 |           -0.1181 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | validation  |       20 |              1 |          2.4928 |              -37.5072 |             -57.5072 |             0.2500 |           -0.0750 |
| DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR | holdout     |       28 |              2 |          2.9601 |              -37.0399 |             -57.0399 |             0.2500 |           -0.1037 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | all         |       73 |              6 |          7.6440 |              -32.3560 |             -52.3560 |             0.3151 |           -0.2362 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | development |       37 |              3 |          8.7248 |              -31.2752 |             -51.2752 |             0.3784 |           -0.1157 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | validation  |       17 |              1 |          5.5105 |              -34.4895 |             -54.4895 |             0.1765 |           -0.0586 |
| DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR | holdout     |       19 |              2 |          7.4481 |              -32.5519 |             -52.5519 |             0.3158 |           -0.0618 |

This is an overlapping historical coverage extension motivated by v17.5.
Even a pass can create only a forward-watch research lead, never a candidate.
The signal, graph, spread, horizon, and costs are unchanged. No live permission
changes are authorized.

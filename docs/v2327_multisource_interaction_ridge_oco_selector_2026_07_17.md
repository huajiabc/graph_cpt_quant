# v23.27 Multisource Interaction-Ridge OCO Selector

Verdict: `rejected_no_incremental_complex_model_alpha`.

| model             | scope      |   events |   selected_trades |   selection_rate |   primary_selected_expectancy_bp |   stress_selected_expectancy_bp |   primary_opportunity_return_bp |   stress_opportunity_return_bp |   unfiltered_primary_return_bp |   primary_spearman_ic |
|:------------------|:-----------|---------:|------------------:|-----------------:|---------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|----------------------:|
| interaction_ridge | validation |       47 |                18 |           0.3830 |                          -6.8824 |                        -16.8824 |                         -2.6358 |                        -6.4656 |                        -6.2300 |               -0.0961 |
| interaction_ridge | holdout    |       49 |                17 |           0.3469 |                           7.5314 |                         -2.4686 |                          2.6129 |                        -0.8565 |                       -16.3427 |                0.3045 |
| interaction_ridge | oos        |       96 |                35 |           0.3646 |                           0.1186 |                         -9.8814 |                          0.0432 |                        -3.6026 |                       -11.3917 |                0.0903 |

Passed gates: 1/11.
Random-label percentile: 78.12.
Month-bootstrap q05 (bp/opportunity): -8.3671.

The model uses only causal v23.26 features and fixed temporal fits.
Source ablations are outcome-seen diagnostics and are not promotion-eligible.
No PaperLive, leverage, remote, application, or order state changed.

# v23.30 Direct Volatility-Transmission Selector

Verdict: `rejected_direct_volatility_transmission_selector`.

| scope      |   events |   selected_trades |   selection_rate |   primary_selected_expectancy_bp |   stress_selected_expectancy_bp |   primary_opportunity_return_bp |   stress_opportunity_return_bp |   unfiltered_primary_return_bp |   score_primary_spearman |   mean_selected_score |   mean_unselected_score |
|:-----------|---------:|------------------:|-----------------:|---------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|-------------------------:|----------------------:|------------------------:|
| validation |       47 |                12 |           0.2553 |                         -28.7010 |                        -38.7010 |                         -7.3279 |                        -9.8811 |                        -6.2300 |                  -0.0057 |                0.5541 |                 -0.1431 |
| holdout    |       49 |                15 |           0.3061 |                          -5.0018 |                        -15.0018 |                         -1.5312 |                        -4.5924 |                       -16.3427 |                  -0.0274 |                0.4760 |                 -0.3069 |
| oos        |       96 |                27 |           0.2812 |                         -15.5348 |                        -25.5348 |                         -4.3692 |                        -7.1817 |                       -11.3917 |                   0.0053 |                0.5107 |                 -0.2238 |

Passed gates: 1/10.
Random same-count percentile: 38.26.
Month-bootstrap q05 (bp/opportunity): -15.4486.

The selection score and cutoff are outcome-free.
No PaperLive, leverage, remote, application, or order state changed.

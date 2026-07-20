# v23.32 Sparse Volatility-Tail Selector

Verdict: `rejected_sparse_volatility_tail_selector`.

| train_periods          | predict_period   | feature              | orientation   |   threshold |   training_selected |   prediction_selected |   training_winner_opportunity_return |   training_winner_margin |
|:-----------------------|:-----------------|:---------------------|:--------------|------------:|--------------------:|----------------------:|-------------------------------------:|-------------------------:|
| development            | validation       | leader_shock_breadth | low           |    0.000000 |                  32 |                    21 |                             0.000905 |                 0.000371 |
| development|validation | holdout          | alt_btc_abs_z_gap    | low           |   -0.385265 |                  33 |                    12 |                             0.001057 |                 0.000131 |

| scope      |   events |   selected_trades |   selection_rate |   primary_selected_expectancy_bp |   stress_selected_expectancy_bp |   primary_opportunity_return_bp |   stress_opportunity_return_bp |   unfiltered_primary_return_bp |
|:-----------|---------:|------------------:|-----------------:|---------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|-------------------------------:|
| validation |       47 |                21 |           0.4468 |                          -0.0874 |                        -10.0874 |                         -0.0390 |                        -4.5071 |                        -6.2300 |
| holdout    |       49 |                12 |           0.2449 |                         -24.3606 |                        -34.3606 |                         -5.9658 |                        -8.4148 |                       -16.3427 |
| oos        |       96 |                33 |           0.3438 |                          -8.9140 |                        -18.9140 |                         -3.0642 |                        -6.5017 |                       -11.3917 |

Passed gates: 2/10.
Full-search random-label percentile: 48.15.
Month-bootstrap q05 (bp/opportunity): -10.3287.

The random control repeats the complete 34-candidate search.
No PaperLive, leverage, remote, application, or order state changed.

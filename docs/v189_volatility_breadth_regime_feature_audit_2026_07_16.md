# v18.9 Volatility-Breadth Regime Feature-Only Audit

|   source_return_quantile | regime                 |   events |   development_events |   validation_events |   holdout_events |   median_breadth |   median_transmitted_receivers |   median_valid_receivers |
|-------------------------:|:-----------------------|---------:|---------------------:|--------------------:|-----------------:|-----------------:|-------------------------------:|-------------------------:|
|                   0.8500 | low_breadth_exhaustion |       15 |                   11 |                   2 |                2 |           0.0222 |                         1.0000 |                  45.0000 |
|                   0.8500 | high_breadth_cascade   |      275 |                  139 |                  51 |               85 |           0.6222 |                        28.0000 |                  45.0000 |
|                   0.9000 | low_breadth_exhaustion |        4 |                    4 |                   0 |                0 |           0.0222 |                         1.0000 |                  45.0000 |
|                   0.9000 | high_breadth_cascade   |      206 |                  106 |                  36 |               64 |           0.7556 |                        34.0000 |                  45.0000 |

Breadth uses only event-time returns standardized by prior-month risk
estimates, and q30/q70 thresholds use shifted prior-30-day breadth.
No future candidate return was calculated or inspected.

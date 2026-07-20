# v19.9 Binance Mark/Index Reference-Price Audit

Verdict: `audit_pass_reference_prices_ready_distinct_from_premium`.

Checks: 16; passed: 16; failed: 0.

No failed checks.

Exact research overlap: 32,459 bars x 46 symbols = 1,493,114 complete points.

The close-to-close implied basis `mark/index - 1` has aggregate correlation 0.959715 with the official premium-index close. Its median absolute difference is 0.00013197, and its 99th percentile is 0.00150946.

The two series are therefore related but not numerically interchangeable. Later feature work must measure incremental information relative to premium innovations instead of counting both as independent alpha.

Weakest per-symbol implied/premium relationships (retained, not removed):

| symbol       |   implied_premium_correlation |   median_abs_implied_premium_difference |   p99_abs_implied_premium_difference |
|:-------------|------------------------------:|----------------------------------------:|-------------------------------------:|
| FARTCOINUSDT |                    0.57783138 |                              0.00019117 |                           0.00153100 |
| 1000PEPEUSDT |                    0.60560666 |                              0.00025544 |                           0.00170465 |
| CRVUSDT      |                    0.67183544 |                              0.00021162 |                           0.00230532 |
| STRKUSDT     |                    0.70682942 |                              0.00031624 |                           0.00137905 |
| GALAUSDT     |                    0.71635718 |                              0.00018055 |                           0.00187526 |

All timestamps are exact completed-bar times; no fill and no future return was used. No live, PaperLive, application, leverage, remote, or order scope changed.

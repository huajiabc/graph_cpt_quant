# v23.22 Alt-First Volatility Ignition Independent Audit

Verdict: `audit_passed`.

| check                                 | passed   |
|:--------------------------------------|:---------|
| feature_audit_rebuilds                | True     |
| feature_hash_exact                    | True     |
| all_artifacts_exact_within_tolerance  | True     |
| all_numeric_errors_within_tolerance   | True     |
| exactly_100_events                    | True     |
| all_events_have_controls              | True     |
| failed_gate_set_exact                 | True     |
| rejection_verdict_exact               | True     |
| gross_result_negative_all_scopes      | True     |
| random_percentile_below_90_all_scopes | True     |
| primary_width_frozen                  | True     |

| artifact         |   rows_rebuilt |   rows_saved |   maximum_numeric_error | exact_within_tolerance   |
|:-----------------|---------------:|-------------:|------------------------:|:-------------------------|
| feature_states   |           9763 |         9763 |               0         | True                     |
| feature_events   |            100 |          100 |               0         | True                     |
| feature_summary  |              4 |            4 |               5.551e-17 | True                     |
| feature_checks   |             15 |           15 |               0         | True                     |
| control_universe |           3487 |         3487 |               0         | True                     |
| control_pools    |            883 |          883 |               0         | True                     |
| outcomes         |            100 |          100 |               0         | True                     |
| controls         |           3487 |         3487 |               0         | True                     |
| variants         |            300 |          300 |               0         | True                     |
| summary          |              4 |            4 |               3.553e-15 | True                     |
| variant_summary  |             12 |           12 |               3.553e-15 | True                     |
| random_paths     |           4000 |         4000 |               0         | True                     |
| random_summary   |              4 |            4 |               0         | True                     |
| bootstrap        |           5000 |         5000 |               0         | True                     |
| leaveout         |             12 |           12 |               3.553e-15 | True                     |
| gates            |              9 |            9 |               3.553e-15 | True                     |

The audit validates the negative gross result, weak matched-control
rank, all saved paths, and the rejection decision.

No live, PaperLive, leverage, remote, application, or order state changed.

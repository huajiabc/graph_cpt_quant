# v23.25 q90 Broad-Taker Confirmation Independent Audit

Verdict: `audit_passed`.

| check                                    | passed   |
|:-----------------------------------------|:---------|
| feature_audit_rebuilds                   | True     |
| feature_hash_exact                       | True     |
| all_artifacts_exact_within_tolerance     | True     |
| all_numeric_errors_within_tolerance      | True     |
| exactly_26_confirmed_and_27_unconfirmed  | True     |
| exactly_two_unmatched_controls           | True     |
| failed_gate_set_exact                    | True     |
| permutation_p_exact                      | True     |
| rejection_verdict_exact                  | True     |
| full_sample_gross_negative               | True     |
| confirmation_permutation_not_significant | True     |

| artifact            |   rows_rebuilt |   rows_saved |   maximum_numeric_error | exact_within_tolerance   |
|:--------------------|---------------:|-------------:|------------------------:|:-------------------------|
| feature_context     |             53 |           53 |               0         | True                     |
| feature_events      |             26 |           26 |               0         | True                     |
| feature_summary     |              4 |            4 |               5.551e-17 | True                     |
| feature_checks      |             14 |           14 |               0         | True                     |
| outcomes            |             26 |           26 |               0         | True                     |
| delayed             |             26 |           26 |               0         | True                     |
| unconfirmed         |             27 |           27 |               0         | True                     |
| control_universe    |           3318 |         3318 |               0         | True                     |
| control_pools       |            219 |          219 |               0         | True                     |
| controls            |           3318 |         3318 |               0         | True                     |
| summary             |              4 |            4 |               5.551e-17 | True                     |
| delayed_summary     |              4 |            4 |               3.553e-15 | True                     |
| unconfirmed_summary |              4 |            4 |               5.551e-17 | True                     |
| random_paths        |           4000 |         4000 |               0         | True                     |
| random_summary      |              4 |            4 |               7.105e-15 | True                     |
| bootstrap           |           5000 |         5000 |               0         | True                     |
| leaveout            |             10 |           10 |               1.776e-15 | True                     |
| permutations        |           5000 |         5000 |               0         | True                     |
| gates               |             12 |           12 |               3.553e-15 | True                     |

The audit confirms that broad taker buying does not explain q90 as
directional long alpha and reproduces the rejection exactly.

No live, PaperLive, leverage, remote, application, or order state changed.

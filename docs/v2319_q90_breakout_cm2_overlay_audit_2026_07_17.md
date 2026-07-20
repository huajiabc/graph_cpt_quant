# v23.19 q90 Breakout + CM2 Overlay Independent Audit

Verdict: `audit_passed`.

| check                                   | passed   |
|:----------------------------------------|:---------|
| v2317_feature_audit_rebuilds            | True     |
| frozen_feature_hash_exact               | True     |
| all_artifacts_exact_within_tolerance    | True     |
| all_numeric_errors_within_tolerance     | True     |
| exactly_53_events_and_49_weeks          | True     |
| all_events_triggered                    | True     |
| frozen_overlay_weight_exact             | True     |
| cross_week_events_use_realization_clock | True     |
| only_bootstrap_lower_gate_failed        | True     |
| rejection_verdict_exact                 | True     |

| artifact            |   rows_rebuilt |   rows_saved |   maximum_numeric_error | exact_within_tolerance   |
|:--------------------|---------------:|-------------:|------------------------:|:-------------------------|
| feature_mapping     |             53 |           53 |               0         | True                     |
| feature_summary     |              4 |            4 |               4.441e-16 | True                     |
| feature_checks      |             14 |           14 |               0         | True                     |
| event_outcomes      |             53 |           53 |               0         | True                     |
| weekly_portfolio    |             49 |           49 |               0         | True                     |
| summary             |              4 |            4 |               5.684e-14 | True                     |
| sensitivity         |             12 |           12 |               4.441e-16 | True                     |
| bootstrap           |          10000 |        10000 |               0         | True                     |
| leave_one_month_out |             12 |           12 |               4.441e-16 | True                     |
| evidence_gates      |             14 |           14 |               4.441e-16 | True                     |

The audit reproduces the feature clock, realization-week mapping,
event compounding, portfolio metrics, bootstrap, gates, and rejection
decision independently from the saved v23.18 artifacts.

No live, PaperLive, leverage, remote, application, or order state changed.

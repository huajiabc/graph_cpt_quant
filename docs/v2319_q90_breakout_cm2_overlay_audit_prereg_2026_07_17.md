# v23.19 q90 Breakout + CM2 Overlay Audit Preregistration

The audit will independently rebuild the v23.17 feature mapping from the
outcome-free q90 feature file and the CM2 calendar, then rebuild every v23.18
event, weekly portfolio, summary, sensitivity, bootstrap, leave-one-month-out,
and evidence-gate artifact from the frozen source files.

The audit passes only if timestamps, 53 event joins, 49 weekly rows, feature
hash, 10% overlay weight, numeric outputs, failed-gate set, and final verdict
all reproduce within `1e-12`. The expected research decision is rejection if
any v23.18 gate failed; the audit does not reinterpret that rule.

No live, PaperLive, leverage, remote, application, or order state is changed.

# v23.13 Positive-q80 Breakout Audit

Verdict: `audit_pass_validates_rejection`.

Audit checks: 13/13 passed.

All q80 event/control paths, matched subsets, 4,000 random paths,
5,000 month bootstraps, leave-one-month-out values, the two unmatched
events, and the rejection were replayed.

No live, PaperLive, leverage, remote, application, or order state changed.

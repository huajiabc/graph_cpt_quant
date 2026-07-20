# v22.6 Alt-Book Vacuum Pressure Independent Audit

Verdict: `audit_pass_validates_rejection`.

Audit checks: 16/16 passed.

Failed promotion gates: `positive_1h_gross`, `positive_primary`, `positive_stress`, `positive_validation`, `positive_holdout`, `positive_long`, `positive_short`, `positive_bootstrap_lower`, `random_percentile_95`, `beats_delayed`, `beats_no_vacuum`, `month_concentration`

Exact prices, all candidate/control returns, variance ratios, all
1,000 random paths, bootstrap summaries and governance metadata were
independently reconstructed. The rejection is valid.

No live, PaperLive, leverage, remote, application, or order state changed.

# v22.3 SFI-on-FSS3 Independent Audit

Verdict: `audit_pass_validates_rejection`.

Audit checks: 23/23 passed.

## Failed promotion gates

`positive_active_fss3_primary`, `positive_active_fss3_stress`, `positive_active_cm2_primary`, `positive_active_cm2_stress`, `positive_development`, `positive_validation`, `positive_holdout`, `positive_bootstrap_lower`, `random_percentile_95`, `beats_reversed`, `month_concentration`, `positive_lomo`

The audit independently reconstructed position PnL, funding, turnover,
costs, beta/gross constraints, CM2 arithmetic and inference summaries.
It validates the v22.2 rejection; it does not promote the overlay.

No live, PaperLive, leverage, remote, application, or order state changed.

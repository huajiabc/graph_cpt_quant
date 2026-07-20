# v15.6 Independent Audit of v15.5 BD2

Verdict: `rejection_confirmed`.

| raw_feature_audit_pass   | portfolio_audit_pass   | rejection_confirmed   |   raw_samples |
|:-------------------------|:-----------------------|:----------------------|--------------:|
| True                     | True                   | True                  |            80 |

|   days | source_lag_exact   | universe_exact   |   max_abs_turnover_difference |   max_abs_gross_return_difference |   max_abs_primary_return_difference |   max_abs_stress_return_difference |   max_abs_residual_btc_beta |   max_abs_gross_notional_drift | portfolio_audit_pass   |
|-------:|:-------------------|:-----------------|------------------------------:|----------------------------------:|------------------------------------:|-----------------------------------:|----------------------------:|-------------------------------:|:-----------------------|
|    375 | True               | True             |                     2.220e-16 |                         1.214e-17 |                           1.214e-17 |                          1.214e-17 |                   2.671e-16 |                      4.441e-16 | True                   |

The audit independently re-read deterministic raw ZIP samples,
matched all stored feature hashes to the download manifest, and
recomputed every portfolio return, turnover charge, BTC beta residual
and gross normalization. It confirms the v15.5 rejection; it does not
promote the reversed control. PaperLive and remote state are unchanged.

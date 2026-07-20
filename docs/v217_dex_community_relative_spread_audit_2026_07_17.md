# v21.7 DEX Community Relative-Spread Independent Audit

Verdict: `audit_pass_v216_rejection_reproduced`.

Passed 26/26 independent checks.

The audit independently reloaded prices, recomputed strictly prior monthly betas, and reproduced the dollar- and beta-neutral weights, symbol PnL, costs, chronology, 500 random-rank paths, day-block bootstrap, concentration, and rejection decision.

The 9.83 bp historical gross spread is below the 20 bp hurdle and falls to 3.55 bp in holdout. It should not be levered or promoted.

No live, PaperLive, application, leverage, remote, or order state changed.

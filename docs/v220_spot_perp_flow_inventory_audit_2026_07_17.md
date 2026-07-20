# v22.0 Spot-Perpetual Flow-Inventory Independent Audit

Verdict: `audit_pass_v219_rejections_reproduced`.

Passed 26/26 independent checks.

The audit independently reloaded perpetual prices, recomputed strictly prior hourly BTC betas, and reproduced weights, symbol PnL, costs, timing, all 1,000 random paths, day-block bootstrap, concentration, and both rejection decisions.

SFI2's 0.956 random percentile is real but economically too small: 5.00 bp gross overall and 9.30 bp in holdout, both below the 20 bp hurdle.

No live, PaperLive, application, leverage, remote, or order state changed.

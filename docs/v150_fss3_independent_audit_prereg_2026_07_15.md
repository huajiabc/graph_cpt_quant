# v15.0 FSS3 Independent Audit Preregistration

## Scope

Audit `FSS3_CURRENT_SIGN_070_TURNOVER_CAP` without importing or calling the
v14.9 target construction, beta-neutralization, turnover-cap, portfolio, PnL,
bootstrap, or null functions.

## Frozen checks

- Rebuild every seven-day funding score and future funding cash flow from raw
  settled Bybit funding records, with the same left/right endpoint timing.
- Rebuild every weekly coin and BTC return from raw closed one-hour Bybit bars.
- Independently reconstruct current-sign targets, current-beta BTC hedge,
  gross-one normalization, 0.70 capped transitions, initial entry, terminal close,
  funding cash flow, price PnL, 20bp primary cost and 40bp stress cost.
- Compare all reconstructed weekly fields to the saved v14.9 artifact with a
  5e-12 maximum absolute tolerance.
- Use a different seed, 10,000 circular four-week block-bootstrap draws, and
  5,000 full-universe random breadth-preserving paths. Each random path must use
  its own independently reconstructed 0.70 execution cap and realized turnover.
- Recheck every frozen v14.9 promotion gate. Also report leave-one-month-out,
  recent-six-week, funding-only-after-cost, realized BTC beta/correlation,
  up/down-BTC, drawdown, and symbol-contribution diagnostics.

Audit passes only if the structural reconstruction is within tolerance, all
frozen promotion gates pass, the alternate bootstrap lower bound is positive,
and the alternate null percentile is at least 95.

This audit cannot grant PaperLive, leverage, or real-order permission.

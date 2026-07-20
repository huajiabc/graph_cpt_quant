# v10.1 Exact-Flow Persistence and BTC Attribution - Pre-Registration

## Provenance and permission

This is a post-discovery follow-up to the secondary 240-minute diagnostic in
v10.0. It is not an independent confirmation and cannot change P2, shadow,
PaperLive, canary, leverage, or live permissions.

No v10.0 feature threshold may change. The only candidate is the frozen
`OF1_CONFIRM_LONG` state:

- 5m taker imbalance at least +0.10;
- 15m taker imbalance non-negative;
- 5m price return positive;
- 5m turnover divided by one third of 15m turnover at least 1.0.

## Primary outcome and attribution

- Primary holding period: 240 minutes.
- Raw long return: token entry-to-240m return minus 20bp round-trip cost.
- Cost ladder: 10/20/30bp.
- BTC attribution: token gross return minus simultaneous BTC gross return.
- A tradable unit-notional token-long/BTC-short view subtracts 40bp total
  round-trip cost. BTCUSDT events are excluded from the hedged view.
- BTC minute source: the existing continuous Bybit execution-public BTCUSDT
  kline file. Events without complete BTC coverage are excluded only from BTC
  attribution, not from the raw-long test.

The chronological development, validation, and holdout splits and 60-minute
per-symbol cooldown remain identical to v10.0.

## Controls

- 500 same-symbol/same-day random 15-minute timestamps, reapplying the frozen
  OF1 state and 240-minute outcome.
- The same random panels are evaluated for raw and BTC-hedged returns.
- A +60-minute shifted-event placebo.
- Day-block bootstrap with 2,000 iterations.
- Full, validation, holdout, path, symbol, day, and positive-day concentration
  summaries.

## Frozen decision gates

`OF1_CONFIRM_LONG_240M` remains at most a research clue only if every gate is
true:

1. at least 80 full, 25 validation, and 25 holdout raw-long trades;
2. at least six symbols and ten active days;
3. validation and holdout raw-long net20 are positive;
4. full raw-long net30 is positive;
5. day-block bootstrap 95% lower bound for raw net20 is positive;
6. raw net20 exceeds the matched-random 90th percentile;
7. raw net20 beats the +60-minute shifted placebo;
8. both original event paths have non-negative full raw net20;
9. no positive day supplies more than 35% of positive raw net20;
10. non-BTC hedged net40 is positive in validation and holdout and exceeds its
    matched-random 90th percentile.

Failure means the observed 240-minute shape is beta, timing noise, cost-fragile,
or too small to distinguish. It must not be reframed as an alpha pass.

# v15.2 Negative-Funding OI-State Spread Preregistration

Date frozen: 2026-07-16, before constructing or inspecting any v15.2 return.

## Hypothesis

Within the negative-funding universe, rising open interest represents fresh
short inventory and greater squeeze fuel, while falling open interest represents
an unwind already in progress. The incremental signal is therefore the future
price spread between OI-building and OI-unwinding coins, after removing the
common negative-funding bucket return.

The parent v13.8 study tested each OI state as a separate long/BTC-hedged basket;
it did not form a synchronous build-minus-unwind cross-sectional spread.

## Frozen candidate

`OI1_NEG_FUNDING_BUILD_MINUS_UNWIND`

- Weekly Monday entry and seven-day holding horizon.
- Eligible symbols have strictly negative prior-seven-day Bybit settled funding,
  valid causal Binance `oi_change_7d`, price return and prior-month BTC beta.
- Rank eligible symbols by `oi_change_7d` descending, breaking ties by symbol.
- Long the top `floor(n/2)` and short the bottom `floor(n/2)`; omit the middle
  symbol when `n` is odd. Require at least two names per side.
- Equal weight within each side with 0.5 raw gross per side.
- Add an exact current estimated BTC-beta hedge and normalize total gross to one.
- Use the frozen v14.9 execution rule: on continuous weekly transitions, move
  as far as possible toward the current target subject to full-L1 turnover at
  most 0.70. Initial entry, terminal close, gaps and mandatory exits are fully
  charged and not hidden by the cap.
- Primary/stress one-way costs are 20/40bp times realized full-L1 turnover.

The reversed OI direction is a diagnostic control and cannot be promoted.

## Frozen controls and gates

- 2,000 four-week moving-block bootstrap draws.
- 1,000 full-universe-within-negative-funding random OI-rank permutations.
  Every random path preserves weekly side breadth, uses the same beta hedge and
  0.70 execution rule, and pays its own realized turnover.
- Report thin (`4-7`) and broad (`8+`) eligible-breadth states, chronological
  splits, price/funding attribution, month concentration and FSS3/TG1 correlation.

Promotion requires at least 45 weeks, 11 months, ten validation and ten holdout
weeks; mean fully charged turnover no greater than 0.75; all cap-applicable
transitions no greater than 0.70; positive price return, 40bp stress return,
development, validation, holdout, thin and broad states, and bootstrap lower
bound; random-null percentile at least 95; positive-month concentration no
greater than 35%; worst period at least -40bp/week; absolute correlation with
FSS3 no greater than 0.50; exact beta/gross constraints within 1e-12; and mean
return above the reversed-direction control.

Passing means a new raw forward-shadow candidate. It grants no PaperLive,
leverage, remote-host or real-order permission.

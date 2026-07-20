# v13.6 Severity-Weighted Negative-Funding Basket Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v13.6 return.

## Motivation

v13.5 established positive cost-adjusted returns across 49 weeks, all three periods, and both full
and contracted negative-funding breadth states, but missed its random-basket attribution gate at
89.4% versus 90%. v13.6 replaces the discrete top-nine rank cut with the economically direct
quantity: the magnitude of the funding paid to a long position. It tests whether more severe
negative funding deserves more capital while retaining every negative-funding name for breadth.

## Frozen candidate

`NF3_ALL_NEGATIVE_Q75_SEVERITY_BTC_BETA_NEUTRAL` is the only candidate.

- Reuse the exact causal v13.4 weekly panel, chronology, price, funding, and monthly BTC beta.
- Emit a portfolio when at least four coins have strictly negative prior-seven-day settled funding.
- Hold every eligible negative-funding coin. Define raw severity as `-score_7d`, cap each severity
  at the eligible cross-section's 75th percentile, then normalize capped severities to sum to one.
- Let `b` be the severity-weighted causal BTC beta. Scale the long basket by `1 / (1 + b)` and
  short BTC by `b / (1 + b)`. Gross notional is one and estimated BTC beta is zero.
- Recompute weights only Monday 00:00 UTC and hold for seven days. No hold band is used because
  allocation varies continuously with the causal carry magnitude.
- No price, basis, volatility, graph, regime, weekday, or outcome filter is allowed.

## Costs and controls

- Charge 20 bp one-way primary and 40 bp one-way stress costs times exact signed-weight L1
  turnover, including initial entry and terminal close.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 within-week severity-permutation controls: preserve each week's eligible names and
  exact capped severity weights, randomly permute those weights across names, recompute the causal
  beta hedge, and charge the observed candidate cost path.
- Report contracted (4-8 eligible names) and broad (9+) states; both means must be positive.

## Promotion gate

Require at least 45 weeks, 11 months, ten validation weeks, and ten holdout weeks; positive primary
net return in development, validation, holdout, contracted, and broad states; positive funding and
stress-cost means; four-week-block bootstrap lower bound above zero; severity-permutation
percentile at least 90; positive-month concentration at most 35%; worst period at least -40
bp/week; mean turnover at most 0.50; and maximum absolute residual BTC beta at most `1e-12`.

Passing means forward-shadow candidacy only. PaperLive, live-order, leverage, and status permissions
remain unchanged by this retrospective test.

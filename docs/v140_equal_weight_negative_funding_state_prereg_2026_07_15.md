# v14.0 Equal-Weight Negative-Funding State Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v14.0 return.

## Motivation

v13.5-v13.9 repeatedly found positive beta-neutral returns in negative-funding baskets, while
funding severity, cross-venue confirmation, OI state, and taker-flow composites did not reliably
identify better coins within that state. This motivates the most direct remaining hypothesis: the
cross-sectional state `funding < 0` is itself the signal, and diversified exposure to every coin in
that state is preferable to another rank transform.

## Frozen candidate

`NF8_ALL_NEGATIVE_EQUAL_BTC_BETA_NEUTRAL` is the sole candidate.

- Reuse the exact v13.4 causal Monday panel through July 2026.
- At each Monday 00:00 UTC, select every coin whose settled Bybit funding sum over
  `[entry - 7d, entry)` is strictly negative and whose causal monthly BTC beta and forward return
  fields are complete. Require at least four names.
- Give every selected coin equal pre-hedge weight. Let `b` be their equal-weight causal BTC beta;
  scale the long basket by `1 / (1 + b)` and short BTC by `b / (1 + b)` so gross notional is one
  and estimated BTC beta is zero.
- Recompute weekly; there is no rank, top-k cut, hold band, score magnitude, OI, flow, Binance
  funding, price, basis, volatility, graph, regime, weekday, or outcome filter.

## Costs and controls

- Charge 20/40 bp one-way primary/stress costs times exact signed-weight L1 turnover, including
  initial entry, any cash gap, re-entry, and terminal close.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 within-week random baskets drawn from the full complete cross-section, preserving each
  week's exact observed breadth, using the identical equal-weight beta hedge and observed cost
  path. This tests the negative-funding state itself rather than ranking within it.
- Report contracted (4-8 names) and broad (9+) states; both means must be positive.

## Promotion gate

Require at least 45 weeks, 11 months, ten validation and ten holdout weeks; positive development,
validation, holdout, contracted, broad, funding, and stress-cost means; positive four-week-block
bootstrap lower bound; full-universe random-basket percentile at least 90; positive-month
concentration at most 35%; worst period at least -40 bp/week; mean turnover at most 0.50; and
maximum absolute residual BTC beta at most `1e-12`.

Passing means forward-shadow candidacy only. PaperLive, live-order, leverage, and lifecycle status
remain unchanged by this retrospective test.

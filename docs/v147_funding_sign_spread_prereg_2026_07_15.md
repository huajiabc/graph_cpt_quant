# v14.7 Funding-Sign Spread Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v14.7 portfolio return.

## Evidence-driven hypothesis

The validated NF8 result shows that broad equal weighting inside the negative-funding state is more
robust than severity ranks, OI/flow filters, or graph diversification. v12.3 also shows that a
low-funding-versus-high-funding spread has positive cash flow and price PnL, but its fixed top/bottom
nine buckets are statistically noisy. v14.7 tests the unexamined sign-only decomposition without a
within-sign rank.

At every frozen Monday entry with at least four negative-funding names and four positive-funding
names, use the preceding seven-day settled Bybit funding sum:

1. `FSS1_ALL_NEGATIVE_LONG_ALL_POSITIVE_SHORT`: long every negative name and short every positive
   name, equal weight inside each sign, with 50% raw notional per side.
2. `PFS1_ALL_POSITIVE_SHORT`: short every positive name, equal weight.

For both candidates, add the exact causal BTC hedge that zeroes the portfolio's trailing monthly
BTC beta, then rescale all positions to gross notional one. Hold seven days. Zero-funding names are
unused. No rank, severity, graph, OI, flow, price, volatility, or regime filter is allowed.

## Returns and execution

- Use the exact v14.0 weekly symbol panel through July 2026.
- Funding cash flow uses settlements strictly after entry and through exit.
- Price PnL uses exact weekly endpoints.
- Primary cost is 20 bp per one-way realized turnover; stress cost is 40 bp.
- Initial opening, weekly signed-weight changes, interruptions, and final liquidation are charged.

## Controls and promotion

- One thousand random full-universe paths preserve each week's observed long/short breadth,
  disjointness, candidate construction, BTC hedge, and observed turnover cost.
- Because two candidates are tested, compare each real mean with the per-iteration random family
  maximum and require the 95th percentile.
- Use 2,000 four-week moving-block bootstrap draws.
- Promotion requires at least 45 weeks, 11 months, 10 validation and 10 holdout weeks; mean turnover
  at most 0.75; positive funding return; positive primary return in development, validation,
  holdout, contracted-negative-breadth, and broad-negative-breadth states; positive stress return;
  positive bootstrap lower bound; random-family percentile at least 95%; positive-month
  concentration at most 35%; worst period at least -40 bp; and residual BTC beta at most `1e-12`.

Passing grants usable forward-shadow candidacy only. No PaperLive, leverage, or real-order
permission is granted.

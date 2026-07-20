# v13.5 Adaptive-Breadth Negative-Funding Rebound Preregistration

Date frozen: 2026-07-15, after the v13.4 coverage audit and before constructing or inspecting any
v13.5 return.

## Motivation

v13.4 passed every economic and statistical gate but emitted only 39 weeks because ten otherwise
fully covered decision dates contained fewer than nine strictly negative-funding coins. The
missing observations were verified as genuine signal breadth states, not missing beta, price, or
funding data. v13.5 tests whether the same mechanism remains executable when negative-funding
breadth contracts, while preserving a diversified minimum rather than forcing cash whenever the
cross-section has fewer than nine candidates.

## Frozen candidate

`NF2_ADAPTIVE4TO9_HOLD18_BTC_BETA_NEUTRAL` is the sole candidate.

- Reuse the exact v13.4 weekly causal panel, Monday timing, prior-seven-day settled funding score,
  forward seven-day return, monthly prior-30-day BTC beta, and BTC funding.
- Eligible names must have strictly negative prior-seven-day funding.
- Target breadth is `min(9, eligible_count)` and a portfolio is emitted only when at least four
  names are eligible.
- Retain previously held eligible names whose ascending negative-funding rank remains no worse
  than 18, capped at the current target breadth; fill vacancies from the most negative names.
- Let `b` be the selected names' mean causal BTC beta. Long each coin with weight
  `1 / (selected_count * (1 + b))` and short BTC with weight `b / (1 + b)`. Gross notional remains
  one and estimated BTC beta remains zero.
- No price, basis, volatility, graph, regime, weekday, or outcome filter is allowed.

## Costs, controls, and gate

- Primary/stress costs remain 20/40 bp one-way times exact signed-weight L1 turnover, including
  initial entry and terminal close.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 within-week random strictly-negative-funding baskets with each week's exact selected
  breadth, the same beta-neutral formula, and the observed candidate cost path.
- Preserve period boundaries and the full v13.4 promotion gate: at least 45 weeks, 11 months, ten
  validation weeks, ten holdout weeks; positive development/validation/holdout, funding, and
  stress-cost means; positive bootstrap lower bound; random percentile at least 90; positive-month
  concentration at most 35%; worst period at least -40 bp/week; mean turnover at most 0.50; and
  maximum absolute residual BTC beta at most `1e-12`.
- Additionally report results separately for full-nine-name and contracted-four-to-eight-name
  weeks. Both state means must be positive for promotion.

Passing means forward-shadow candidacy only. No PaperLive, live-order, leverage, or status
permission changes from this retrospective test alone.

# v13.8 Negative-Funding OI-State Family Preregistration

Date frozen: 2026-07-15, before extending the Binance metrics archive beyond 2026-06-05 and before
constructing or inspecting any v13.8 return.

## Motivation

Negative funding can arise from two distinct inventory processes. If open interest is rising,
fresh shorts are entering and may create squeeze fuel. If open interest is falling, a deleveraging
wave may already have exhausted forced selling. The previously rejected v12.2 study used four-hour
OI impulses as a generic directional cross-section; it did not condition a seven-day
negative-funding rebound basket on the inventory process that created the funding state.

## Causal OI feature

- Extend the official Binance USD-M five-minute metrics archives through 2026-07-14, merging rather
  than replacing the existing 2025-07-01 onward history.
- At Monday entry `t`, use the final `sum_open_interest` snapshot stamped strictly before `t` and
  the final snapshot stamped strictly before `t - 7d`.
- Define `oi_change_7d = log(OI_before_t) - log(OI_before_t_minus_7d)`.
- No OI value, price, return, future row, or interpolated snapshot enters the feature.

## Frozen candidates

Two candidates form one family; neither direction may be selected from outcomes.

1. `NF5_OI_BUILD_ADAPTIVE4TO9_BTC_BETA_NEUTRAL`: Bybit prior-seven-day funding is strictly
   negative and `oi_change_7d > 0`.
2. `NF6_OI_UNWIND_ADAPTIVE4TO9_BTC_BETA_NEUTRAL`: Bybit prior-seven-day funding is strictly
   negative and `oi_change_7d <= 0`.

Within each branch, rank by Bybit funding most negative first; target `min(9, eligible_count)` with
a four-name minimum; retain a prior name only while eligible and ranked no worse than 18; fill from
the most negative names. Use the exact v13.5 gross-one causal BTC-beta-neutral construction and
Bybit-only realized return. Binance OI is signal confirmation and contributes no return.

No price, basis, volatility, graph, Binance funding, taker flow, large-trader ratio, regime,
weekday, or outcome filter is allowed.

## Controls and promotion

- Charge 20/40 bp one-way primary/stress costs times exact signed-weight L1 turnover, including
  initial entry, cash gaps, re-entry, and terminal close.
- Use 2,000 four-week moving-block bootstrap draws per candidate.
- Use 1,000 within-week random baskets from the same candidate-specific OI/funding eligible set,
  with exact observed breadth, identical beta hedge, and observed cost path.
- Because two OI directions are tested, require random-basket percentile at least 95 rather than
  90 for either promotion.
- Preserve the other gates: at least 45 weeks, 11 months, ten validation and ten holdout weeks;
  positive development, validation, holdout, contracted-four-to-eight, full-nine, funding, and
  stress-cost means; at least five contracted and five full weeks; positive bootstrap lower bound;
  positive-month concentration at most 35%; worst period at least -40 bp/week; mean turnover at
  most 0.50; residual BTC beta at most `1e-12`.

Passing means forward-shadow candidacy only. PaperLive, live-order, leverage, and lifecycle status
remain unchanged by this retrospective family.

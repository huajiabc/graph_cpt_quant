# v13.9 Continuous Short-Crowding Pressure Rebound Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v13.9 return.

## Motivation

v13.8 found that a hard positive-OI branch was economically stronger than the unwind branch, but
the binary state caused high turnover and too few active weeks. A squeeze mechanism should be
strongest when three independently observed facts coincide continuously: funding is deeply
negative, open interest has increased, and aggressive takers have been net sellers. v13.9 ranks
that joint pressure without a hard OI or taker cutoff.

## Frozen signal

At Monday entry `t`, reuse v13.8's strictly prior Bybit seven-day funding and Binance seven-day OI
change. From the same official Binance five-minute metrics, compute the mean of
`log(sum_taker_long_short_vol_ratio)` over `[t - 7d, t)`. Lower values mean stronger aggressive
selling.

Among coins with strictly negative Bybit funding and complete OI/taker data, compute within-week
percentile ranks, where larger means more squeeze pressure:

- `funding_rank = pct_rank(-bybit_funding_7d)`;
- `oi_rank = pct_rank(oi_change_7d)`;
- `sell_rank = pct_rank(-mean_log_taker_ratio_7d)`;
- `pressure_score = (funding_rank + oi_rank + sell_rank) / 3`.

No learned coefficient, z-score threshold, sign switch, or outcome fitting is allowed.

## Frozen portfolio

`NF7_SHORT_CROWDING_PRESSURE_ADAPTIVE4TO9_BTC_BETA_NEUTRAL` is the sole candidate.

- Target breadth is `min(9, eligible_count)`, requiring at least four eligible names.
- Rank pressure descending. Retain a prior name while eligible and ranked no worse than 18; fill
  from the highest-pressure names.
- Use the exact v13.5 gross-one causal BTC-beta-neutral weights and Bybit-only realized return.
- Binance OI and taker flow are signal inputs only and contribute no return.
- No price, basis, volatility, graph, Binance funding, regime, weekday, or outcome filter is used.

## Controls and promotion

- Charge 20/40 bp one-way primary/stress costs times exact signed-weight L1 turnover, including
  entry, cash gaps, re-entry, and terminal close.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 random baskets from the same week's negative-funding eligible set with exact breadth,
  identical beta hedge, and observed cost path.
- Require the v13.5 gates: at least 45 weeks, 11 months, ten validation and ten holdout weeks;
  positive development, validation, holdout, contracted, full-nine, funding, and stress-cost
  means; positive bootstrap lower bound; random percentile at least 90; month concentration at
  most 35%; worst period at least -40 bp/week; mean turnover at most 0.50; residual BTC beta at
  most `1e-12`.

Passing means forward-shadow candidacy only. Because this hypothesis follows the v13.8 diagnosis,
it cannot authorize live capital without new forward observations. PaperLive, live-order,
leverage, and lifecycle permissions remain unchanged by this retrospective result.

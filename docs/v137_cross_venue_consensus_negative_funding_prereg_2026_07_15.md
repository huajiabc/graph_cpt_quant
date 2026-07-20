# v13.7 Cross-Venue Consensus Negative-Funding Rebound Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v13.7 return.

## Motivation

v13.5 showed that the most negative Bybit funding names contain substantial beta-neutral return,
but narrowly missed random-basket attribution. v13.6 showed that broadly weighting every negative
Bybit name is not sufficient. v13.7 adds an orthogonal, independently settled confirmation source:
a coin is eligible only when both Bybit and Binance USD-M funding were negative over the same
strictly prior seven-day window. This distinguishes market-wide short crowding from venue-local
funding noise.

## Frozen candidate

`NF4_DUAL_VENUE_NEGATIVE_ADAPTIVE4TO9_BTC_BETA_NEUTRAL` is the sole candidate.

- Reuse the exact v13.4 Bybit weekly causal panel and add Binance USD-M settled funding from the
  v13.2 historical-plus-recent archive.
- At Monday 00:00 UTC, compute separate Bybit and Binance funding sums over
  `[entry - 7d, entry)`. Both must be strictly negative.
- Rank eligible names by `Bybit seven-day sum + Binance seven-day sum`, most negative first.
- Target breadth is `min(9, eligible_count)`, with a minimum of four. Retain prior names while they
  remain dual-negative and rank no worse than 18; fill from the most negative ranks.
- Use the exact v13.5 gross-one BTC-beta-neutral weights based on causal monthly Bybit betas.
- Trade only Bybit coin perps plus the Bybit BTC hedge. Binance funding is signal confirmation and
  never credited to portfolio return.
- No price, basis, volatility, graph, regime, weekday, or outcome filter is allowed.

## Costs, controls, and gate

- Charge 20/40 bp one-way primary/stress costs times exact signed-weight L1 turnover, including
  entry and terminal close.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 within-week random baskets with each week's exact breadth, sampled only from the
  dual-negative eligible set, beta-neutralized identically, and charged the observed cost path.
- Preserve the full v13.5 gates: at least 45 weeks, 11 months, ten validation and ten holdout weeks;
  positive development, validation, holdout, contracted, broad, funding, and stress-cost means;
  bootstrap lower bound above zero; random percentile at least 90; month concentration at most
  35%; worst period at least -40 bp/week; mean turnover at most 0.50; residual BTC beta at most
  `1e-12`.

Passing means forward-shadow candidacy only. No PaperLive, live-order, leverage, or status
permission changes from this retrospective test.

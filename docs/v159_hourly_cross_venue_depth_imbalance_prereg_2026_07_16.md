# v15.9 Hourly Cross-Venue Depth Imbalance Pre-registration

Date frozen: 2026-07-16, after the daily v15.5 and v15.7 candidates were rejected
and before constructing or inspecting any hourly depth-return portfolio.

## Hypothesis

Order-book imbalance may be a short-lived state that is destroyed by daily
aggregation. The frozen direction is continuation: coins with relatively bid-heavy
Binance cumulative notional within 1% should outperform on Bybit during the next
non-overlapping hour, while ask-heavy coins should underperform.

This is a second-generation study using a previously seen calendar span but a new
30-second-to-hour information set and return horizon. Passing can only create a local
forward-shadow candidate; new future data are required before PaperLive.

## Frozen universe, time and feature

Use the same fixed 16 pre-sample symbols as v15.5. Raw Binance `bookDepth` archives
span 2025-07-01 through 2026-07-14 UTC. Bybit is the execution/mark venue.

For every UTC decision hour `H`, use all valid snapshots timestamped in the strict
half-open interval `[H-60 minutes, H)`. At each snapshot compute:

`(bid_notional_-1 - ask_notional_+1) /
 (bid_notional_-1 + ask_notional_+1)`.

The hourly feature is the median of these values and requires at least 90 valid
snapshot pairs. A decision requires finite features for all 16 coins, exact Bybit
marks at `H` and `H+1 hour`, and all causal betas. Whole hours with any missing
input are dropped; the universe is not resized.

- Development: decisions through 2025-12-31 23:00 UTC.
- Validation: 2026-01-01 through 2026-03-31.
- Holdout: decisions from 2026-04-01 onward.

## Frozen portfolio

Candidate: `BD3_HOURLY_CROSS_VENUE_DEPTH_CONTINUATION`.

- Rank the 16 coins descending by the hourly feature, with symbol as tie-break.
- Hold four longs and four shorts.
- Retain a long while it stays in the top eight and a short while it stays in the
  bottom eight; fill vacancies in rank order.
- Initial alt weights are +0.125 and -0.125 per selected coin.
- At each hour estimate BTC beta from the trailing 720 hourly Bybit returns ending
  at `H`, requiring at least 500 paired observations. Add the exact BTC hedge and
  normalize gross notional to 1.0.
- Hold from `H` to `H+1 hour`; horizons do not overlap.
- Primary cost is 20 bp per one-way L1 turnover and stress cost is 40 bp. Opening,
  hourly rebalancing, BTC hedge changes and closes around missing hours are charged.
  The final sample is not artificially force-closed.

## Frozen controls

1. Sign-reversed portfolio with identical ranking state, beta hedge and costs.
2. One-hour-stale feature with identical trade hours and construction.
3. 1,000 independent within-hour random rankings with the same 4/4 cardinality,
   top/bottom-eight holding band, current beta hedge and costs.
4. The analogous 5% notional imbalance is diagnostic-only and cannot be promoted.

## Frozen gates

All gates must pass for local forward-shadow candidacy:

- at least 7,500 trade hours and 12 calendar months;
- at least 1,800 validation hours and 2,200 holdout hours;
- positive overall primary and stress mean net return;
- positive primary mean in development, validation and holdout;
- 24-hour moving-block bootstrap 95% lower bound above zero;
- observed primary mean at or above the 99th percentile of random rankings;
- largest positive month no more than 25% of total positive-month PnL;
- mean one-way turnover no more than 0.25 per hour;
- primary mean above reversed and one-hour-stale controls;
- maximum absolute residual BTC beta and gross-normalization drift at most `1e-10`.

Failure of any gate rejects the candidate without changing sign, horizon, depth
band, aggregation window, universe, bucket size, holding band or costs. PaperLive
and remote state remain unchanged.

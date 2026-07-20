# v16.1 Hourly Liquidity-Withdrawal Amplification Pre-registration

Date frozen: 2026-07-16, after static signed-depth imbalance was rejected at daily
and hourly horizons and before constructing total-depth change features or inspecting
their returns.

## Hypothesis

Static bid/ask direction may be uninformative while changes in available liquidity
still matter. A beta-residual price move should be more likely to continue during
the next hour when displayed cumulative liquidity is being withdrawn. The candidate
therefore combines prior-hour residual price direction with a cross-sectional rank
of one-percent total-depth withdrawal.

This is a second-generation study on a previously used calendar span. Passing may
create only a local forward-shadow candidate and requires genuinely new future data
before PaperLive.

## Frozen data and timing

Use the same fixed 16-symbol universe, Binance 30-second `bookDepth` archives, Bybit
hourly marks, sample splits and rolling BTC betas as v15.9.

At decision hour `H`:

1. For each valid snapshot in `[H-60m, H)`, define one-percent total displayed
   notional as `bid_notional_-1 + ask_notional_+1`.
2. `total_depth_H` is its median over the interval, requiring at least 90 valid
   snapshot pairs.
3. The causal depth change is
   `log(total_depth_H / total_depth_(H-1))`, where the denominator uses snapshots
   strictly in `[H-120m, H-60m)`.
4. Withdrawal is the negative depth change. Rank withdrawal cross-sectionally among
   the fixed 16 symbols from 0 to 1 using average ranks.
5. Compute prior-hour Bybit residual return from `H-1` to `H` with the current causal
   trailing-720-hour beta: `coin_return - beta_H * BTC_return`.
6. Frozen score is `prior_residual_return * withdrawal_percentile`.

Thus depth never supplies direction by itself; it scales an already observed price
move. All inputs are available at `H`, and the portfolio is held from `H` to `H+1`.

## Frozen portfolio

Candidate: `LW1_HOURLY_LIQUIDITY_WITHDRAWAL_AMPLIFICATION`.

- Rank the 16 scores descending with symbol as tie-break.
- Hold four longs and four shorts.
- Retain longs in the top eight and shorts in the bottom eight; fill vacancies in
  rank order.
- Start alt weights at +0.125/-0.125, add the exact current Bybit BTC beta hedge,
  and normalize gross notional to 1.0.
- Primary cost is 20 bp and stress cost is 40 bp per one-way L1 turnover. Opening,
  rebalancing, BTC hedge changes and closes around missing hours are charged. The
  final sample is not force-closed.

## Frozen controls

1. `price_only`: rank the same prior-hour residual returns without the withdrawal
   percentile. Primary must outperform it.
2. `reversed`: invert the primary score direction with otherwise identical state,
   beta hedge and costs.
3. `one_hour_stale`: use the complete score formed one hour earlier.
4. `random_depth_pairing`: 1,000 paths that independently permute the 16 withdrawal
   percentiles across symbols within every hour, recompute scores, and apply the same
   holding band, beta hedge and costs.
5. Five-percent total-depth withdrawal is diagnostic-only and cannot be promoted.

## Frozen gates

Every condition is required for local forward-shadow candidacy:

- at least 7,500 trade hours and 12 calendar months;
- at least 1,800 validation and 2,200 holdout hours;
- positive primary and stress net mean overall;
- positive primary mean in development, validation and holdout;
- 24-hour moving-block bootstrap 95% lower bound above zero;
- primary at or above the 99th percentile of random-depth-pairing paths;
- largest positive month at most 25% of total positive-month PnL;
- mean one-way turnover at most 0.35 per hour;
- primary mean above price-only, reversed and one-hour-stale controls;
- maximum absolute residual BTC beta and gross-normalization drift at most `1e-10`.

Failure of any gate rejects the candidate without changing sign, horizon, depth
band, withdrawal transform, score, universe, holding band or costs. PaperLive and
remote state remain unchanged.

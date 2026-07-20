# v16.3 Within-Bucket Liquidity Quality Pre-registration

Date frozen: 2026-07-16, after v16.1's hourly withdrawal-amplification candidate
was independently rejected and before inspecting weekly depth-to-volume portfolio
returns.

## Hypothesis

Short-horizon changes in depth did not predict direction, but structural liquidity
quality may still earn a low-turnover cross-sectional premium. Within each frozen
graph bucket pair, the coin with more displayed one-percent depth relative to its
traded notional is expected to outperform the more fragile peer over the next week.

The direction is frozen as **long higher liquidity quality / short lower quality**.
This is a second-generation use of the same calendar span; passing can create only a
local forward-shadow candidate and requires new future data before PaperLive.

## Frozen graph, feature and timing

Use the eight fixed pairs from v15.7:

- BSP01: `SOLUSDT`, `DOGEUSDT`
- BSP02: `1000PEPEUSDT`, `WIFUSDT`
- BSP03: `ETHUSDT`, `ENAUSDT`
- BSP04: `HBARUSDT`, `AVAXUSDT`
- BSP05: `LINKUSDT`, `ONDOUSDT`
- BSP06: `XRPUSDT`, `XLMUSDT`
- BSP07: `FARTCOINUSDT`, `WLDUSDT`
- BSP08: `SEIUSDT`, `TIAUSDT`

At every Monday `W 00:00 UTC`, use only the preceding seven complete UTC days
`[W-7d, W)`:

- depth numerator: median of the causal hourly Binance one-percent total displayed
  notional (`bid_-1 + ask_+1`);
- activity denominator: mean Binance hourly quote volume over the same timestamps;
- liquidity quality: `log(depth_numerator / activity_denominator)`.

Each symbol requires at least 150 valid depth hours and 150 valid quote-volume hours.
All 16 symbols, Bybit entry/exit marks at `W` and `W+7d`, and trailing-720-hour
causal betas are required. Whole weeks with any missing input are dropped.

- Development decisions: through 2025-12-29.
- Validation: 2026-01-05 through 2026-03-30.
- Holdout: decisions from 2026-04-06 onward.

## Frozen portfolio

Candidate: `LQ1_WITHIN_BUCKET_LIQUIDITY_QUALITY`.

- In every pair, long the member with higher liquidity quality and short the other;
  symbol name breaks exact ties.
- Initial alt weights are +1/16 on each of eight longs and -1/16 on each of eight
  shorts.
- Add an exact current Bybit BTC beta hedge and normalize gross notional to 1.0.
- Hold for one non-overlapping week.
- Primary cost is 20 bp and stress cost is 40 bp per one-way L1 turnover. Opening,
  weekly rebalancing, BTC hedge changes and closes around missing weeks are charged.
  The final sample is not artificially force-closed.

## Frozen controls

1. `reversed`: long the lower-quality member of every pair.
2. `one_week_stale`: use the complete quality ranks formed one week earlier.
3. `raw_depth_only`: sort each pair by the seven-day median depth numerator without
   quote-volume normalization. Primary must outperform it.
4. `random_quality_pairing`: 1,000 paths that independently permute the 16 quality
   values across symbols within each decision week, re-evaluate the same eight pairs,
   and apply identical beta hedges and costs.
5. Five-percent depth-to-volume quality is diagnostic-only and cannot be promoted.

## Frozen gates

All gates must pass for local forward-shadow candidacy:

- at least 48 trade weeks and 11 calendar months;
- at least 12 validation weeks and 13 holdout weeks;
- positive overall primary and stress mean net return;
- positive primary mean in development, validation and holdout;
- four-week moving-block bootstrap 95% lower bound above zero;
- primary at or above the 99th percentile of random-quality-pairing paths;
- largest positive month at most 35% of total positive-month PnL;
- mean one-way weekly turnover at most 0.35;
- primary mean above reversed, one-week-stale and raw-depth-only controls;
- maximum absolute residual BTC beta and gross-normalization drift at most `1e-10`.

Failure of any gate rejects the candidate without changing direction, pair map,
normalization, lookback, horizon, weights or costs. PaperLive and remote state remain
unchanged.

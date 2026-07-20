# v15.7 Pair-Shock to Fragile-Receiver Pre-registration

Date frozen: 2026-07-16, after v15.5's own-depth directional hypothesis was
independently rejected and before inspecting any return from the propagation
feature below.

## New hypothesis

Displayed depth imbalance is not asked to predict its own direction. Instead, its
absolute magnitude defines a fragile receiver. Direction comes from the already
realized, causal beta-residual move of the receiver's peer in the same frozen graph
bucket. The hypothesis is that a large peer shock propagates during the next 24
hours more strongly when the other member's order book is one-sided.

This is a second-generation study on an already used calendar sample. Even if all
gates pass, it may only become a local forward-shadow candidate and requires new
future observations before PaperLive.

## Frozen graph and sample

The eight pairs are the two pre-sample July-2025 turnover leaders from each frozen
August-2025 v11.0 community:

- BSP01: `SOLUSDT`, `DOGEUSDT`
- BSP02: `1000PEPEUSDT`, `WIFUSDT`
- BSP03: `ETHUSDT`, `ENAUSDT`
- BSP04: `HBARUSDT`, `AVAXUSDT`
- BSP05: `LINKUSDT`, `ONDOUSDT`
- BSP06: `XRPUSDT`, `XLMUSDT`
- BSP07: `FARTCOINUSDT`, `WLDUSDT`
- BSP08: `SEIUSDT`, `TIAUSDT`

Use the same 375 fully causal decision days and Bybit mark returns established by
v15.5. Development ends 2025-12-31, validation is 2026-01-01 through 2026-03-31,
and holdout begins 2026-04-01.

## Frozen signal timing

At decision `D 00:00 UTC`:

1. Estimate trailing-30-day hourly BTC betas through `D`, with at least 500 paired
   observations, exactly as in v15.5.
2. Compute each coin's prior-day residual return from `D-1 00:00` to `D 00:00`:
   `coin_return - beta_at_D * BTC_return`.
3. Within each frozen pair, call the member with larger absolute residual return
   the source and the other member the receiver. Symbol name breaks exact ties.
4. Source direction is the sign of its residual return.
5. Receiver fragility is the absolute median Binance 1% notional-depth imbalance
   from source day `D-1`.
6. Propagation strength is
   `abs(source_residual_return) * receiver_fragility`.

The strategy then holds the receiver from `D` to `D+1`; no future or same-day
depth observation enters the signal.

## Frozen portfolio

Candidate: `VT4_PAIR_SHOCK_TO_FRAGILE_RECEIVER`.

- Among pair receivers whose source direction is positive, select the two highest
  propagation strengths as longs. Among negative-source receivers, select the two
  highest strengths as shorts.
- Retain an existing long/short while its source sign is unchanged and its
  side-specific strength remains in the top four; fill vacancies by strength.
- If either sign has fewer than two eligible pairs, hold cash for that day.
- Start active alt weights at +0.25 per long and -0.25 per short.
- Add an exact current Bybit BTC beta hedge and normalize active gross notional to
  1.0. Cash days have gross zero.
- Primary one-way L1 cost is 20 bp and stress cost is 40 bp. Opening, rebalancing,
  BTC hedge changes, and transitions to/from cash are charged. The final sample is
  not force-closed.

## Frozen controls

1. `reversed`: identical selected receivers but trade against source direction.
2. `one_day_stale`: use the complete signal formed at `D-1` to trade returns at `D`.
3. `source_only`: rank each side only by absolute source residual return, removing
   receiver fragility; primary must outperform it.
4. `random_pairing`: 1,000 iterations that permute the eight receiver assignments
   among the eight source signals within each day, then recompute fragility products,
   side selection, holding band, beta hedge and costs.

No alternative bucket, return horizon, depth band, fragility transform, or source
definition can replace the primary candidate in v15.7.

## Frozen gates

All conditions must hold for local forward-shadow candidacy:

- 375 calendar observations, at least 250 active days, at least 65 active validation
  days and 65 active holdout days;
- positive primary and stress mean net return over all calendar observations;
- positive primary mean in development, validation and holdout;
- 7-day moving-block bootstrap 95% lower bound above zero;
- primary mean at or above the 99th percentile of random-pairing paths;
- largest positive month no more than 35% of total positive-month PnL;
- mean calendar one-way turnover no more than 0.60;
- primary mean above reversed, one-day-stale and source-only controls;
- maximum active absolute residual BTC beta and gross-normalization drift at most
  `1e-10`.

Failure of any gate rejects the candidate without tuning. PaperLive and remote
state remain unchanged.

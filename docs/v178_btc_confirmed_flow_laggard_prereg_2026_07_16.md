# v17.8 BTC Confirmed-Flow Laggard Transmission Preregistration

Status: `PREREGISTERED_OFFLINE_RESEARCH_ONLY`

No result from this BTC-source/15-minute rule was inspected before freezing the
specification. It is distinct from rejected v14.1: v14.1 allowed any alt flow
node to lead any other node and held for one hour; v17.8 uses only BTC as the
source, requires price/active-flow confirmation, and trades only contemporaneous
laggards for 30 minutes.

## Data and timing

- Binance USD-M 15m closed bars for the 50 locally frozen high-turnover symbols.
- Exclude BTC from receiver candidates and exclude XAUT because of partial
  history and different asset behavior.
- Available window: 2025-06-03 through 2026-06-04 UTC.
- Development: before 2026-01-01; validation: 2026-01-01 through 2026-02-28;
  holdout: 2026-03-01 onward.
- Every signal is known only at `bar_close_time`; entry uses that closed price and
  exit uses the closed price 30 minutes later.

## BTC source event

From the previous 30 days of BTC 15m bars, shifted one bar before every decision,
require at least 20 days of history and all:

1. absolute BTC 15m return at or above its rolling 97.5th percentile;
2. signed taker imbalance `direction * (2*taker_buy_quote/turnover - 1)` at or
   above the rolling 80th percentile of absolute imbalance;
3. BTC turnover at or above its rolling 75th percentile;
4. price direction and taker imbalance agree;
5. global four-bar (one-hour) cooldown.

## Causal receiver graph

At each calendar month start, use only the preceding 30 days:

- Estimate each alt's contemporaneous BTC beta.
- Residualize alt 15m returns by that beta.
- Calculate robust rank correlations from BTC return at `t` to alt residual at
  `t+1`, and from absolute BTC return at `t` to absolute alt residual at `t+1`.
- Calculate the reverse alt-to-next-BTC absolute correlation.
- Require at least 2,000 paired bars and positive signed forward correlation.
- Score = signed forward correlation + 0.5 times absolute direction advantage.
- Freeze the top ten receivers for the month.

At a source event, compute each receiver's same-bar BTC-residual return. A
laggard must have non-positive return after multiplying by BTC event direction.
Take up to the five most negative laggards and require at least three.

## Frozen candidates

- `BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP`: trade the selected alt laggards in BTC
  event direction for 30m. Primary/stress costs: 20/30 bp total.
- `BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP`: same alt positions with beta-scaled BTC
  hedge, normalized to unit gross. Primary/stress costs: 30/40 bp total.

No leverage is used.

## Required controls

- 500 random receiver pools of matching size, with laggards reselected causally;
  compare against the two-candidate family maximum.
- Exact accepted source events delayed by one 15m bar.
- Reversed trade direction on the same selected laggards.
- Source-return thresholds at rolling 95th and 99th percentiles.
- Holding periods of 15m and 60m as diagnostics.
- Day-block bootstrap with 2,000 iterations and month concentration.

## Candidate gate

Each candidate must pass all:

- at least 100 full, 20 validation, and 25 holdout events;
- primary-cost mean positive in development, validation, and holdout;
- full stress-cost mean positive;
- day-block bootstrap 95% lower bound positive;
- random receiver-family percentile at least 95%;
- beats both one-bar delay and reversed direction;
- 95th/99th-percentile source sensitivities remain positive after primary cost;
- 15m and 60m holding diagnostics have the same positive sign;
- no positive month contributes more than 35% of total positive PnL.

Passing creates an offline research candidate only. No PaperLive, application,
remote, leverage, or real-order permission is granted.

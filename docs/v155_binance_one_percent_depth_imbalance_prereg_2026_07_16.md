# v15.5 Binance One-Percent Depth Imbalance Pre-registration

Date frozen: 2026-07-16, after the v15.4 0.2% field failed its data-coverage gate
and before inspecting any portfolio return from a Binance `bookDepth` field.

## Availability-driven change from v15.4

The signed 1% band exists across the full archive window and was already named as
a v15.4 diagnostic. It becomes the sole v15.5 primary feature because it has a
consistent historical schema, not because of observed returns. The 0.2% field is
retired for this sample. The 5% version remains diagnostic only.

## Frozen hypothesis, universe and sample

Hypothesis direction is continuation: relatively bid-heavy cumulative displayed
liquidity within 1% predicts stronger next-24-hour Bybit returns, and relatively
ask-heavy liquidity predicts weaker returns.

The same pre-sample, fixed 16-symbol universe is used:

`SOLUSDT, DOGEUSDT, 1000PEPEUSDT, WIFUSDT, ETHUSDT, ENAUSDT, HBARUSDT,
AVAXUSDT, LINKUSDT, ONDOUSDT, XRPUSDT, XLMUSDT, FARTCOINUSDT, WLDUSDT,
SEIUSDT, TIAUSDT`.

- Binance source-depth window: 2025-07-01 through 2026-07-14 UTC.
- Bybit is the frozen execution/mark-price venue for alts and the BTC hedge.
- Development decisions: through 2025-12-31.
- Validation: 2026-01-01 through 2026-03-31.
- Holdout: from 2026-04-01 onward.
- A decision requires all 16 prior-day features, all 16 Bybit entry/exit marks, a
  BTC entry/exit mark, and causal betas. Whole days with any missing input are
  dropped; the universe is never resized.

## Frozen feature and timing

For each symbol and UTC source day, compute at every valid snapshot:

`(bid_notional_-1 - ask_notional_+1) /
 (bid_notional_-1 + ask_notional_+1)`.

The feature is the median of those snapshot imbalances. Source day `D-1` ranks the
portfolio at `D 00:00 UTC`; it is held at Bybit marks from `D 00:00` to `D+1 00:00`.
No observation timestamped on decision day `D` enters the feature.

## Frozen portfolio and costs

Candidate: `BD2_PRIOR_DAY_ONE_PERCENT_DEPTH_CONTINUATION`.

- Rank descending, with symbol name as the deterministic tie-break.
- Hold four longs and four shorts.
- Retain a long while it ranks in the top eight and a short while it ranks in the
  bottom eight; fill vacancies in rank order.
- Initial alt weights are +0.125 per long and -0.125 per short.
- Estimate each alt's BTC beta from the trailing 30 calendar days of Bybit hourly
  returns ending at the decision mark, requiring at least 500 paired observations.
- Add the exact BTC hedge and normalize total gross notional to 1.0.
- Primary cost: 20 bp per one-way L1 weight turnover. Stress cost: 40 bp.
- Turnover includes the initial opening and all BTC hedge changes. No artificial
  terminal close is charged to the final daily observation.

## Frozen controls

1. Sign-reversed portfolio with identical timing, beta hedge and costs.
2. One-day-stale feature (`D-2` at decision `D`) with identical construction.
3. 1,000 within-day random rankings using the same holding band, beta hedge and
   costs.
4. The 5% notional-imbalance version is reported only as an availability/signal
   diagnostic and cannot be promoted in v15.5.

## Frozen gates

Promotion is only to a local `forward-shadow candidate`, never directly to
PaperLive. Every gate must pass:

- at least 300 usable decision days and 10 calendar months;
- at least 80 validation and 80 holdout days;
- positive overall primary and stress mean net return;
- positive primary mean in development, validation and holdout;
- 7-day moving-block bootstrap 95% lower bound above zero;
- observed primary mean at or above the 95th percentile of random rankings;
- largest positive month at most 35% of total positive-month PnL;
- mean one-way turnover at most 0.50;
- primary mean above both reversed and one-day-stale controls;
- maximum absolute residual BTC beta and gross-normalization drift at most `1e-10`.

Failure of any gate rejects v15.5 without tuning sign, universe, horizon, bucket
size, holding band, cost, or depth percentage. PaperLive and remote state remain
unchanged.

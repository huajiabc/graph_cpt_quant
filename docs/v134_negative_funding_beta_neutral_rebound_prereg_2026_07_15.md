# v13.4 Negative-Funding Beta-Neutral Rebound Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v13.4 portfolio return.

## Motivation

The preregistered FC1 long-low/short-high funding portfolio had positive returns in all three
chronological periods, but its bootstrap lower bound failed. Prior sleeve attribution showed that
its economic cash flow came mainly from the long extremely negative-funding contracts, whereas
shorting high-positive-funding contracts was not the source of the edge. v13.4 therefore tests a
different executable structure: a diversified negative-funding capitulation basket hedged with a
causal BTC-beta leg, rather than an unstable high-funding short sleeve.

## Frozen data and causal timing

- Reuse the exact v13.2 Bybit perpetual prices, settled funding, and causally generated monthly
  universe through July 2026; add BTC funding from the same official archive and recent backfill.
- Decisions occur only Monday 00:00 UTC and hold for exactly seven days.
- The signal is each coin's settled funding sum in `[entry - 7d, entry)`.
- Monthly BTC betas use only the preceding 30 days of hourly Bybit returns before the month starts.
- Realized price uses entry and exit closes; realized funding uses settlements in `(entry, exit]`.

## Frozen candidate

`NF1_LOW9_HOLD18_BTC_BETA_NEUTRAL` is the sole candidate.

- Eligible names must have a strictly negative prior-seven-day funding sum.
- Initially hold the nine most negative names. At later decisions retain a name only if it remains
  negative and its ascending funding rank is no worse than 18; fill vacancies from the most
  negative eligible names until nine are held.
- Let `b` be the selected basket's mean causal BTC beta. If `b <= 0` or fewer than nine names are
  eligible, emit no portfolio for that week.
- Long each selected coin with weight `1 / (9 * (1 + b))`; short BTC with weight `b / (1 + b)`.
  Thus gross notional is one and estimated BTC beta is zero without a fitted outcome model.
- Funding PnL for signed weight `w` is `-w * settled_funding`; positive BTC funding is received by
  the BTC short hedge.

There is no price-momentum, realized-volatility, basis, graph-community, regime, weekday, or
outcome-informed filter.

## Costs and controls

- Primary cost is 20 bp one-way times exact L1 change in signed portfolio weights, including
  initial entry and terminal close. Stress cost is 40 bp one-way.
- Report long-basket price, BTC-hedge price, coin funding, BTC funding, gross return, exact turnover,
  and estimated residual BTC beta.
- Preserve chronological boundaries at 2026-01-01 and 2026-04-01.
- Use 2,000 four-week moving-block bootstrap draws.
- Use 1,000 within-week random baskets of nine negative-funding names, beta-neutralized by the same
  formula and charged the observed candidate's exact cost path.

## Promotion gate

Promotion means forward-shadow candidacy only and requires:

- at least 45 weeks, 11 months, ten validation weeks, and ten holdout weeks;
- positive primary net return in development, validation, and holdout;
- positive full-sample stress net return and funding contribution;
- four-week-block bootstrap 95% lower bound above zero;
- random-negative-funding-basket percentile at least 90;
- positive-month contribution concentration no greater than 35%;
- worst period mean no worse than -40 bp/week;
- mean weekly turnover no greater than 0.50;
- maximum absolute estimated post-hedge BTC beta no greater than `1e-12`.

No PaperLive, live-order, leverage, or strategy-status permission changes from this retrospective
test alone.

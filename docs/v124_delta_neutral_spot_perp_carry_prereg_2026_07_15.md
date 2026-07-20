# v12.4 Delta-Neutral Spot/Perpetual Carry Preregistration

Date frozen: 2026-07-15, before downloading or inspecting any v12.4 spot/perpetual portfolio return.

## Motivation and mechanism

v12.3 found that high/low perpetual funding dispersion contains a large realized
cash-flow component, but its focal portfolio failed the bootstrap lower-bound gate
because the long/short perpetual sleeves still carried material coin-direction risk.
v12.4 tests a distinct implementation: own spot and short the same coin's Bybit
linear perpetual. The pair should retain positive funding paid to the perpetual
short while replacing directional coin return with spot/perpetual basis return.

## Frozen as-of data and eligible universe

- Frozen v11.0 monthly 8-by-9 memberships.
- Bybit settled funding records. A decision at `t` may use settlements strictly
  before `t`; a settlement stamped exactly `t` is excluded from the score.
- Binance public spot one-hour klines and the existing Bybit 15-minute perpetual
  close panel. Prices at `t` are from bars fully closed by `t`.
- Symbol aliases are frozen as `1000BONKUSDT -> BONKUSDT`,
  `1000PEPEUSDT -> PEPEUSDT`, and `SHIB1000USDT -> SHIBUSDT`.
- A symbol-week is eligible only when it belongs to that month's frozen
  membership, has exact spot and perpetual entry/exit closes, and has funding
  observations in the score window. Missing symbols are not replaced after
  portfolio construction.

The formal period starts 2025-08-04 and ends at the last complete seven-day
label. Weekly entry is Monday 00:00 UTC and exit is exactly seven days later.
Funding PnL includes settlements in `(entry, exit]`.

## Frozen portfolios

Every selected coin is held as one dollar long Binance spot and one dollar short
Bybit perpetual. Pair return per dollar of spot notional is:

`spot_return - perpetual_return + sum(future settled funding rate)`.

The basis and funding components are reported separately. No leverage is applied
inside the research return; required perpetual margin is an implementation/risk
overlay and cannot rescue a failing alpha result.

1. `DN1_7D_TOP_FUNDING`: rank eligible symbols by the preceding seven-day sum
   of settled funding and equally weight the top nine, provided all selected
   scores are positive.
2. `DN2_30D_TOP_FUNDING`: rank by the preceding 30-day mean settled funding and
   equally weight the top nine, provided all selected scores are positive.
3. `DN3_COMMUNITY_TOP_FUNDING`: select the highest preceding seven-day funding
   symbol in each of all eight frozen communities; every selected score must be
   positive and all eight communities must be represented.

There is no basis filter, volatility filter, sign optimization, horizon selection,
or post-result replacement. The 7-day direct portfolio is the focal candidate;
the other two are separately gated diversification/persistence variants.

## Costs and chronology

- Focal all-in round-trip cost: 40 bp per pair portfolio, corresponding to two
  entries and two exits across the spot and perpetual legs.
- Stress all-in round-trip cost: 80 bp.
- Descriptive realized-turnover cost: 20 bp one-way per unit of pair-name weight,
  with initial entry and terminal close included.
- Development: through 2025-12-31; validation: 2026-01-01 through 2026-03-31;
  holdout: 2026-04-01 onward.

## Frozen controls and promotion gates

- 2,000 week-block bootstrap resamples of each observed candidate.
- 500 within-week random positive-funding baskets of the same size for DN1/DN2.
- 200 random monthly 8-by-9 partitions with the same top-positive rule for DN3.
- A low-funding/reverse diagnostic may be reported but cannot be promoted.

Promotion requires at least 40 complete weeks, ten active months, ten validation
weeks, and eight holdout weeks; positive net 40 bp in development, validation,
and holdout; positive full-sample net 80 bp; positive mean funding contribution;
positive bootstrap 95% lower bound after 40 bp cost; null percentile at least 90;
positive-month contribution no greater than 35%; and worst period mean no worse
than -40 bp per week. A promoted candidate would enter forward shadow evaluation,
not PaperLive, until live spot/perpetual execution and margin controls are built.

No existing PaperLive strategy is modified.

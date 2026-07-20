# v12.5 Cross-Venue Same-Coin Perpetual Carry Preregistration

Date frozen: 2026-07-15, before downloading or inspecting any v12.5
cross-venue portfolio return.

## Motivation and mechanism

The v12.3 attribution shows that its funding cash flow came mainly from the
long sleeve of extremely negative Bybit funding, while v12.4 shows that merely
shorting high-positive funding against spot is too small and unstable after
cost. Shorting spot against a long negative-funding perpetual would require
historical coin-borrow availability and cost that are not available in the
research box. v12.5 instead tests an executable same-coin hedge: long the
low-funding Bybit perpetual and short the Binance USD-M perpetual when the
cross-venue funding spread is positive.

## Frozen as-of data and universe

- Frozen v11.0 monthly 8-by-9 memberships.
- Bybit linear and Binance USD-M perpetual settled funding. Funding timestamps
  are normalized down to the scheduled whole second before comparison so a
  Binance archive's millisecond recording offset does not move an event across
  a decision boundary. Scores use settlements strictly before entry.
- Existing Bybit 15-minute and Binance public one-hour perpetual klines. Prices
  at `t` use only bars fully closed by `t`.
- The only symbol alias is `SHIB1000USDT -> 1000SHIBUSDT` on Binance USD-M.
- A symbol-week is eligible only when that month's frozen member has exact
  entry/exit closes and both venues have funding observations in the score
  window. Missing contracts are not replaced after portfolio construction.

The formal period starts 2025-08-04 and ends at the last complete seven-day
label. Entry is Monday 00:00 UTC; exit is exactly seven days later. Future
funding includes settlements in `(entry, exit]`.

## Frozen portfolios

Each selected pair holds one dollar long Bybit perpetual and one dollar short
Binance USD-M perpetual. Return per dollar of pair notional is:

`Bybit perp return - Binance perp return + Binance funding - Bybit funding`.

The price-basis and net-funding components are reported separately.

1. `CV1_7D_FUNDING_SPREAD`: rank the preceding seven-day cumulative spread
   `Binance funding - Bybit funding`; equally weight the top nine, provided all
   selected spreads are positive.
2. `CV2_30D_FUNDING_SPREAD`: apply the same rule to the preceding 30-day
   cumulative funding spread.
3. `CV3_COMMUNITY_FUNDING_SPREAD`: select the highest positive preceding
   seven-day spread in each of all eight frozen communities.

There is no price-basis filter, volatility filter, venue-direction optimization,
horizon selection, or post-result symbol substitution. CV1 is focal; CV2 and
CV3 are independently gated persistence/diversification variants.

## Costs, chronology, and controls

- Focal all-in round-trip cost: 40 bp per pair portfolio for two perpetual legs.
- Stress all-in round-trip cost: 80 bp.
- Descriptive realized-turnover cost: 20 bp one-way per unit pair-name weight,
  including initial entry and terminal close.
- Development: through 2025-12-31; validation: 2026-01-01 through 2026-03-31;
  holdout: 2026-04-01 onward.
- 2,000 week-block bootstrap resamples.
- 500 within-week random positive-spread baskets for CV1/CV2.
- 200 random monthly 8-by-9 partitions using the same top-positive rule for CV3.
- The reverse venue orientation is diagnostic only and cannot be promoted.

Promotion requires at least 40 complete weeks, ten active months, ten validation
weeks, and eight holdout weeks; positive net 40 bp in development, validation,
and holdout; positive full-sample net 80 bp; positive mean net-funding component;
positive bootstrap 95% lower bound after 40 bp; null percentile at least 90;
positive-month contribution no greater than 35%; and worst period mean no worse
than -40 bp per week. Passing means forward shadow candidacy, not PaperLive.

No existing PaperLive strategy is modified.

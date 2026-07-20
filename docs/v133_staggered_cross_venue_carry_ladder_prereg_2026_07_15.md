# v13.3 Staggered Cross-Venue Carry Ladder Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v13.3 return.

## Motivation

The exact forward-extended TG1 mechanism has positive development, validation, holdout,
stress-cost, bootstrap, and random-basket evidence. Its sole formal failure is positive-month
concentration, and six newly observed weeks are mixed. This test changes execution timing rather
than the alpha score: seven independently rebalanced weekday sub-books remove dependence on a
single Monday entry and form one continuously invested, same-coin cross-venue carry ladder.

## Frozen data and causal signal

- Reuse the exact Bybit and Binance perpetual price/funding histories and the causally generated
  monthly universe through July 2026 from v13.2.
- Every weekday sub-book rebalances only at 00:00 UTC on its own weekday.
- Its score is the strictly prior 30-day settled funding spread, `Binance - Bybit`.
- Venue orientation remains one dollar long Bybit perpetual and one dollar short Binance USD-M
  perpetual for each selected coin.
- Daily pair PnL is `Bybit return - Binance return + Binance funding - Bybit funding`, with funding
  settlements in `(day_start, day_end]`.

## Frozen portfolio

`SL1_7COHORT_30D_TOP9_HOLD18` is the only candidate.

- Seven equal-capital sub-books correspond to Monday through Sunday.
- At each sub-book's weekly rebalance, retain a name only while its current positive-spread rank is
  no worse than 18; fill to nine from the highest positive-spread ranks.
- Each sub-book is equal weighted across nine names and receives exactly one seventh of portfolio
  capital. There is no basis, volatility, graph-community, regime, or outcome-informed filter.
- The first calendar week establishes the seven cohorts and is a burn-in. Its entry costs are
  carried into the first evaluated full-ladder week; its PnL is not counted. The last evaluated
  complete calendar week receives the terminal close cost.
- Evaluation uses non-overlapping Monday-to-Monday sums of daily portfolio PnL.

## Frozen costs and controls

- Primary cost: 20 bp one-way times exact portfolio-level L1 name-weight turnover.
- Stress cost: 40 bp one-way times the same turnover.
- Report price-basis, funding-spread, gross, turnover, and both net returns.
- Preserve the v13.2 chronological boundaries: development before 2026-01-01, validation before
  2026-04-01, and holdout thereafter.
- Bootstrap 2,000 four-week moving blocks of non-overlapping calendar-week returns.
- Run 500 causal random-positive-spread ranking paths through the same seven-cohort schedule. For
  attribution, each null uses the observed candidate cost path, so random turnover cannot create
  an artificial disadvantage.
- Report correlation to the Monday-only TG1 return on overlapping evaluation weeks.

## Promotion gate

Promotion means **forward shadow candidacy only** and requires all of:

- at least 45 full-ladder weeks, 11 active months, ten validation weeks, and ten holdout weeks;
- positive primary net return in development, validation, and holdout;
- positive full-sample stress net return and funding contribution;
- four-week-block bootstrap 95% lower bound above zero;
- random-path percentile at least 90;
- positive-month contribution concentration no greater than 35%;
- worst chronological-period mean no worse than -40 bp/week;
- mean weekly portfolio turnover no greater than 0.50;
- absolute correlation to Monday-only TG1 below 0.95, proving entry-time diversification is not an
  identical replay.

No existing PaperLive, live-order, leverage, or status permission changes from this retrospective
result alone.

# v12.6 Turnover-Governed Cross-Venue Carry Preregistration

Date frozen: 2026-07-15, before inspecting any v12.6 portfolio return.

## Motivation

v12.5's 30-day same-coin Bybit-long/Binance-short funding-spread portfolio had
positive gross carry and positive descriptive turnover-adjusted return, but the
weekly full-liquidation cost gate failed. Its selected names were persistent:
mean weekly name-weight turnover was about one half rather than a full round
trip. v12.6 tests whether a frozen hold-band rule can turn that persistence into
an executable portfolio without changing the alpha score or venue direction.

## Frozen data and return

The v12.5 weekly symbol panel, chronology, symbol mapping, funding normalization,
and pair return are reused without alteration. At Monday 00:00 UTC, only the
preceding 30-day cumulative funding spread `Binance funding - Bybit funding` is
available. Each pair remains one dollar long Bybit perpetual and one dollar
short Binance USD-M perpetual, with weekly return:

`Bybit perp return - Binance perp return + Binance funding - Bybit funding`.

## Frozen portfolio

`TG1_30D_TOP9_HOLD18` is the only promotable candidate.

- Start with the top nine positive 30-day funding spreads.
- At each later weekly decision, retain a held name only if it remains in the
  current frozen monthly membership, its spread is positive, and its rank is
  no worse than 18.
- Fill open slots from the highest-ranked positive names not already held until
  nine names are held. If fewer than nine positive names exist, no portfolio is
  emitted for that week and the next valid week starts afresh.
- All names are equal weighted. There is no basis filter, volatility gate,
  score threshold beyond positivity, discretionary buffer, or venue reversal.

## Costs and controls

- Primary cost is 20 bp one-way times realized name-weight turnover, including
  initial entry and terminal close.
- Stress cost is 40 bp one-way times the same turnover path.
- Gross, funding-only, price-basis, and both net returns are reported.
- Development, validation, and holdout boundaries remain 2026-01-01 and
  2026-04-01.
- 2,000 week-block bootstrap resamples use primary net return.
- 1,000 within-week random positive-30-day-spread baskets of nine names use the
  observed candidate's exact weekly cost path. This deliberately removes any
  null advantage or penalty from different turnover.

Promotion requires at least 40 weeks, ten active months, ten validation weeks,
and eight holdout weeks; positive primary net return in development, validation,
and holdout; positive full-sample stress net return and funding contribution;
positive bootstrap 95% lower bound; null percentile at least 90; positive-month
contribution no greater than 35%; worst period mean no worse than -40 bp/week;
and mean weekly realized turnover no greater than 0.50.

Passing means forward shadow candidacy only. No PaperLive strategy is modified.

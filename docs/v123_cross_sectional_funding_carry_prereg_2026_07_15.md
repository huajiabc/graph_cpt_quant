# v12.3 Cross-Sectional Funding Carry Preregistration

Date frozen: 2026-07-15, before inspecting any v12.3 portfolio return.

## Mechanism

Funding has previously been used as a crowding feature, but not tested as a
direct multi-coin cash-flow portfolio. This study asks whether persistent
cross-sectional funding dispersion can pay for a slow, market-neutral bucket:
long contracts with the lowest trailing funding and short contracts with the
highest trailing funding, collecting the subsequent settled funding while
holding for one week.

## As-of data

- Frozen v11.0 monthly 8-by-9 memberships.
- Bybit settled funding records. A decision at `t` may use settlements strictly
  before `t`; a settlement stamped exactly `t` is excluded from the score.
- Existing 15-minute close panel. Weekly entry is Monday 00:00 UTC and exit is
  the close exactly seven days later. The funding PnL includes settlements in
  `(entry, exit]`.
- Monthly BTC betas are estimated from the preceding 30 days of hourly returns.

The formal period begins 2025-08-04 and ends at the last complete seven-day
label. Missing symbols are not replaced inside a week.

## Frozen portfolios

1. `FC1_7D_FUNDING_CARRY`: rank the preceding seven-day sum of settled funding;
   long the bottom nine and short the top nine.
2. `FC2_30D_FUNDING_CARRY`: rank the preceding 30-day mean settled funding;
   long the bottom nine and short the top nine.
3. `FC3_COMMUNITY_NEUTRAL_CARRY`: inside each eligible frozen community, long
   its lowest and short its highest seven-day funding symbol. Combine all eight
   community pairs equally.

Long and short sleeves each carry 0.5 absolute weight. No score threshold,
volatility gate, sign selection, or horizon selection is allowed.

For weight `w` (`w > 0` long, `w < 0` short), funding PnL is
`-w * sum(settled funding rate)` over the holding interval. Total gross return
is price return plus funding PnL. Price-only, funding-only, and BTC-residual
price-plus-funding components are reported separately.

## Costs and chronology

- Conservative round-trip cost: 40 bp per weekly portfolio.
- Stress cost: 60 bp.
- Descriptive realized-turnover cost: 20 bp one-way, including initial entry
  and terminal close.
- Development: through 2025-12-31; validation: 2026-01-01 through 2026-03-31;
  holdout: 2026-04-01 onward.

## Controls and promotion

- 2,000 week-block bootstrap resamples.
- 500 within-week random bucket rankings for FC1/FC2.
- 200 random monthly 8-by-9 partitions for FC3, applying the same within-group
  high/low funding rule.
- Reverse-carry diagnostics are reported but cannot be promoted.

Promotion requires at least 40 complete weeks, ten active months, ten
validation weeks, and eight holdout weeks; positive net 40 bp in development,
validation, and holdout; positive full-sample net 60 bp and BTC-residual net
40 bp; positive funding contribution; positive bootstrap 95% lower bound;
null percentile at least 90; positive-month contribution no greater than 35%;
and worst period mean no worse than -40 bp per week.

No existing PaperLive strategy is modified.

# v12.2 Positioning Absorption and Impulse Preregistration

Date frozen: 2026-07-15, after v12.1 rejected level-based top-trader rotation
and before any v12.2 candidate return is inspected.

## Question

The v12.1 result does not rule out changes in inventory. It only rejects the
level of large-trader positioning as a four-hour rotation signal. This study
tests a distinct microstructure mechanism: do large traders absorb aggressive
flow in the opposite direction, or does a fresh positioning impulse propagate
through a multi-coin bucket or frozen price community?

## Data and as-of contract

- The frozen v11.0 monthly 8-by-9 communities and existing Bybit feature panel.
- Binance USD-M five-minute metrics archives acquired for v12.1.
- At decision time `t`, a metrics row stamped `t - 5 minutes` is admissible;
  a row stamped `t` is not. The one-hour taker statistic is the mean log taker
  long/short volume ratio from `[t - 60 minutes, t - 5 minutes]` inclusive.
- The position and OI snapshots are the final admissible observations in that
  window. Four-hour changes use snapshots at `t` and `t - 4 hours`, both under
  the same availability rule.
- Four-hour candidates decide at 00:00, 04:00, ..., 20:00 UTC. Twelve-hour
  candidates decide at 00:00 and 12:00 UTC. Holdings do not overlap.

No missing symbol is replaced. The general direct bucket requires 48 covered
symbols. The OI-filtered bucket requires 24 covered symbols. A frozen community
requires six covered members.

## Causal features

For symbol `i` and time `t`:

- `D = log(top-trader position ratio / all-account ratio)`;
- `delta_D_4h = D(t) - D(t - 4h)`;
- `T_1h = mean(log(taker long/short volume ratio))` over the admissible hour;
- `delta_OI_4h = log(OI value(t)) - log(OI value(t - 4h))`.

Each of `delta_D_4h`, `T_1h`, and `delta_OI_4h` is standardized per symbol
against the preceding 30 days of hourly observations, shifted by one hour,
with at least 20 days of history and clipping to `[-5, 5]`.

The absorption score is:

`A = z(delta_D_4h) - z(T_1h)`.

A high score means large-trader inventory moved long while aggressive taker
flow was relatively short; the preregistered interpretation is bullish
absorption. A low score is the symmetric bearish case. The sign is fixed and
will not be chosen from development returns.

## Candidate families

1. `AB1_4H_INVENTORY_ABSORPTION`: every four hours, long the top nine `A`
   symbols and short the bottom nine for four hours.
2. `AB2_12H_INVENTORY_ABSORPTION`: at 00:00/12:00 UTC, the same top-nine versus
   bottom-nine portfolio held for twelve hours.
3. `AB3_12H_OI_CONFIRMED_ABSORPTION`: the AB2 portfolio restricted to symbols
   with `z(delta_OI_4h) >= 0`, requiring 24 symbols.
4. `PI1_12H_POSITIONING_IMPULSE`: long the top nine `z(delta_D_4h)` symbols and
   short the bottom nine for twelve hours.
5. `CA1_12H_COMMUNITY_ABSORPTION`: rank eligible frozen communities by median
   `A`; long all covered members of the highest community and short all covered
   members of the lowest for twelve hours.

Weights are `+0.5` across the high sleeve and `-0.5` across the low sleeve.
Gross exposure is one and net market exposure is zero.

## Chronology and costs

- Development is reported but cannot choose direction: 2025-08 through
  2025-12.
- Validation: 2026-01 through 2026-03.
- Holdout: 2026-04 onward.
- Primary conservative cost: 40 bp per completed holding period.
- Stress cost: 60 bp.
- Descriptive realized-turnover result: 20 bp one-way, including initial entry
  and terminal close. It cannot replace the conservative promotion gate.

## Nulls and promotion

- 2,000 day-block bootstrap resamples.
- 200 within-timestamp random bucket permutations for AB1-AB3 and PI1.
- 100 random monthly nine-symbol community partitions for CA1.
- Raw and trailing-BTC-beta-residual returns, IC, period stability, worst
  holding return, drawdown, coverage, and positive-month concentration are
  reported.

Promotion requires at least 500 decisions and ten months; positive net 40 bp
in development, validation, and holdout; positive full-sample net 60 bp and
BTC-residual net 40 bp; positive bootstrap 95% lower bound; null percentile at
least 90; positive-month concentration no greater than 35%; and worst period
mean no worse than -40 bp. No threshold, score, bucket size, horizon, sign, or
cost may be changed after candidate returns are inspected.

Existing PaperLive strategies are not modified by this study.

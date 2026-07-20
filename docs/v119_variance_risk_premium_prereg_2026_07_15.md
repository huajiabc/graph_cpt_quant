# v11.9 BTC Variance Risk Premium Preregistration

Date: 2026-07-15

Status before outcome inspection: `PREREGISTERED`.

## Question

Does the 30-day option-implied variance embedded in BTC DVOL exceed the next
30 days of realized BTC variance consistently enough to support a direct
short-volatility research sleeve?

This audit does not pretend that a futures breakout earns volatility. It tests
the economic object a delta-hedged option or variance position is intended to
harvest.

## Frozen data and sampling

- Hourly Deribit BTC DVOL index.
- Hourly Deribit BTC perpetual closes.
- One signal at 08:00 UTC on the first calendar day of each month.
- Each signal requires at least 30 prior days and 30 future days of hourly BTC
  returns, at least 700 finite hourly observations on each side, and a DVOL
  observation no more than two hours old.
- The future window begins strictly after the signal. No overlapping daily
  signals are created.

Annualized 30-day realized volatility is
`sqrt(sum(hourly_log_return^2) * 365 / 30) * 100`.

## Frozen candidates and payoff proxy

`VRP1_MONTHLY_SHORT_VARIANCE` shorts variance every covered month.

`VRP2_RICH_IV_SHORT_VARIANCE` shorts only when current DVOL exceeds trailing
30-day realized volatility by at least five volatility points.

The normalized gross variance payoff is
`(IV^2 - future_RV^2) / IV^2`, with IV and RV in decimal units. A positive
number favors short variance. This is a variance-swap-style research proxy,
not an account return and not a leverage recommendation.

Transaction and replication uncertainty are represented by reducing the
effective variance strike by one volatility point (`net_1vol`) and two
volatility points (`net_2vol`) before computing the payoff.

## Frozen splits and controls

- Development: signals through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- Holdout: signals from 2025-07-01 onward.
- Controls: long-variance direction, one-month stale DVOL strike, 2,000 random
  permutations of DVOL across the same realized-volatility outcomes, 5,000
  monthly bootstrap draws, chronological fifths, worst month, and cumulative
  normalized maximum drawdown.

## Gate

VRP1 requires at least 24 full, six validation, and eight holdout months. VRP2
requires at least 12 full, four validation, and four holdout months. Both must
have positive `net_1vol` in full, validation, and holdout; positive `net_2vol`
in full and holdout; a positive 95% bootstrap lower bound; at least the 95th
random-IV percentile; a worst `net_1vol` month above -100%; and no single
positive month above 35% of positive payoff.

Even a pass cannot go to PaperLive because the proxy omits actual option skew,
discrete delta hedging, bid/ask, margin, and liquidation. A pass only permits a
second-stage executable options reconstruction or a live DVOL/options spread
recorder.

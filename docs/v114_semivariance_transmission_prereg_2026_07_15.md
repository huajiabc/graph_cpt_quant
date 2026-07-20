# v11.4 Sign-Specific Semivariance Transmission Preregistration

Date: 2026-07-15

Status: frozen after v11.3 was rejected and before the first v11.4 outcome run. Research only.

## Motivation

v11.3 showed that absolute volatility expansion after receiver compression is real but not
graph-specific, and a direction-free OCO breakout has no gross edge. Absolute shocks may erase
the only monetizable information: whether the propagating tail is upside or downside. v11.4 tests
that separate nonlinear mechanism without changing the v11.3 thresholds after seeing PnL.

## Frozen construction

- Use the same 73-symbol 15-minute panel and chronological labels as v11.3.
- At each month start, estimate static BTC beta and residual scale from the strictly preceding
  30 days.
- Build two nonnegative shock panels:
  - upside semivariance shock: `max(residual / scale, 0)`;
  - downside semivariance shock: `max(-residual / scale, 0)`.
- Separately for each sign, estimate 15/30/60-minute directed shock correlations, require positive
  forward-minus-reverse advantage, and retain three leaders per follower.

## Frozen event and trade

At completed hourly timestamps, a follower is eligible when its same-sign weighted leader score
exceeds the trailing-month 95th percentile, at least two of three leaders exceed their own
trailing-month 90th-percentile same-sign shock, the follower's own last-hour same-sign realized
semivolatility is below its trailing-month median, and the transmission-gap rank is at least the
80th percentile.

Select up to five followers and require two. Apply a four-hour cooldown per sign.

- `SVT1_DOWNSIDE_CASCADE`: short the equal-weight receiver bucket for four hours.
- `SVT2_UPSIDE_CASCADE`: long the equal-weight receiver bucket for four hours.

Primary return is raw four-hour futures return after 20 bp round-trip cost. Net 30 bp and net
50 bp are stress outcomes. BTC-residual return is attribution only. No stop, take-profit, OCO,
leverage, or post-result holding-period search is allowed in this version.

## Controls and gates

- one-day shifted signal state;
- 50 random directed graphs per sign, preserving follower edge count, lag, and edge-weight slots;
- random-family maximum across the two signs;
- entry-day block bootstrap, five chronological slices, month and receiver concentration;
- development/validation/holdout-label reporting.

Forward-watch eligibility requires at least 100 observations, at least 25 validation and 25
holdout observations, positive validation and holdout net 20 bp, positive full net 30 bp, at least
the random-family 90th percentile, better than the one-day shift, positive bootstrap lower bound,
five non-negative chronological slices, and no more than 35% positive PnL concentration in one
month or receiver.

Even a pass remains retrospective research and cannot change PaperLive or real-order permission.

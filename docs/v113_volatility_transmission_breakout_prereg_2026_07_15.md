# v11.3 Directed Volatility-Transmission Breakout Preregistration

Date: 2026-07-15

Status: frozen before the first v11.3 outcome run. Research only. The remote v11.2
PaperLive observer remains unchanged.

## Question

Can a graph predict *where volatility will expand next*, even though earlier signed-return
lead/lag graphs did not predict economically useful return direction? If so, can that
direction-free forecast be monetized with an executable range-break OCO rule?

## Data and as-of boundary

- Bybit 15-minute OHLC and returns from the existing 73-symbol research panel.
- Each target month uses only the trailing 30 calendar days ending strictly before month start.
- Static BTC beta, residual scale, directed edges, leader-shock thresholds, and receiver
  compression thresholds are frozen at month start.
- Development: before 2026-01; validation: 2026-01 through 2026-03; holdout label: 2026-04 onward.
  These are chronological labels, not a claim of a fresh untouched holdout after prior graph work.

## Frozen graph

1. Remove static trailing-month BTC beta from each coin's 15-minute return.
2. Convert residual returns to absolute standardized shocks using trailing-month scale.
3. For 15, 30, and 60 minute lags, estimate `leader_abs_shock(t) ->
   follower_abs_shock(t+lag)` correlation.
4. Keep an edge only when its positive forward correlation exceeds the reverse direction.
5. Retain the strongest three leaders per follower. No target-period edge fitting is allowed.

## Frozen event

Evaluate only completed hourly timestamps. A follower is eligible when:

- its weighted leader-shock score is above its trailing-month 95th percentile;
- at least two of its three leaders are individually above their trailing-month 90th-percentile
  absolute-shock threshold;
- its own trailing one-hour residual realized volatility is below its trailing-month median;
- its cross-sectional transmission-gap rank is at least the 80th percentile.

At a timestamp, select up to five followers by transmission gap. Require at least two. Apply a
four-hour portfolio cooldown.

## Frozen monetization rule

For each selected receiver:

- reference range: high/low of the four completed 15-minute bars ending at the signal;
- entry window: the next four 15-minute bars;
- enter long at the known reference high if it breaks first, or short at the known reference low
  if it breaks first;
- if both boundaries first trigger in the same bar, mark the leg ambiguous and do not trade it;
- exit at the close of the bar ending four hours after the signal;
- no stop, take-profit, trailing rule, or post-result parameter tuning;
- portfolio return is the equal-weight mean of filled, unambiguous legs; require two filled legs.

The single candidate is `VTB1_VOL_RECEIVER_OCO`. Primary outcome is four-hour breakout return
after 20 bp round-trip cost. Net 30 bp and net 50 bp are stress outcomes.

## Mechanism outcomes

Before considering PnL, report:

- future four-hour residual realized volatility divided by prior four-hour residual realized
  volatility;
- OCO fill rate and same-bar ambiguity rate;
- filled long/short balance;
- event, day, month, and receiver-symbol coverage.

## Controls and gates

- one-day shifted event state;
- 50 random directed graphs preserving follower edge count, lag, and edge-weight slots;
- entry-day block-bootstrap confidence interval;
- five chronological slices;
- positive-PnL month and receiver concentration;
- explicit 20/30/50 bp cost stress.

Forward-watch eligibility requires all of the following: at least 100 portfolio observations,
at least 25 validation and 25 holdout observations, positive validation and holdout net 20 bp,
positive full-sample net 30 bp, real result at or above the random-control 90th percentile,
real result above the shifted control, positive bootstrap lower bound, all five chronological
slices non-negative, positive future-volatility expansion in validation and holdout, and no more
than 35% positive PnL concentration in one month or receiver.

Even a formal pass remains retrospective research. It does not authorize PaperLive, leverage,
or real orders.

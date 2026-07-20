# v14.1 Directed Taker-Flow Graph Preregistration

Date frozen: 2026-07-15, before constructing or inspecting any v14.1 edge, event, or return.

## Distinct hypothesis

Earlier high-frequency graph studies used price returns or open-interest shocks as the source
variable. They did not have the subsequently acquired continuous Binance USD-M five-minute taker
long/short-volume archive. v14.1 tests a distinct directed edge:

`leader aggressive-flow shock at t -> follower future BTC-residual return`.

This remains single-venue flow history and does not replace the separately frozen synchronized
Binance/Bybit forward-tape study v10.7.

## Causal data contract

- Frozen monthly universe: v13.2's causally generated memberships through July 2026, excluding BTC
  as a follower and any symbol without Binance metrics.
- Binance source rows use `sum_taker_long_short_vol_ratio > 0`. A 15-minute decision at `t` may
  use only rows stamped strictly before `t`; source value is the mean log ratio in `[t-15m, t)`.
- Per-symbol source z-score uses the strictly preceding seven days of 15-minute values, shifted one
  bar, with at least five days of history and clipping to `[-5, 5]`.
- Bybit target prices are fully closed 15-minute bars. Future one-hour and four-hour returns use
  closes at `t+1h` and `t+4h`.
- Monthly BTC betas use only the preceding 30 days of hourly returns.

## Frozen directed graph

For each target month:

- train on the preceding 30 days, ending one hour before month start so every target is realized;
- sample source/target pairs only at hourly timestamps to reduce overlap;
- source is leader taker-flow z-score; target is follower future one-hour Bybit return after
  subtracting frozen BTC beta times BTC future one-hour return;
- require 500 common observations;
- edge score is positive Spearman source/target correlation minus the reverse-pair correlation,
  multiplied by `sqrt(n/(n+500))`;
- retain the strongest three positive-advantage leaders per follower.

## Frozen candidates

1. `TFG1_POSITIVE_FLOW_PROPAGATION`: leader flow z-score at least +2 and leader trailing-15-minute
   Bybit return positive; propagate long pressure.
2. `TFG2_NEGATIVE_FLOW_PROPAGATION`: leader flow z-score at most -2 and leader trailing-15-minute
   return negative; propagate short pressure.

A follower activates only when at least two of its frozen leaders are simultaneously active.
At least three followers must activate; hold the top five equal weighted in the candidate direction.
Use a candidate-level one-hour cooldown, so primary one-hour holdings do not overlap.

Primary return is the one-hour follower bucket plus causal BTC hedge after 40 bp total round-trip
cost. Naked bucket net20/net30 is secondary. Four-hour raw/residual return is a preregistered
secondary-frequency diagnostic and cannot promote v14.1.

## Controls and promotion

- Fifty within-month random-leader graphs preserve follower indegree and edge-weight slots.
- A reversed-edge graph and a one-day (96-bar) delayed source signal are mandatory controls.
- Two-thousand entry-day block bootstrap draws use primary residual net40.
- Because two directions are tested, compare each real candidate against the per-iteration random
  family maximum and require at least the 95th percentile.
- Promotion requires at least 200 full, 50 validation, and 50 holdout observations; positive
  primary residual net40 and naked net20 in development, validation, and holdout; positive naked
  net30; positive bootstrap lower bound; real mean above reversed and delayed controls; positive
  month contribution concentration at most 35%; and worst period mean no worse than -40 bp.

Passing means forward-shadow candidacy only. No PaperLive, leverage, or live-order permission is
granted by this retrospective test.

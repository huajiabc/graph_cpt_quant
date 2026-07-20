# v15.3 Crowded-Source Exhaustion Reversal Preregistration

Date frozen: 2026-07-16, before inspecting any source-coin future return.

## Distinct question

v11.8 tested whether a crowded leader's unwind propagates to other members of a
frozen price community. It did not trade the leader itself. v12.2 tested
large-trader inventory absorption rather than account crowding plus simultaneous
price/OI liquidation. This study asks whether the shocked source coin reverses
after forced inventory has exited.

## Frozen event definition

Reuse the exact causal v11.8 hourly z-scores and thresholds:

- crowded-long unwind: crowding z >= +2.0, one-hour return z <= -1.5 and
  one-hour OI-value-delta z <= -1.0;
- crowded-short squeeze: crowding z <= -2.0, one-hour return z >= +1.5 and
  one-hour OI-value-delta z <= -1.0.

All rolling statistics use the prior 30 days, shifted one hour, with at least
20 days of history and clipping to +/-5. Account-ratio observations must be no
more than 90 minutes old.

## Only candidate

`LR1_CROWDED_SOURCE_EXHAUSTION_REVERSAL`

- At a signal hour, long every crowded-long-unwind source and short every
  crowded-short-squeeze source.
- If both sides fire, assign 0.5 raw gross to each side; if only one side fires,
  assign raw gross one to that side.
- Equal weight within side, add the exact prior-month BTC-beta hedge, and
  normalize total gross notional to one.
- Hold for the already defined next four hours. Enforce one global four-hour
  cooldown, combining simultaneous source events and preventing overlap.
- Every isolated event portfolio is opened and closed completely: primary cost
  is 20bp one-way times L1 turnover two (40bp round trip); stress cost is 80bp.
- Funding is not modeled separately at this horizon; the 80bp stress result is
  mandatory and no leverage is considered.

The exact reversed position is a negative control and cannot be promoted.

## Controls and promotion

- Repeat the candidate with crowding shifted by 24 hours while keeping current
  price/OI conditions.
- 500 random-source paths preserve every event's month, timestamp, long/short
  counts, beta hedge and cost, sampling from the exact frozen monthly universe.
- 2,000 day-block bootstrap draws resample complete UTC event days.

Promotion requires at least 200 event portfolios, ten months, 50 validation and
50 holdout portfolios; positive primary return in development, validation and
holdout; positive full-sample stress return; positive bootstrap lower bound;
random-source percentile at least 95; positive-month concentration no greater
than 35%; worst period at least -40bp/event; maximum residual beta and gross
drift within 1e-12; and the main mean above both reversed and 24-hour-shifted
controls.

Passing means a raw forward-shadow candidate only. No PaperLive, leverage,
remote-host or real-order permission is granted.

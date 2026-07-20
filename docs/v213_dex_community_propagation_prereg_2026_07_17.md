# v21.3 DEX Community-Propagation Preregistration

Status: `frozen_before_return_reveal`.

This reveal tests whether a token-level DEX volume-attention event, confirmed by a
causally observed CEX price innovation, propagates into *other* members of the
token's forward-frozen monthly graph community.  The v21.2 audit selected the
rules using feature coverage only; no post-event portfolio return was inspected.

## Frozen candidates

- `DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION`: after the v21.2 source confirmation
  and four-hour community cooldown, trade every other available community member
  in the source token's one-hour return direction.  At least four peers are
  required.
- `DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS`: under the same event rule, rank peers by
  their already observed one-hour return in the source direction and trade the
  slowest half, with at least three names.

The DEX record becomes usable only at `event_available_time`.  The source feature
is the first 15-minute close strictly afterward.  Entry is the following
15-minute close, so signal observation precedes execution by a complete bar.

## Portfolio and return

- Peers are equal-weighted in the source direction.
- Each symbol's BTC beta is estimated from the 30 calendar days strictly before
  the event month, requiring at least 2,000 paired 15-minute observations.
- A BTC hedge offsets estimated portfolio beta; all weights are then normalized to
  unit gross notional.
- Primary holding period: 12 hours (48 bars).
- Book costs: 20 bp round trip primary and 40 bp stress, charged against unit gross.
- No DEX price, source-token return after entry, funding payment, leverage, or
  overlapping-strategy PnL is credited.

## Frozen chronology

- Development: 2025-08-01 through 2025-11-30.
- DEX vendor transition: 2025-12-01 through 2026-02-28.  These four feature events
  remain reported but are excluded from eligibility statistics.
- Validation: 2026-03-01 through 2026-04-30.
- Holdout: 2026-05-01 onward.

## Controls

- 15-minute additional entry delay.
- Four-hour and 24-hour holding horizons.
- Reversed direction at the primary timing.
- A +24-hour event-time placebo using the same frozen names and direction.
- Source-token-only beta-neutral response.
- 500 random baskets per event: global monthly graph-universe baskets for DAP1 and
  same-community peer baskets for DAP2, preserving selection count, event time,
  direction, beta hedge, and cost convention.
- 2,000 UTC-day block-bootstrap paths.
- Month and source-token contribution concentration.

## Eligibility gates

A candidate is only eligible for later natural-forward shadow observation if all
of the following hold:

1. At least 200 realized events, at least 30 in each of development, validation,
   and holdout, and at least eight active months.
2. Mean primary gross return exceeds 20 bp and primary net return is positive in
   all three chronological periods.
3. Delayed-entry primary net return is positive; reversed-direction primary net
   is negative.
4. The relevant random-control percentile is at least 0.95 and the day-block
   bootstrap lower 95% bound of primary net return is positive.
5. The +24-hour placebo primary net return is non-positive.
6. The candidate's gross return exceeds the source-token-only gross response.
7. No month or source token contributes more than 35% of positive aggregate
   primary net PnL.

Passing is research eligibility only.  It does not authorize PaperLive, live,
application, leverage, remote, or order changes.

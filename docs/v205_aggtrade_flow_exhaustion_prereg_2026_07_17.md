# v20.5 AggTrade Flow-Exhaustion Preregistration

Status: `frozen_before_post_event_return_reveal`.

This branch is a post-hoc offline follow-up to the rejected v20.1/v20.3
community trade-overshoot family. Even if it passes every offline gate, it is
natural-forward-only and cannot be promoted from this reveal.

## Frozen inputs

- Raw event-window features: `event_window_features.parquet`, SHA256
  `78A1A1FFB3DDFB938F5DB4F646F97F8AD52009EB7BFCA6DA14325F30E4CB7E2C`.
- v20.4 receiver features: `receiver_features.parquet`, SHA256
  `6B6418B3AB9AC2337C56F64912B28A82C596BDA3F7097F1491DD2BB74E4DD64F`.
- v20.4 candidate feature events: `candidate_feature_events.parquet`, SHA256
  `A83BCACFBD6FBDC7040E83800C50301C9933A66D5DF6CF348BAF68046D799712`.
- Candidate event counts are frozen at 53 for RFX1 and 52 for RFX2.
- The source universe remains the v20.0 community coherent index shock at
  `z2.0`, trade-vs-mark receiver overshoot threshold `2.0`, and at least three
  original receivers.

## Frozen candidates

### RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE

An event qualifies when at least half of its quality-screened original
receivers have final-third taker-flow imbalance opposite to the graph source
direction. At the feature timestamp, fade every quality-screened original
receiver equally before the BTC beta hedge. Receiver direction is
`-source_sign`.

### RFX2_EXHAUSTED_VS_PERSISTENT_FLOW_SPREAD

Strict exhausted receivers have positive early-third source-aligned imbalance
and negative late-third source-aligned imbalance. Persistent receivers have
positive early- and non-negative late-third source-aligned imbalance. An event
requires at least one strict exhausted receiver and at least two persistent
receivers. Allocate 50% raw gross to fading the exhausted bucket and 50% raw
gross to following the persistent bucket, equal-weighted within each side.

## Portfolio and timing

- Entry is the frozen feature timestamp; primary exit is one 15-minute bar
  later.
- Monthly BTC betas use only the prior 30 days with at least 2,000 observations.
- Add a BTC hedge that offsets prior-window beta, then normalize total gross
  notional, including the hedge, to one.
- Primary round-trip book cost is 20 bp; stress cost is 40 bp.
- Report the gross break-even cost and fixed 5/10/20/40 bp cost frontier.
- Auxiliary robustness checks use one-bar delayed entry and 30/60-minute
  holding horizons. They do not replace the primary horizon.

## Frozen controls

- Reversed-direction primary PnL.
- RFX1: 500 period-matched random selections from all 216 frozen extreme
  community events, preserving the observed development/validation/holdout
  event counts and using the same full-receiver fade book.
- RFX2: 500 within-event random bucket assignments, preserving each event's
  exhausted and persistent bucket sizes.
- Seed: `20500`.
- Nonparametric event bootstrap: 2,000 draws.

## Decision gates

Each candidate must independently satisfy all of the following:

1. At least 45 total events and at least five events in every frozen period.
2. Mean gross return exceeds the 20 bp primary cost.
3. Primary net mean is positive in development, validation, and holdout.
4. One-bar delayed primary net mean is positive.
5. The observed gross mean is at or above the 95th percentile of its frozen
   random control.
6. Reversed-direction primary net mean is negative.
7. The 95% event-bootstrap confidence interval for primary net mean has a
   positive lower bound.

Failure of any gate rejects that candidate. Passing only labels it an offline
research candidate requiring untouched natural-forward evidence.

At preregistration time, no post-event price, candidate PnL, live, PaperLive,
application, leverage, remote, or order state was read or changed.

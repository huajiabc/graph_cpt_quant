# v20.7 Unhedged Flow-Exhaustion Diagnostic Preregistration

Status: `posthoc_decomposition_followup_frozen_before_unhedged_recomputation`.

This is not a clean independent reveal. The v20.5 beta-hedged RFX1 result and
its alt/BTC contribution split are already known. The sole purpose of this
diagnostic is to decide whether the frozen flow-exhaustion event filter merits
untouched natural-forward observation, or whether its apparent effect is only
BTC/market reversal.

## Frozen sample and candidate

- Use exactly the 53 frozen
  `RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE` feature events from v20.4.
- Candidate feature file SHA256:
  `A83BCACFBD6FBDC7040E83800C50301C9933A66D5DF6CF348BAF68046D799712`.
- `RFX3_UNHEDGED_EVENT_WIDE_RECEIVER_FADE`: equal-weight every frozen
  quality-screened receiver with direction `-source_sign`; total receiver gross
  is one and there is no BTC hedge.
- `BTC1_EVENT_MATCHED_SOURCE_FADE_CONTROL`: on the same 53 events, hold only
  BTC with direction `-source_sign` and unit gross. This is a control, not a
  promotable candidate.

## Frozen timing, costs, and controls

- Primary entry is the feature timestamp and exit is one 15-minute bar later.
- Charge the same conservative 20 bp primary and 40 bp stress round-trip cost
  to both books.
- Report the candidate's 5/10/20/40 bp cost frontier.
- Report one-bar delayed entry and 30/60-minute holding horizons.
- Build 500 period-matched random event selections from all 216 frozen extreme
  community events, preserving RFX3's 30/15/8 period counts and using the same
  unhedged full-receiver fade book.
- Use 2,000 nonparametric event bootstraps with seed `20700`.
- No receiver, threshold, period, holding time, or hedge-ratio grid is allowed.

## Diagnostic gates

All gates must pass to retain RFX3 for natural-forward observation:

1. Exactly 53 events with 30/15/8 development/validation/holdout coverage.
2. Mean RFX3 gross return exceeds the 20 bp primary cost.
3. RFX3 primary net mean is positive in every frozen period.
4. One-bar delayed RFX3 primary net mean is positive.
5. RFX3 gross mean is at or above the 95th percentile of its period-matched
   random-event control.
6. The 95% bootstrap lower bound of RFX3 primary net mean is positive.
7. Mean paired `RFX3 gross - BTC1 gross` is positive, and its 95% bootstrap
   lower bound is also positive.

Failure of gate 7 means the effect is market/BTC reversal rather than new
receiver-bucket alpha. Passing every gate still provides no promotion evidence;
it only justifies untouched natural-forward monitoring.

No new post-event return calculation was performed while freezing this
document. No live, PaperLive, application, leverage, remote, or order state was
read or changed.

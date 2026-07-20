# v20.3 Community Peer-Hedge Posthoc Preregistration

Status: frozen after the v20.2 feature-only audit and before inspecting the
future return of the peer-hedged book. This branch is posthoc because v20.1
alt/BTC sleeve attribution motivated the hedge redesign.

## Frozen construction

Reuse only `RPT4_COMMUNITY_TRADE_OVERSHOOT_FADE` events frozen before the v20.1
reveal:

- community median index-return z-score absolute value at or above 2.0;
- receiver trade-vs-mark overshoot score at or above 1.5;
- receiver completed-bar return aligned at least 0.25 standard deviations with
  the community source;
- four-bar per-community cooldown, top eight receivers, and at least three.

For each event, selected overshoot receivers take the preregistered fade
direction, opposite the community source. Every other available non-BTC member
of that frozen monthly community forms the peer sleeve in the source direction.
Require at least two peers; do not substitute symbols. Selected and peer sleeves
each carry 0.5 absolute notional, producing exact dollar neutrality and gross
notional one. No BTC hedge is added.

The v20.2 feature audit retained 237 of 360 original events: 108 development, 59
validation, and 70 holdout events across 11 months. All 123 exclusions have
fewer than two unselected peers. Median selected/peer counts are 3/3. Median and
90th-percentile absolute prior BTC-beta exposures are 0.0483 and 0.1447; beta is
a reported risk diagnostic, not a selection input or a neutralization target.

## Frozen candidate and execution

Candidate: `RPH1_COMMUNITY_TRADE_OVERSHOOT_PEER_HEDGE`.

Primary holding is one 15-minute bar. Entry uses the futures close at the
completed feature timestamp (the next bar opening boundary); exit uses the next
completed close. Primary/stress round-trip costs are 20/40 bp on the normalized
book. Selected-sleeve and peer-sleeve gross contributions and the gross
break-even cost are reported separately.

## Frozen controls and diagnostics

- Exact reversed direction.
- One-bar delayed entry.
- Two- and four-bar holding diagnostics.
- Original v20.1 BTC-hedged RPT4 result as a known posthoc benchmark, not a null.
- Selected-only directional sleeve as attribution only, not a promotion route.
- 500 deterministic same-event random partitions: choose the same number of
  selected symbols uniformly from the same frozen community, use the complement
  as peers, retain the same source directions and 0.5/0.5 sleeve notionals.
- Day-block bootstrap with 2,000 iterations.
- Positive/negative source sides, monthly concentration, symbol contribution,
  dollar exposure, gross notional, and prior-beta exposure diagnostics.

## Frozen gates

- At least 200 full, 50 validation, and 60 holdout events across ten months.
- Positive primary mean in development, validation, and holdout.
- Positive full stress mean and day-block bootstrap 95% lower bound.
- At or above the random-partition 95th percentile.
- Beat reversed direction, one-bar delay, and the original BTC-hedged RPT4
  primary mean.
- Two- and four-bar holding diagnostics remain positive after primary cost.
- Both positive- and negative-source full-sample primary means are positive.
- Maximum dollar exposure and gross-notional drift are at most 1e-10.
- No single profitable month supplies more than 35% of positive monthly PnL and
  no symbol supplies more than 25% of positive gross contribution.

Even if every gate passes, the verdict can only be
`posthoc_offline_discovery_natural_forward_required`. It cannot authorize
PaperLive, live, application, leverage, remote, or real-order changes on this
historical sample.

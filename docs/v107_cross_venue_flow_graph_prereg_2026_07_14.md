# v10.7 Cross-Venue Flow Graph Preregistration

Date: 2026-07-14

Status: `DATA_ACCUMULATING`. This specification cannot produce an alpha verdict before the
data gate passes.

## Distinct hypothesis

The rejected v10.3-v10.6 graphs used price returns to define both edges and impulses. v10.7
uses synchronized aggressor flow as the source variable:

`leader Binance/Bybit taker-flow shock -> another symbol's future Bybit residual return`.

It is a cross-symbol downstream-bucket test, not another same-symbol Binance-to-Bybit lag test
and not a price-correlation graph.

## Frozen data contract

- Immutable one-minute Binance and Bybit public-trade aggregates from v9.6.
- Frozen initial 20-symbol universe from the recorder start.
- Earliest admissible minute: 2026-07-13 11:01:00Z.
- A symbol-minute is eligible only when both exchanges have complete bars, event lag is at most
  ten seconds, and the minute is present on both venues.
- No event-conditioned historical archive may be spliced into the forward tape.

## Data gate

No graph edge, strategy return, or alpha verdict is reported until all are true:

- at least 90 calendar days since the first admissible minute;
- at least three calendar months represented;
- at least 15 of the frozen 20 symbols remain usable;
- every evaluated full symbol-day has synchronized coverage >= 95%;
- at least 80 eligible full days per usable symbol.

Before that point the only allowed statuses are `DATA_ACCUMULATING`, `DATA_QUALITY_FAIL`, or
`DATA_UNAVAILABLE_LOCAL`. The earliest time-only date is 2026-10-11 11:01:00Z; quality and
sample gates can move the actual date later.

## Frozen edge construction

Starting after the first 30 complete days, rebuild the graph weekly using only the preceding
30 days.

For every leader/follower pair:

1. Source variable: leader's trailing-five-minute Binance imbalance, standardized against its
   strictly trailing seven-day history.
2. Source agreement: Binance and Bybit five-minute imbalances have the same sign.
3. Target variable: follower's next-one-hour Bybit return after removing its as-of BTC beta.
4. Edge score: positive rank correlation between source flow and follower residual return,
   minus the reverse-pair correlation.
5. Require at least 200 independent hourly source observations and positive directional
   advantage; apply sample shrinkage and retain three leaders per follower.

The graph is frozen for the following week.

## Frozen bucket signal

`CVFG1_POSITIVE_FLOW_PROPAGATION`:

- leader five-minute imbalance >= +0.15;
- leader flow z-score >= +2;
- both venues agree positive;
- leader five-minute turnover exceeds its strictly trailing seven-day 95th percentile;
- aggregate at least two active leaders for a follower;
- select up to five followers by edge-weighted source pressure and require at least three.

Entry is the next complete Bybit minute. The portfolio has a four-hour global cooldown.
Primary outcome is the equal-weight follower-bucket four-hour Bybit return after 20 bp total
round-trip cost. BTC-beta-neutral return after 40 bp is attribution-only.

The symmetric negative-flow state is logged as a diagnostic and cannot approve a short strategy
in this version.

## Controls and decision gates

Controls: within-week random leaders preserving indegree, reversed edges, source shifted one day,
same-symbol-only propagation, and a price-only edge graph.

After the 90-day gate, use days 31-60 as validation and days 61-90+ as untouched holdout. A
forward-watch candidate must have at least 200 entries, at least 50 in validation and holdout,
positive validation/holdout net20, positive 30 bp stress, random-family percentile >= 90%, beat
every frozen control, positive entry-day bootstrap lower bound, and month/symbol positive-PnL
shares below 35%.

No PaperLive, leverage, or live permission changes are authorized by this preregistration.

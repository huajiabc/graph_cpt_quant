# v10.2 Short-Squeeze Flow Persistence Findings

Status: `reject_single_venue_flow_persistence_branch` as a standalone alpha.
No PaperLive, live, leverage, sizing, or lifecycle status changed.

## Absolute strategy result

The frozen `short_squeeze + OF1_CONFIRM_LONG` candidate produced:

| Segment | n | raw net20 | raw net30 | BTC-hedged net40 |
|---|---:|---:|---:|---:|
| Full | 55 | +0.3893% | +0.2893% | +0.0374% |
| Development | 25 | +0.5125% | +0.4125% | -0.0804% |
| Validation | 16 | +0.1636% | +0.0636% | -0.1101% |
| Holdout | 14 | +0.4271% | +0.3271% | +0.3996% |

The shape is attractive but fails the third-look controls:

- raw net20 day-bootstrap 95% interval: -0.1783% to +0.9964%;
- path-specific same-symbol/same-day random percentile: 88.6%, below 95%;
- hedged matched-random percentile: 83.8%, below 90%;
- validation hedged net40 is negative.

## Selection-alpha result

Exact flow does carry a stronger relative-ranking clue inside short-squeeze
events. OF1 minus non-OF1 raw net20 lift was:

- full +0.5621%;
- development +0.3686%;
- validation +0.4011%;
- holdout +1.1450%;
- day-bootstrap 95% interval +0.0642% to +1.1012%.

This is the strongest new information found in the v10 branch. It says exact
active-buy coherence separates better from worse short-squeeze events. It does
not show that the selected sleeve has stable standalone absolute expectancy
after uncertainty, discovery correction, and beta-aware costs.

## Decision

Close the single-venue exact-flow branch as an independent strategy. Preserve
OF1 as a research-only ranking/attribution feature for genuinely new forward
short-squeeze observations. Do not optimize another historical threshold on
the same 55 selected events.

The next independent alpha opportunity remains synchronized cross-venue flow,
where venue lead-lag supplies new information rather than another transform of
the same Bybit event tape.

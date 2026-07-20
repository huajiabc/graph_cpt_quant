# v23.0--v23.9 Volatility Monetization Round Summary

## Scope and data boundary

This round tested whether the v22.4 multi-coin book-vacuum event can be converted
from a variance forecast into tradeable return. No live, PaperLive, leverage,
remote, application, or order state changed.

The local Deribit archive contains actual positive-volume option trade OHLCV but
not synchronized historical bids/asks. A causal preselected ATM call/put pair was
observable at entry for 61/159 events and at both entry and four-hour exit for
only 26 events, including four holdout events. Deribit's public historical mark
endpoint rejects the expired contracts. Therefore no historical executable
option-PnL claim is made.

## Branch A: causal-IV synthetic straddle

- v23.0 passed 14/14 feature checks and attached causal ATM IV to 123 events
  (46/44/33 development/validation/holdout).
- v23.1 found mean four-hour realized/implied variance of 1.4433 and a 100th
  matched-random percentile, confirming relative volatility information.
- The constant-IV ATM straddle earned only +20.35 bp gross on premium and -79.65
  bp after a 1% premium hurdle. Holdout gross was -17.99 bp.
- v23.2 passed 15/15 independent checks and validated rejection.

Conclusion: the event forecasts variance, but unconditional long options do not
clear the priced-volatility and friction hurdle.

## Branch B: unconditional BTC perpetual OCO

- v23.3 passed 13/13 feature checks on all 159 events with complete 16-bar
  15-minute paths. The frozen one-sigma barrier median was 43.44 bp.
- v23.4 one-sigma OCO triggered 149 times but returned -7.96 bp/event after 10
  bp cost; holdout was -22.16 bp and matched-random percentile 71.4.
- Width response improved toward two sigma. A development+validation scan froze
  two sigma for a new temporal confirmation.
- v23.6 two-sigma OCO returned +2.05 bp full sample and reached the 97.8th full
  matched-random percentile, but holdout was -9.52 bp and only the 34.7th
  holdout matched percentile. Month-bootstrap lower bound was -7.02 bp.
- v23.5 and v23.7 passed 14/14 and 12/12 audits respectively.

Conclusion: wider breakout confirmation contains relative structure but is not
an unconditional stable alpha.

## Branch C: positive-pressure narrow breakout

This is explicitly post-selection robustness, not an untouched holdout result.
The rule uses only information known at event time:

1. retain v22.4 events whose aggregate book-pressure direction is positive;
2. compute causal hourly sigma from the preceding 24 completed BTC hours;
3. place symmetric BTC OCO barriers at plus/minus 0.625 sigma;
4. use the first 15-minute trigger, pessimistic gap fill, pessimistic same-bar
   ambiguity, and exit at four hours;
5. subtract 10 bp round-trip cost, with 20 bp as stress.

Observed robustness:

- 53 events across 11 months; all 53 trigger and none is same-bar ambiguous.
- Primary return: +19.69 bp/event; 20 bp stress return: +9.69 bp/event.
- Development/validation/holdout: +10.44/+38.02/+14.69 bp/event.
- Matched-random percentiles: 94.2/96.6/91.1 by temporal split, 99.4 overall.
- Adjacent 0.75-sigma width remains positive in every temporal split.
- A 15-minute activation delay and both three- and four-hour exits remain
  positive in every temporal split.
- Leave-one-month-out minimum mean: +10.48 bp/event.
- Positive-minus-negative pressure difference: 51.68 bp; within-month sign
  permutation upper-tail p=0.0072; month-bootstrap sign-difference lower bound
  +25.47 bp.
- The absolute strategy month-bootstrap lower bound remains -9.08 bp.
- v23.9 passed 13/13 artifact checks with zero or floating-point-epsilon errors.

Status: `forward_shadow_candidate_not_statistically_confirmed`.

The candidate has useful economic magnitude, temporal split consistency,
matched-time evidence, adjacent-width stability, and execution sensitivities.
It still lacks a positive absolute month-block confidence lower bound, and its
filter/width were chosen after inspecting this sample. It must remain frozen and
collect genuinely new events after the current feature cutoff (2026-07-15 UTC)
before any promotion decision.

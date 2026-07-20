# v20.9-v21.1 Funding-Settlement Round Summary

Verdict: `settlement_rebound_rejected_persistent_carry_state_remains_distinct`.

## What was tested

This round isolated the price response *after* an already completed Binance USD-M
funding settlement.  It did not credit the funding payment itself and did not reuse
the existing weekly carry result as return.

- FSE1 selected every alt with a just-settled negative funding rate, with at least
  five names in the cross-section.
- FSE2 selected newly negative names whose current settled rate was negative and
  immediately prior settled rate was non-negative, again with at least five names.
- The signal was observed at 00/08/16 UTC settlement, entry waited one complete
  15-minute bar, and the primary holding period was 60 minutes.
- Portfolios were beta-neutral against BTC and charged 20/40 bp round-trip book
  costs.  Delayed entry, 30/120-minute horizons, reversed direction, 1,000 random
  same-settlement baskets, and UTC-day block bootstrap were frozen controls.

## Evidence

| Candidate | Events | Gross bp | Net at 20 bp | Random percentile | Day-bootstrap lower 95%, net bp |
|---|---:|---:|---:|---:|---:|
| FSE1 all negative | 813 | -0.1038 | -20.1038 | 0.8460 | -21.4601 |
| FSE2 new negative onset | 505 | 0.8322 | -19.1678 | 0.8380 | -21.0250 |

FSE1 gross results by development/validation/holdout were -0.1216, -1.8165,
and +0.9695 bp.  FSE2 produced +0.1624, -2.5820, and +3.8611 bp.  The apparent
FSE2 holdout improvement remained far below the 20 bp cost hurdle and was not
stable across settlement hours.  Delayed and alternate-horizon controls did not
repair either candidate.

## Audit and interpretation

The independent v21.1 audit passed 20/20 checks, including source prices, causal
beta estimates, timing, weights, PnL contributions, costs, period/hour summaries,
all random paths, block bootstrap, and rejection logic.

The short-horizon post-settlement response is therefore rejected as a standalone
alpha.  This does **not** invalidate the separately observed weekly negative-funding
state/carry branch: the evidence says its economics come from persistent funding
and positioning state, not from a mechanical rebound immediately after payment.

No live, PaperLive, application, leverage, remote, or order state was read or
changed in this round.

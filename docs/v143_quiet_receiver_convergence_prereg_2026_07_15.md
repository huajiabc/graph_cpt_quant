# v14.3 Quiet-Receiver Graph Convergence Preregistration

Date frozen: 2026-07-15, after inspecting v14.2 and before constructing or inspecting any v14.3
edge, event, or return.

## Adaptive status

This is an explicitly adaptive follow-up. v14.2 showed that community aggregation improved gross
residual continuation but did not cover costs or time splits. Consequently, even a historical pass
here can grant only provisional forward-shadow status and cannot establish independent alpha until
new post-freeze data arrive.

## Distinct hypothesis

A broad residual shock reveals direction in a source community. A separate volatility graph should
identify quiet communities likely to become volatile next. Pairing source fade with receiver
catch-up may monetize graph convergence while reducing market-direction exposure.

## Frozen state and volatility graph

- Reuse v14.2's causal monthly membership, BTC betas, hourly community return/volatility z-scores,
  breadth, and minimum-five-member rule.
- A source release requires absolute return z at least 2, volatility z at least 1, and same-sign
  breadth at least 65%.
- A receiver is quiet only when absolute return z is at most 0.75 and volatility z is at most 0.
- For every target month, train on the preceding 30 days and stop four hours before month start.
  Sample every four hours and require 150 complete observations.
- Graph source is community volatility z. Graph target is the follower's mean absolute residual
  one-hour node return over the next four hours. Retain positive source-target correlations whose
  magnitude exceeds the reverse pair, shrink by `sqrt(n/(n+150))`, and keep three followers per
  leader.

## Frozen events and candidates

At an hourly decision, require one active source with at least two quiet graph followers. Select the
strongest source opportunity and its two highest-weight quiet receivers. Construct a 50/50 spread:

- receiver leg follows the source return sign, equally weighted across all receiver members;
- source leg takes the opposite sign, equally weighted across all source members.

Test exactly three holding horizons with horizon-matched cooldowns:

1. `QRC1_QUIET_RECEIVER_CONVERGENCE_4H`;
2. `QRC2_QUIET_RECEIVER_CONVERGENCE_8H`;
3. `QRC3_QUIET_RECEIVER_CONVERGENCE_12H`.

The primary endpoint is BTC-residual spread return after 40 bp total round-trip cost. Naked net20
and net30 are secondary.

## Controls and gates

- Fifty within-month random-follower graphs preserve leader outdegree and edge-weight slots.
- Reversed edges and a 24-hour delayed source state are mandatory controls.
- Two-thousand entry-day block bootstrap draws use residual net40.
- The random family maximum spans all three horizons; require the 95th percentile.
- Each horizon requires at least 120 full, 30 validation, and 30 holdout observations; positive
  residual net40 and naked net20 in development, validation, and holdout; positive full naked
  net30; positive bootstrap lower bound; superiority to reversed and delayed controls; positive
  month contribution concentration at most 35%; and worst period no worse than -40 bp.

No PaperLive, leverage, or live-order permission is granted. A passing result remains provisional
until confirmed on data after this freeze.

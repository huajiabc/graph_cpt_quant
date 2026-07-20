# v22.4--v22.6 Alt-Book Vacuum Pressure Round Summary

Date: 2026-07-17

## Verdict

The 16-alt synchronized order-book state is a valid realized-volatility
forecasting primitive, but not an executable BTC direction signal. The primary
candidate is rejected and receives no strategy or deployment status.

## Evidence

- v22.4 feature audit passed 16/16 checks: 159 causal events over 11 months,
  split 63/47/49 across development/validation/holdout and 53/106 long/short.
- BTC four-hour gross was `+2.82 bp/event`; after the preregistered 10bp cost,
  primary return was `-7.18 bp/event`, and 20bp stress was `-17.18 bp/event`.
- Development primary was `+6.54 bp`, but validation and holdout were
  `-15.39/-16.96 bp`.
- One-hour gross was `-11.80 bp`; a one-hour delayed four-hour entry was better
  than the candidate at `-1.07 bp` net.
- The observed result ranked at only the 39th percentile of 1,000 same-month,
  same-direction random non-event paths. Pressure without broad withdrawal was
  also better (`-5.28 bp` net).
- The equal-weight 16-alt directional bucket was worse: `-3.37 bp` gross and
  `-23.37 bp` after its 20bp cost.
- Future/prior BTC four-hour realized-variance ratio averaged `1.77x`; it was
  above one in both validation (`2.34x`) and holdout (`1.47x`). The median ratio
  was only `0.84x`, so expansion is right-skewed rather than universal.
- v22.6 independently repriced every event/control, replayed all 1,000 random
  paths and reproduced bootstrap/gates; 16/16 audit checks passed.

## Interpretation

Broad depth withdrawal contains information about the *size distribution* of
future BTC moves, but the contemporaneous alt-book imbalance sign does not tell
which way BTC will move. This cleanly separates a volatility primitive from
directional alpha. Retuning the pressure q90, breadth, withdrawal count, delay,
or BTC horizon on these outcomes would be specification search.

The remaining non-duplicate question inside this frozen event family is
cross-sectional rather than common direction: whether the relative ordering of
individual alt-book imbalances predicts a top-versus-bottom alt return spread
when the system is already in a broad liquidity-vacuum state.

No live, PaperLive, leverage, remote, application, or order state changed.

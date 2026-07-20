# v22.1--v22.3 SFI-on-FSS3 Overlay Round Summary

Date: 2026-07-17

## Result

The causal spot-minus-perpetual flow inventory feature is **not** a usable
coin-level enhancement to weekly FSS3 under the sole preregistered 0.50 rank
tilt. The overlay is rejected and receives no strategy or deployment status.

## Evidence chain

- v22.1 feature audit: 13/13 checks passed. The source precedes Monday FSS3
  entry by 12--36 hours, preserves every name and funding-sign side, covers 35
  weeks/nine months, and contains no future outcome field.
- v22.2 zero-tilt reconstruction: maximum error `2.55e-15` versus saved FSS3.
- v22.2 active-week FSS3 increment: `-5.69 bp/week` primary and
  `-6.09 bp/week` stress.
- Chronological primary increments: development `-2.89`, validation `-11.57`,
  and holdout `-3.22 bp/week`.
- Attribution: price `-6.09 bp/week`, funding `+0.80 bp/week`. The failure is
  therefore price selection, not merely extra cost.
- Fixed 80/20 CM2 increment: `-4.55 bp/week` primary.
- Paired four-week bootstrap 95% interval: `[-16.36, +3.27] bp/week`.
- Within-side random-rank percentile: `26.1%` across 1,000 paths. The observed
  overlay is worse than most arbitrary rank assignments.
- Reversed-rank increment: `-2.25 bp/week`, also better than the hypothesized
  direction.
- v22.3 independent audit: 23/23 checks passed and validates rejection.

## Interpretation

The earlier standalone SFI2 result (`+5.00 bp/event` gross but far below a
20bp execution cost) did not transfer into a cost-efficient weekly overlay.
Its limited rank information is either too short-lived for Monday-to-Monday
holding, misaligned with the funding-sign cross section, or dominated by a few
large adverse price weeks. More SFI tilt tuning would now be outcome-driven and
is not justified.

The reusable lesson is architectural: orthogonal microstructure features must
be tested at their natural propagation horizon before being compressed into a
weekly structural basket. The next price/volatility study should therefore
focus on causal shock-to-receiver timing and bucket aggregation first, and only
then test a no-extra-cycle portfolio overlay if the direct propagation edge is
large enough.

No live, PaperLive, leverage, remote, application, or order state changed.

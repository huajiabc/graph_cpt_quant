# v10.4 P2 Graph-Bucket Overlay Findings

Date: 2026-07-14

Verdict: `reject_p2_graph_bucket_overlay`.

## What was tested

The v10.3 bucket definition was applied only as a fail-open filter on the existing P2 candidate
pool. A candidate was blocked when its as-of neighbor bucket was strong and broad while the
target lagged the bucket by at least 30 bp. P2's original maximum-eight-position selection was
then rerun so released capacity could be filled by later candidates. Five hundred within-month
joint feature permutations and a one-day shifted signal served as controls.

## Results

- Candidate pool: 158; graph-covered: 113; blocked: 8.
- Baseline selected 147 trades and produced 10.8673% portfolio net return at the explicit
  40 bp round-trip label.
- Overlay selected 142 trades and produced 10.9597%, a lift of only 0.0924 percentage points.
- Validation lift was +0.6287 percentage points, but holdout lift was exactly zero because no
  holdout candidate was blocked.
- The eight blocked candidates averaged **+0.6326% net**, contrary to the filter hypothesis.
- The real overlay ranked at the 90.4th percentile of permutations, but the one-day shifted
  overlay produced a much larger +0.4837 percentage-point lift.
- The day-block bootstrap interval for overlay-minus-baseline was
  [-0.2239%, +0.3059%].

Seven blocked names had been selected by the baseline. Two replacement trades were admitted.
The development-period replacements and removals explain why the full-period lift is small and
not monotonic: the filter removed both large winners and losers rather than isolating a stable
bad state.

## Interpretation

The validation improvement is sparse and not attributable to contemporaneous graph-bucket
information: coverage and holdout sample gates fail, blocked trades are profitable on average,
the shifted placebo is stronger, and the confidence interval crosses zero. The overlay must not
be promoted or attached to P2.

No PaperLive, leverage, or live permission changed.

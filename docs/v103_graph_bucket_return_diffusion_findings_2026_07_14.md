# v10.3 Graph-Bucket Return Diffusion Findings

Date: 2026-07-14

Verdict: `reject_fixed_top5_graph_bucket_diffusion`.

## What was tested

Monthly as-of top-five return-correlation neighbors were converted into a continuous,
equal-weight neighbor bucket. Three false-to-true states tested laggard catch-up, lag without
a 15-minute turn, and target/bucket co-impulse. The primary horizon was four hours, with
explicit 20 bp single-leg and 40 bp two-leg round-trip costs, 50 random graphs, a one-day
shifted placebo, chronological splits, and a day-block bootstrap.

## Results

| Candidate | Observations | Raw net20 | Target-minus-bucket net40 | Validation net40 | Holdout net40 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|
| Broad lag catch-up | 4,287 | -23.31 bp | -41.82 bp | -44.67 bp | -43.15 bp | 0% |
| Lag, no 15m turn | 3,748 | -19.51 bp | -36.16 bp | -40.95 bp | -35.09 bp | 86% |
| Co-impulse continuation | 4,401 | -25.68 bp | -47.61 bp | -50.03 bp | -42.12 bp | 0% |

All five chronological slices were negative for all three candidates. The 95% bootstrap
intervals for the primary target-minus-bucket return were entirely negative.

The least-bad candidate, lag without a turn, had only +3.84 bp gross target-minus-bucket
return before the 40 bp two-leg cost. It did not clear the random-family 90th-percentile gate.

## Interpretation

- A strong correlated-neighbor bucket does not create a tradable four-hour target catch-up.
- In the broad-lag and co-impulse cases, the real high-correlation graph was worse than random
  neighbor sets. The earlier NIR result was therefore more consistent with market-density or
  hot-time context than with edge-specific diffusion.
- This rejects the fixed top-five, high-frequency laggard-catch-up mechanism. It does not reject
  all graph information or using a bucket as context for an already independent entry signal.

No PaperLive, leverage, or live permission changed.

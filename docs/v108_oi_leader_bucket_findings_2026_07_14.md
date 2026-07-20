# v10.8 OI-Leader Downstream Bucket Findings

Date: 2026-07-14

Verdict: `reject_oi_graph_bucket_family`.

## What was tested

The continuous OI panel provided 73 symbols, approximately 2.46 million warm-up-complete rows,
and 99.99% OI-z coverage from June 2025 through June 2026. Monthly edges used the preceding
30 days and linked a leader's one-hour OI z-score to another symbol's future four-hour
BTC-residual return. The last four hours before every month boundary were excluded from graph
training so all edge targets were realized at freeze time.

The graph tested positive-price/rising-OI propagation as a downstream long bucket and
negative-price/rising-OI propagation as a downstream short bucket. Portfolios had a four-hour
global cooldown, three-to-five followers, 50 random graphs, reversed edges, one-day shifted
signals, chronological splits, explicit costs, and day-block bootstrap intervals.

## Results

| Candidate | Observations | Residual gross | Residual net40 | Validation net40 | Holdout net40 | Validation raw net20 | Holdout raw net20 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive OI propagation, long | 1,316 | -2.59 bp | -42.59 bp | -44.06 bp | -43.34 bp | -25.93 bp | -27.66 bp | 2% |
| Negative OI propagation, short | 544 | -18.24 bp | -58.24 bp | -68.87 bp | -51.01 bp | -34.80 bp | -19.95 bp | 0% |

The bootstrap intervals were entirely negative: [-49.21 bp, -36.12 bp] for the long bucket and
[-79.47 bp, -42.24 bp] for the short bucket. Real edges also lost to reversed and one-day shifted
controls. Every complete holdout month was negative; the partial June observations are too few
and do not change the verdict.

## Interpretation

Cross-sectional OI shocks are abundant and cleanly measured, but the historical pairwise
OI-to-future-return correlations do not identify stable propagation paths. The real graph is
worse than randomized membership, so this is not a cost-only failure and should not be rescued
by leverage or threshold tuning.

Together with v10.3-v10.6, the result closes price-only and OI-only high-frequency graph edges.
The remaining graph-alpha hypothesis is the genuinely synchronized cross-venue aggressor-flow
graph in v10.7. It must wait for forward data rather than being approximated with biased archives.

No PaperLive, leverage, or live permission changed.

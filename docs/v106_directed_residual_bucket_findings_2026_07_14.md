# v10.6 Directed Residual Graph-Bucket Findings

Date: 2026-07-14

Verdict: `reject_directed_residual_bucket_family`.

## What changed

This was not another threshold search on the rejected contemporaneous correlation graph.
Every monthly graph used only the preceding 30 days, removed static BTC beta from 15-minute
returns, and retained directionally asymmetric 15/30/60-minute lead-lag edges. The strongest
three leaders per follower were frozen for the target month.

At each timestamp, multiple leader residual impulses produced a predicted downstream follower
score. Up to five followers formed the traded bucket. The primary outcome was four-hour
BTC-beta-neutral return after 40 bp total round-trip cost; naked bucket return after 20 bp was
secondary.

## Coverage

- Test months: 2025-09 through 2026-06; June contains only 12 portfolio observations.
- Monthly edges: 105-123.
- Monthly followers: 35-41.
- Selected lags: 508 edges at 15 minutes, 385 at 30 minutes, and 285 at 60 minutes.
- Portfolio observations: 1,424 propagation and 1,376 laggard observations.

## Results

| Candidate | Residual gross | Residual net40 | Validation net40 | Holdout net40 | Validation raw net20 | Holdout raw net20 | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Directed propagation | -1.50 bp | -41.50 bp | -43.83 bp | -36.81 bp | -25.76 bp | -18.14 bp | 0% |
| Directed laggard | +1.20 bp | -38.80 bp | -40.70 bp | -39.99 bp | -26.31 bp | -21.48 bp | 12% |

The propagation bootstrap 95% interval was [-46.45 bp, -36.77 bp]. The laggard interval was
[-43.90 bp, -33.29 bp]. Every full month except the 12-observation partial June had negative
primary mean return for both candidates.

The real graph also failed attribution controls:

- propagation real -41.50 bp versus shifted -37.47 bp and reversed -37.86 bp;
- laggard real -38.80 bp versus shifted -37.03 bp and reversed -38.41 bp;
- random-graph family maximum had median -37.11 bp and 90th percentile -34.98 bp.

## Interpretation

BTC residualization and lag asymmetry produce visually directional edges, but those edges do
not generate economically directional downstream returns. The laggard version's +1.20 bp
gross residual return is negligible relative to costs and weaker than random graph membership.

This closes high-frequency price-only graph propagation in three increasingly strict forms:

1. contemporaneous correlation-neighbor catch-up;
2. direct contemporaneous bucket continuation/reversal;
3. BTC-residual directed lead-lag downstream buckets.

The graph framework itself is not rejected. The next defensible graph alpha must introduce an
orthogonal edge variable rather than another transform of price returns. The strongest remaining
route is a directed flow graph: leader nodes defined by synchronized taker-flow/OI/funding shocks,
with downstream bucket returns evaluated only after the flow tape has sufficient forward coverage.
A slower community-risk allocation layer is also possible, but it is portfolio conditioning rather
than a new entry alpha.

No PaperLive, leverage, or live permission changed.

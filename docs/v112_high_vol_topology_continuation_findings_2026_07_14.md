# v11.2 High-Volatility Topology-Continuation Findings

## Verdict

`reject_high_vol_topology_family` for deployment. The month-frozen BTC high-volatility state greatly
strengthens the sparse topology-continuation payoff and the real balanced communities beat the
fully rebuilt random-community horizon family. The result is nevertheless too sparse, unstable,
and result-informed for PaperLive.

## Formal results

| Horizon | Observations | Gross | Net 20 bp | Net 50 bp | Validation net 20 | Holdout net 20 | Random-family percentile |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1h | 43 | +38.51 bp | +18.51 bp | -11.49 bp | +24.75 bp | +46.92 bp | 90% |
| 2h | 43 | +77.84 bp | +57.84 bp | +27.84 bp | +89.46 bp | +61.06 bp | 100% |
| 4h | 43 | +82.03 bp | +62.03 bp | +32.03 bp | +68.39 bp | +32.76 bp | 100% |
| 8h | 42 | +38.34 bp | +18.34 bp | -11.66 bp | +21.31 bp | +62.71 bp | 90% |
| 12h | 42 | +73.74 bp | +53.74 bp | +23.74 bp | +19.29 bp | +116.50 bp | 100% |

The four-hour real result exceeds all fifty random-partition family maxima. The random-family
maximum has mean -6.89 bp net 20, 90th percentile +17.05 bp, 95th percentile +21.98 bp, and maximum
+51.92 bp. The same-horizon shifted signal is -5.61 bp net 20.

## Why it still fails

The four-hour result has only 43 observations over 29 active days. Validation has 21 observations
and the holdout label only seven, below the unchanged minimums of 25. The formal entry-day bootstrap
95% interval is [-25.75, +176.86] bp. Five chronological net means are -32.40, +208.38, -42.11,
+76.18, and +106.62 bp. The largest positive month contributes 41.90%, above the 35% limit; the
largest community contributes 30.32%.

Monthly four-hour net means range from -140.28 bp in 2026-01 to +659.32 bp in 2026-03, where only
two observations occur. The strategy is therefore a rare convex state interaction, not yet a
stable average-return process.

## Interpretation

This is the strongest graph-specific mechanism evidence in the current research line:

1. balanced real communities outperform exact-size random communities;
2. the effect survives 20, 30, and 50 bp costs at 2h, 4h, and 12h;
3. both validation and holdout labels have the intended sign;
4. a one-day signal displacement destroys the result;
5. the interaction has a plausible state interpretation—community leaders and laggards continue
   separating when the global market is already in a high-volatility state.

The same evidence also shows why leverage is premature: leverage scales a payoff dominated by a
few days and does not repair its negative bootstrap lower bound.

## Next action

Freeze this exact v11.2 rule for forward observation without trade permission. Do not add more
retrospective thresholds to the same sample. Continue independent work on network break breadth
and synchronized cross-venue flow, but treat them as separate preregistered hypotheses rather than
filters used to rescue v11.2.

No PaperLive or live permission changed.

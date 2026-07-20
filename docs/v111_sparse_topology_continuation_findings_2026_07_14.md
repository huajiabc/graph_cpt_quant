# v11.1 Sparse Topology-Continuation Findings

## Verdict

`reject_sparse_topology_family` for deployment. The expanding historical 80th-percentile severity
rule converts the v11.0 gross signal into positive cost-adjusted means at several horizons, and the
four-hour horizon clears most formal gates. It still fails day-level uncertainty and chronological
stability. Because this is a result-informed follow-up, even a full pass would have remained
retrospective forward-watch research rather than a PaperLive candidate.

## Formal horizon results

The sparse rule yields 174 four-hour portfolio observations across 117 active days and eight
months. Earlier months do not trade until 100 strictly prior base events exist.

| Horizon | Gross | Net 20 bp | Validation net 20 | Holdout net 20 | Random-family percentile |
|---:|---:|---:|---:|---:|---:|
| 1h | +14.75 bp | -5.25 bp | -5.00 bp | -9.77 bp | 74% |
| 2h | +28.22 bp | +8.22 bp | +12.79 bp | +4.23 bp | 88% |
| 4h | +32.58 bp | +12.58 bp | +10.55 bp | +5.14 bp | 92% |
| 8h | +20.77 bp | +0.77 bp | -0.22 bp | +1.08 bp | 82% |
| 12h | +21.78 bp | +1.78 bp | +1.01 bp | -11.25 bp | 84% |

The four-hour result also remains +2.58 bp after a 30 bp round-trip assumption and beats the
same-horizon one-day shifted control (-21.29 bp net 20). Its positive-PnL month and community
concentration shares are 31.08% and 17.54%, both inside the frozen 35% limits.

## Why four hours still fails

The four-hour entry-day bootstrap 95% interval is [-19.40, +48.72] bp after 20 bp cost. Its five
chronological net means are +15.22, +3.18, +50.63, +30.94, and -38.53 bp. The median entry loses
14.85 bp and only 40.8% of entries are positive. Mean profitability therefore depends on a convex,
right-tailed payoff rather than a stable per-event edge.

Monthly net performance is strongly positive in 2025-10, 2025-12, 2026-02, 2026-03, and 2026-04,
but sharply negative in 2026-01 and 2026-05. This explains both the negative last chronological
slice and the wide bootstrap interval.

## Leg attribution

At four hours, the full-sample top-bucket residual is +39.95 bp and the bottom-bucket residual is
-25.21 bp, producing the normalized +32.58 bp spread. The source changes by period:

- Development is mainly a strong top leg (+120.63 bp), while the bottom leg is also positive.
- Validation is mainly a weak bottom leg (-70.48 bp), while the top leg is slightly negative.
- Holdout has both desired signs: top +35.17 bp and bottom -15.11 bp.

This supports the relative spread more than either standalone leg. A single leg would require a
new BTC-hedge and execution model and is not eligible from this attribution.

## Post-hoc market-state diagnostic

Joining the sparse events to an as-of BTC 24-hour realized-volatility state shows a potentially
important interaction. If volatility is above the month-frozen trailing-30-day 75th percentile,
the four-hour subset has 43 observations and averages +82.03 bp gross and +62.03 bp net 20.
Validation and holdout-label net means are +68.39 and +32.76 bp.

This filter is post-hoc, spans only 29 active days, and still has a bootstrap lower bound near
-23.36 bp. Two March entries contribute an unusually large mean. It is mechanism evidence for a
separate high-volatility topology-continuation version, not a promotion result.

## Next research boundary

The next clean version should freeze the trailing volatility percentile before replay, retain the
expanding severity rule, correct for all horizons and regime branches, and require new forward data
before promotion. It should also test whether a network-wide break-breadth state explains the poor
January and May behavior without stacking several post-hoc filters.

No PaperLive or live permission changed.

# v11.0 Balanced-Community Topology-Break Findings

## Verdict

`reject_topology_break_family` for deployment. The balanced graph is structurally valid and the
break-continuation sign contains a repeatable gross signal, but it does not survive the frozen cost
gate. No PaperLive or live permission changes are justified.

## Structural check

Every evaluated month from 2025-08 through 2026-06 produced exactly eight communities of nine
coins. This resolves the v10.9 giant-component defect: the result is now a real local multi-coin
community test rather than a disguised broad-universe spread.

## Formal results

| Candidate | Observations | Gross 4h | Net 20 bp | Validation net 20 bp | Holdout net 20 bp | Random-family percentile |
|---|---:|---:|---:|---:|---:|---:|
| Topology repair | 739 | -8.66 bp | -28.66 bp | -26.27 bp | -35.73 bp | 0% |
| Break continuation | 739 | +8.66 bp | -11.34 bp | -13.73 bp | -4.27 bp | 100% |

For break continuation, the validation and holdout gross means are +6.27 bp and +15.73 bp. It
beats the one-day shifted control (-22.48 bp net 20) and every one of the fifty random-partition
family maxima. Its 95% entry-day bootstrap interval after 20 bp cost is [-20.42, -1.40] bp, so the
formal cost-adjusted conclusion is still decisively negative.

The random-family maximum has mean -16.86 bp net 20, 90th percentile -13.88 bp, and maximum
-12.76 bp. The real continuation candidate's -11.34 bp therefore carries graph-specific ordering
information even though it is not yet executable alpha.

## Concentration and exploratory diagnostics

Positive net PnL is concentrated: the largest positive month contributes 80.85%, although the
largest community contributes only 18.49%. All five chronological net-20 slices remain negative.

An explicitly post-hoc severity diagnostic shows gross returns of -2.06, +4.10, -4.96, +15.80,
and +30.36 bp across ascending breach-severity quintiles. Only the top quintile clears 20 bp cost
(+10.36 bp net). This is useful mechanism evidence, not a promotion result: that severity filter
was not preregistered, April dominates positive PnL, and the same data cannot become a new untouched
holdout after inspection.

## What this closes and what remains

This run closes unconditional trading of every fifth-percentile balanced-community topology break.
It does not close the topology mechanism. The next defensible follow-up is a separately registered
sparse version whose severity threshold is learned from development history only, with leg-level
attribution and lower-turnover holding horizons. Any such retrospective candidate must remain
research-only until new forward observations accumulate.

Cross-venue synchronized flow graphs remain independently gated by data age, and graph-aware
allocation of already independent sleeves remains a separate portfolio-layer question.

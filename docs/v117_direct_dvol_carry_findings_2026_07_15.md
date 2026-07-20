# v11.7 Direct DVOL Carry Findings

Date: 2026-07-15

Verdict: `reject_direct_dvol_basis_carry`. No PaperLive, live, sizing, or
leverage permission changed.

## Result

The 14-day basis-convergence rule produced only 20 executable completed
contracts. Sparse positive-volume bars reduced validation and holdout to four
contracts each, below the preregistered minimum of six.

| Candidate | Full n | Full net10 | Validation net10 | Holdout net10 | Random-sign percentile |
|---|---:|---:|---:|---:|---:|
| All non-zero basis | 20 | -18.60 bp | +189.97 bp | -63.19 bp | 49.55% |
| Absolute basis at least 2 points | 6 | -177.42 bp | -439.68 bp | -36.18 bp | 36.90% |

The primary bootstrap interval was -6.38% to +5.09%. Reversing the primary
direction lost only 1.40 bp, so the frozen convergence direction did not add
information.

The preregistered 7-day and 21-day timing controls were positive (+1.91% and
+3.43% net10), but they are controls rather than eligible candidates. They
cannot be promoted after the 14-day primary failed, particularly with the
same sparse contract archive and no historical bid/ask.

## Interpretation

BTCDVOL futures are directly tradeable, but their archived trade bars are too
sparse for a stable monthly carry claim. A future test must record live
bid/ask and executable size; another historical entry-day search would be
timing optimization on roughly the same twenty contracts.

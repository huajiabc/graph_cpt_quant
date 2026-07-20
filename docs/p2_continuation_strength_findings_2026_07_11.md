# P2 Continuation Strength - Initial Findings

## Decision

Do not promote or gate P2 with the continuous score. Keep the score in the
forward ledger as a pre-registered diagnostic until it reaches 100 timely core
trades and 30 timely bursts.

## Historical Reference

The isolated replay produced 110 search trades, 25 validation trades, and 12
legacy-holdout trades for the P2 max8 baseline.

| Period | Bursts | Trades | Mean burst net20 | Slope | Spearman | Bootstrap slope 95% | Permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Search reference | 46 | 110 | +0.2531% | +0.0240 | +0.245 | [+0.0026, +0.0501] | 0.0115 |
| Validation reference | 10 | 25 | +0.1685% | -0.0019 | +0.236 | [-0.0306, +0.0344] | 0.8870 |
| Legacy holdout | 6 | 12 | -0.4099% | -0.0153 | +0.086 | [-0.0695, +0.6719] | 0.6850 |

The fixed score is monotonic in the search reference but does not reproduce in
validation or the already-observed holdout. This is evidence of search-period
structure, not a usable selector.

## Forward State

- Timely forward trades: 0
- Timely forward bursts: 0
- Decision sample ready: no

The absence of timely rows is expected: the cumulative ledger was introduced
after the historical observations. Existing rows are deliberately not relabeled
as forward evidence.

## Next Action

Log the four fixed score components on each newly observed P2 signal. Do not
change weights, bounds, or bin edges. Re-run the report only as new timely rows
arrive; do not use the legacy holdout to tune a replacement score.

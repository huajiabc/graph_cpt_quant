# v23.10--v23.16 Forward and Density-Extension Round Summary

This round separated genuine forward observation from historical density
interpolation around the post-selected v23.8 positive-pressure breakout.
It did not change any live, PaperLive, leverage, remote, application, or order
state.

## Forward observation

- v23.10 collected all 16 Binance book-depth archives for 2026-07-15 under
  the isolated `data/external/v238_forward_shadow` root.
- The 24 hourly states after the historical cutoff contained zero strict
  v22.4/q90 positive-pressure events.
- Because the frozen candidate did not fire, the forward evaluator did not
  load BTC path outcomes. This is an outcome-free null observation, not a
  losing or winning trade.
- The forward collector preserves prior daily features and BTC bars so later
  archives can be appended without mixing them into the historical root.

## Frozen density extensions

| Candidate | Events | Dev / val / hold primary net | Stress, all | Absolute month bootstrap lower | Decision |
|---|---:|---:|---:|---:|---|
| positive q80, 0.625 sigma | 89 | -9.95 / +20.85 / -5.22 bp | -9.78 bp | -23.62 bp | reject |
| positive q85, 0.625 sigma | 75 | +3.72 / +36.19 / +6.77 bp | +4.00 bp | -11.91 bp | reject |

The q80 feature set was frozen by v23.11 and independently audited by v23.13.
The sole midpoint interpolation, q85, was frozen by v23.14 and independently
audited by v23.16. The q85 result was encouraging in level but failed three
predeclared robustness conditions: its absolute month-bootstrap lower bound
was negative, its holdout matched-random percentile was 87.5 rather than at
least 90, and the adjacent 0.75-sigma width was negative in development.

## Interpretation and next boundary

The return does not extend smoothly as the pressure threshold is relaxed:
q80 is effectively flat after primary costs and q85 remains statistically
fragile. The surviving q90 result therefore belongs to an extreme-tail,
sparse-event hypothesis. It remains a forward-shadow candidate and is not a
confirmed alpha or deployment authorization.

No more pressure-quantile interpolation is justified by this round. The next
useful test is orthogonal: map the frozen q90 event return into the existing
weekly FSS3/TG1 portfolio calendar and ask whether a small, fixed satellite
adds risk-adjusted return without duplicating the core sleeves. Any such test
must preserve the original event timing and costs and must not tune the q90
feature, breakout width, or allocation after observing outcomes.

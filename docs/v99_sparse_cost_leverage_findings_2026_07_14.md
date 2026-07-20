# v9.9 Sparse Cost-Aware Residual Trading and Leverage - Findings

## Decision

Sparse cost-aware trading materially reduced turnover, but it did not produce a
stable unlevered alpha. Every leverage level is therefore ineligible.

Status: `reject_sparse_cost_aware_alpha`. No shadow, P2, paper-live,
canary-live, or real-live permission changes are allowed.

This closes the current price/volume/funding/OI residual-model lane under the
tested direct, hysteresis, sparse, and leverage specifications. Leverage cannot
repair its negative validation or holdout expectancy.

## Four-hour result

January-February development selected `off` for every real model/book
combination. The 20bp hurdle was the least bad active alternative.

- Residual Ridge made no development trades. In March-May its diagnostic long
  book traded in only two periods, earned additive net20 of +0.19%, and had
  essentially zero average exposure. This is insufficient evidence, not a
  sparse candidate.
- Residual XGBoost long-short reached 8.7% average turnover and 15.0% average
  gross exposure, but March-May net20 was -1.79% and net30 was -6.57%.
- Residual XGBoost long-only reached 4.9% turnover and 8.0% exposure, but net20
  was -14.29%.
- The XGBoost long-short monthly net20 path was -7.87%, +8.70%, and -2.62% for
  March, April, and May. The favorable April pocket did not survive holdout.

The honest four-hour action is cash/off. The tiny Ridge diagnostic return
cannot justify leverage because the development selector rejected trading and
there were only two evaluation trade periods.

## Twelve-hour result

Unlike four hours, January-February development selected active Ridge policies
at a 20bp entry hurdle:

| book | development net20 | Mar-May net20 | Mar-May net30 | turnover | exposure |
|---|---:|---:|---:|---:|---:|
| residual Ridge long | +4.34% | -2.80% | -4.30% | 8.2% | 15.7% |
| residual Ridge long-short | +5.84% | -9.35% | -10.80% | 7.9% | 14.9% |

The long Ridge monthly net20 path was -1.88%, +2.66%, and -3.58% for March,
April, and May. The cost-aware policy achieved the preregistered sub-10%
turnover objective, but validation was mixed and the final holdout was
negative.

Residual XGBoost failed more severely. Its development-selected long book lost
45.56% net20 in March-May, while long-short lost 56.81%. Every evaluation
month was negative in both books.

The shuffled-label control again exposed path risk: its development selector
chose `off`, while the diagnostic twelve-hour long-short aggregate happened to
show +1.27% at 20bp but turned negative at 30bp. It is not evidence of alpha.

## Leverage stress

The table below uses the synthetic carry stress: transaction costs plus 2bp per
eight hours per unit of deployed gross exposure. Returns are compounded from
the frozen March-May diagnostic ledger.

| horizon/model/book | leverage | additive return | equity multiple | max drawdown | worst period |
|---|---:|---:|---:|---:|---:|
| 12h Ridge long | 1x | -3.67% | 0.959 | -6.75% | -3.19% |
| 12h Ridge long | 2x | -7.33% | 0.912 | -13.59% | -6.39% |
| 12h Ridge long | 3x | -11.00% | 0.859 | -20.78% | -9.58% |
| 12h Ridge long | 5x | -18.33% | 0.741 | -35.21% | -15.97% |
| 12h XGBoost long-short | 1x | -62.30% | 0.521 | -47.98% | -14.00% |
| 12h XGBoost long-short | 2x | -124.60% | 0.256 | -74.57% | -28.00% |
| 12h XGBoost long-short | 3x | -186.90% | 0.116 | -88.45% | -42.01% |
| 12h XGBoost long-short | 5x | -311.49% | 0.017 | -98.30% | -70.01% |

No tested path crossed the simplified one-period -100% ruin threshold, but
that does not make high leverage safe. At 5x the twelve-hour XGBoost
long-short equity was almost completely erased without needing a single
formal ruin event. Exchange maintenance margin, liquidation fees, gap
execution, and complete realized funding were not available in this ledger;
actual liquidation risk can therefore be worse.

The four-hour XGBoost diagnostics tell the same story. Under carry stress, 5x
left the long-only book at 0.371 equity with a -73.9% drawdown, and the
long-short book at 0.718 equity with a -51.0% drawdown.

## Practical conclusion for aggressive risk

The correct leverage for this newly tested alpha lane is currently `0x`, not
because aggressive risk is categorically wrong, but because the underlying
unlevered expectancy failed. Leverage should only be revisited after a frozen
1x strategy has positive validation and holdout net20, positive net30, and a
non-trivial number of independent trades.

If a future orthogonal signal passes those gates, the next leverage ladder
should begin at 1.5x and 2x with forward margin telemetry. `3x` requires a
separate drawdown budget; `5x` remains tail-risk stress rather than a deployable
recommendation.

The next alpha work should move to genuinely new information, especially the
synchronized cross-venue orderflow tape, while reusing residualization and the
sparse cost-aware executor as fixed controls.

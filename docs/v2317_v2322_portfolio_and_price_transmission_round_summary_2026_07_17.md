# v23.17--v23.22 Portfolio and Price-Transmission Round Summary

This round asked two orthogonal questions after freezing the post-selected q90
book-pressure breakout: whether it adds value to the existing portfolio, and
whether broad alt price shocks provide a cleaner standalone transmission alpha.
No live, PaperLive, leverage, remote, application, or order state changed.

## q90 as a sparse CM2 satellite

The outcome-free v23.17 audit mapped all 53 q90 events to the 49-week CM2
calendar. Events crossing Monday 00:00 UTC were assigned to the week in which
their four-hour return was realized. All 14 feature checks passed.

The v23.18 primary construction added 10% temporary notional only during an
event while leaving the 80% FSS3 / 20% TG1 core unchanged.

| Scope | Weekly increment | Core Sharpe | Combined Sharpe | Satellite/core correlation |
|---|---:|---:|---:|---:|
| all | +2.16 bp | 2.491 | 2.584 | -0.225 |
| development | +1.00 bp | 3.533 | 3.600 | -0.275 |
| validation | +4.41 bp | 0.222 | 0.424 | -0.133 |
| holdout | +1.89 bp | 2.331 | 2.409 | +0.041 |

On the 12 active weeks when CM2 was negative, the full-size satellite return
averaged +89.17 bp. Full-sample downside semideviation improved by 4.16 bp and
additive maximum drawdown improved by 2.02 bp. Fixed 5%, 10%, and 20% scales
were positive in every temporal scope. The only failed gate was the absolute
month-bootstrap lower bound, at -0.69 bp/week. Therefore this is useful
portfolio evidence but not statistical confirmation. v23.19 independently
reproduced all artifacts and passed 10/10 audit checks.

## Alt-first price volatility ignition

The v23.20 signal used only completed prices: at least 8 of 16 alts moved by
more than their own causal 1-sigma scale, the median alt shock exceeded its
prior 30-day q80, and BTC remained below its prior median shock. The rule was
frozen on feature coverage alone and produced 100 events across 12 months
(58/22/20 development/validation/holdout). All 15 feature checks passed.

The preregistered 0.75-sigma BTC OCO result was negative before costs:

| Scope | Gross | Net at 10 bp | Quiet-BTC matched percentile |
|---|---:|---:|---:|
| all | -7.53 bp | -17.43 bp | 13.8 |
| development | -10.12 bp | -19.95 bp | 15.8 |
| validation | -1.76 bp | -11.76 bp | 24.8 |
| holdout | -6.37 bp | -16.37 bp | 52.1 |

Both adjacent widths, 0.625 and 1.0 sigma, were negative in every temporal
scope. The bootstrap lower bound was -38.34 bp and every leave-one-month-out
mean remained negative. v23.22 independently rebuilt 100 events, 3,487 control
hours, 4,000 random paths, and 5,000 bootstrap draws; all 11 audit checks passed.

## Research boundary

The distinction is now sharper. Broad depth withdrawal is useful before price
movement becomes visible; broad hourly alt price movement while BTC is still
quiet is already too late and behaves more like exhaustion than propagation.
Earlier 15-minute graph studies (v11.3--v11.6) also rejected directed
volatility-receiver OCO, sign-specific semivariance, cross-community fronts,
and efficient community continuation. Repeating those families with nearby
thresholds would be outcome mining.

The remaining defensible paths are therefore:

1. keep the q90 rule frozen and accumulate genuinely new forward events;
2. collect a new pre-price information source such as liquidation/forced-flow
   tape or synchronized executable option quotes;
3. retain q90 as a small research-only diversification satellite, but do not
   promote it until its forward sample resolves the negative absolute
   month-bootstrap lower bound.

# v11.3 Directed Volatility-Transmission Breakout Findings

Date: 2026-07-15

Verdict: `reject_volatility_transmission_breakout`.

## Result

The frozen directed absolute-residual-volatility graph produced 229 events and 174 OCO-traded
portfolios across 157 days and ten target months. Receiver volatility did expand after the event:
future four-hour residual realized volatility averaged 1.2456 times prior four-hour realized
volatility. Validation and holdout-label ratios were 1.2173 and 1.3053.

That expansion was not graph-specific. Fifty random directed graphs preserving follower edge
count, lag, and edge-weight slots averaged 1.2459 times future-volatility expansion; the real graph
was at the 50th percentile and below the random 90th percentile of 1.2735. The apparent mechanism
is therefore mostly volatility mean re-expansion after selecting compressed receivers, not
identified edge-level transmission.

## Monetization

The frozen next-hour OCO range break filled 81.36% of selected legs. Same-bar dual-trigger
ambiguity was only 1.36%, and long/short fills were balanced (52.71% long). Execution ambiguity is
not the reason for failure.

| Scope | Traded portfolios | Gross 4h | Net 20 bp | Future RV expansion |
|---|---:|---:|---:|---:|
| All | 174 | -1.51 bp | -21.51 bp | 1.2456x |
| Development | 73 | -0.92 bp | -20.92 bp | 1.2311x |
| Validation | 59 | +0.14 bp | -19.86 bp | 1.2173x |
| Holdout label | 42 | -4.87 bp | -24.87 bp | 1.3053x |

The real net result ranked at the 68th percentile of random graphs and was only slightly better
than the one-day-shifted control (-24.61 bp). The entry-day bootstrap 95% interval was
[-47.22, +2.44] bp. Four of five chronological slices were negative, and the largest positive
month supplied 73.17% of positive PnL.

## Interpretation

This run separates volatility forecasting from volatility monetization:

1. compressed coins often experience later volatility expansion;
2. the directed absolute-shock graph did not improve that forecast over random membership;
3. even known expansion did not create directional futures alpha through a symmetric breakout;
4. trading realized volatility without options requires an independent direction or path-shape
   edge, not merely a forecast that the future range will be larger.

The next distinct test is sign-specific semivariance transmission: downside leader shocks to
downside receiver returns and upside leader shocks to upside receiver returns. It must be treated
as a new family with its own random-graph family correction, not as a threshold rescue of v11.3.

No PaperLive, leverage, or real-order permission changed.

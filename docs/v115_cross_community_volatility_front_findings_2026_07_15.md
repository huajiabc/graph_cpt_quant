# v11.5 Cross-Community Downside-Volatility Front Findings

Date: 2026-07-15

Verdict: `reject_cross_community_volatility_front`.

The exact 8x9 balanced-community downside front generated 378 BTC-short observations across 221
days and eleven months. Four-hour BTC-short gross return was only +6.16 bp; net returns were
-3.84 bp at 10 bp cost, -13.84 bp at 20 bp, and -23.84 bp at 30 bp. Validation and holdout-label
net 20 bp were -14.26 and -8.85 bp.

The community graph added no attribution:

- real graph random-partition percentile: 68%;
- random-community mean net 20 bp: -14.51 bp;
- global downside-breadth comparator: -16.98 bp;
- one-day shifted control: -18.15 bp.

The timing was better than a one-day displacement and global breadth, but remained below the
random-family gate. The bootstrap 95% interval was [-24.74, -2.84] bp, four of five chronological
slices were negative, and positive PnL was fully concentrated in one month.

This closes the tested chain from edge-level absolute volatility, through sign-specific
semivariance, to cross-community systematic-risk breadth. Volatility clusters and downside states
exist, but their location in the price graph has not supplied executable leading information.

The remaining direct price-only route is not another propagation transform. It is path
classification at the already-active source: distinguish coordinated, high-efficiency directional
range expansion from choppy high volatility, and test same-community continuation after costs.

No PaperLive, leverage, or real-order permission changed.

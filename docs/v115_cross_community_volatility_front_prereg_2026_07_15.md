# v11.5 Cross-Community Downside-Volatility Front Preregistration

Date: 2026-07-15

Status: frozen before the first v11.5 outcome run. Research only.

## Question

v11.4's downside receiver bucket had +18.91 bp raw gross return but only +5.53 bp after static BTC
residualization. This suggests a possible system-risk timing effect rather than edge-level
propagation. Does downside residual volatility appearing simultaneously in several frozen graph
communities lead BTC downside *before BTC has already sold off*?

## Frozen construction

- Use the same 73-symbol 15-minute panel and chronological labels as v11.3/v11.4.
- At each month start, use the strictly preceding 30 days to estimate static BTC beta, residual
  scale, and eight balanced nine-coin spectral communities.
- For each non-BTC coin, define downside shock as `max(-residual / scale, 0)` and freeze its
  trailing-month 90th-percentile shock threshold.
- At each completed hour, community downside density is the fraction of its nine members whose
  latest 15-minute downside shock exceeds that symbol's threshold.
- Freeze each community's active-density threshold at its trailing-month 90th percentile.
- Cross-community front breadth is the number of active communities. Freeze its event threshold
  at the trailing-month 90th percentile.

## Frozen event and trade

An event occurs when front breadth is at or above its frozen threshold, at least three communities
are active, and BTC's known trailing one-hour return is above its trailing-month 25th percentile.
The BTC condition explicitly requires the alt-community front to lead rather than simply follow an
already large BTC selloff. Apply a four-hour cooldown.

The single candidate `VCF1_ALT_FRONT_LEADS_BTC` shorts BTC for four hours. Primary outcome uses
20 bp round-trip cost; 10, 30, and 50 bp are reported as sensitivity. No stop, leverage, or
holding-period search is allowed.

## Controls and gates

- 50 exact-size random community partitions rebuilt month by month;
- a global downside-breadth comparator without graph communities;
- one-day shifted event state;
- entry-day block bootstrap, five chronological slices, month concentration;
- development/validation/holdout-label reporting.

Forward-watch eligibility requires at least 100 observations, at least 25 validation and 25
holdout observations, positive validation and holdout net 20 bp, positive full net 30 bp, real
result at or above the random-family 90th percentile, better than both shifted and global-breadth
controls, positive bootstrap lower bound, five non-negative chronological slices, and no more than
35% positive PnL concentration in one month.

Even a formal pass remains retrospective research and cannot change PaperLive or real-order
permission.

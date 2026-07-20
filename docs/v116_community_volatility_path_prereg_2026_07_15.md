# v11.6 Community Volatility Path-Efficiency Preregistration

Date: 2026-07-15

Status: frozen before the first v11.6 outcome run. Research only.

## Question

The v11.3-v11.5 propagation families failed. Can high realized volatility itself become
directional futures alpha when the *source community's price path* is coordinated and efficient,
rather than choppy?

## Frozen construction

- Same 73-symbol 15-minute panel and chronological labels.
- At month start, use the strictly preceding 30 days to estimate static BTC beta, residual scale,
  and eight balanced nine-coin spectral communities.
- For each community and completed hour, form the equal-weight residual bucket return on each of
  the last four 15-minute bars.
- One-hour realized bucket volatility is the square root of the four squared bucket returns.
- Path efficiency is absolute one-hour bucket return divided by the sum of absolute 15-minute
  bucket returns; it lies in [0, 1].
- Direction breadth is the fraction of community members whose one-hour residual return has the
  same sign as the community bucket.
- Freeze community-specific trailing-month thresholds at the 95th percentile for realized
  volatility and the 75th percentile for path efficiency.

## Frozen event and trade

A community is eligible when realized volatility and path efficiency exceed their frozen
thresholds and direction breadth is at least two-thirds. At each timestamp select up to three
communities by volatility-threshold ratio. Require one community and apply a four-hour portfolio
cooldown.

`CVP1_EFFICIENT_VOL_CONTINUATION` trades every selected community equal-weight in the sign of its
known one-hour residual bucket move and holds four hours. The portfolio is equal-weight across
community sleeves. Primary raw return uses 20 bp round-trip cost; 30 and 50 bp are stress cases.
BTC-residual return is attribution only. No stop, leverage, opposite-direction branch, or horizon
search is allowed.

## Controls and gates

- 50 exact-size random partitions with all thresholds rebuilt from prior data;
- one-day shifted signal state;
- entry-day bootstrap, five chronological slices, month and community concentration;
- development/validation/holdout-label reporting.

Forward-watch eligibility requires at least 100 observations, at least 25 validation and 25
holdout observations, positive validation and holdout net 20 bp, positive full net 30 bp, at least
the random-family 90th percentile, better than the shifted control, positive bootstrap lower
bound, five non-negative chronological slices, and no more than 35% positive PnL concentration in
one month or community.

Even a pass remains retrospective research and cannot change PaperLive or real-order permission.

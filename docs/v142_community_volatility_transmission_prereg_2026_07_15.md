# v14.2 Community Volatility-Transmission Preregistration

Date frozen: 2026-07-15, after rejecting v14.1 and before constructing or inspecting any v14.2
edge, event, or return.

## Distinct hypothesis

Symbol-level price, OI, and taker-flow propagation graphs have been too noisy. v14.2 first
aggregates causally frozen monthly communities, then tests whether a broad idiosyncratic volatility
release in one community predicts a four-hour continuation or reversal bucket in other communities.
The source is therefore a multi-coin community state, not a renamed single-coin shock.

## Causal node state

- Use v13.2's eight monthly communities and closed Bybit 15-minute prices.
- For every target month, estimate symbol BTC betas from the preceding 30 days of hourly returns.
- At each hourly decision, node return is the median member one-hour BTC-residual return. Node
  breadth is the fraction of members with the same sign as that median. Node volatility is the
  median absolute member one-hour residual return.
- Node return and volatility z-scores use only their preceding seven days, shifted one hour, with at
  least five days and clipping to `[-5, 5]`.
- A source release is active only when absolute return z-score is at least 2, volatility z-score is
  at least 1, breadth is at least 65%, and at least five members are observed.

## Frozen monthly graph

- Train on the preceding 30 days and stop four hours before the target month.
- Sample every four hours so four-hour targets do not overlap; require 150 complete observations.
- Source is node return z-score; target is follower-node future four-hour median BTC-residual
  return.
- For positive transmission, retain correlations above zero whose magnitude exceeds the reverse
  pair. For reversal transmission, retain correlations below zero whose magnitude exceeds the
  reverse pair. Shrink by `sqrt(n/(n+150))` and keep two leaders per follower and relation sign.

## Frozen candidates and portfolio

1. `CVG1_COMMUNITY_VOL_CONTINUATION`: follow positive-relation edges in the release direction.
2. `CVG2_COMMUNITY_VOL_REVERSAL`: follow negative-relation edges opposite the release direction.

Follower pressure is the edge-weighted signed excess above the source return-z threshold. At least
two follower communities and five follower symbols must activate. Select the two strongest
follower communities, include every available member equally, and apply the community-specific
long/short direction. Use a candidate-level four-hour cooldown.

The primary endpoint is the four-hour BTC-residual bucket after 40 bp total round-trip cost. Naked
net20 and net30 are secondary.

## Controls and promotion

- Fifty within-month random-community graphs preserve follower indegree, edge-weight slots, and
  relation sign.
- Reversed edges and a 24-hour delayed source state are mandatory controls.
- Two-thousand entry-day block bootstrap draws use primary residual net40.
- Compare both real candidates to the per-iteration random family maximum and require the 95th
  percentile.
- Promotion requires at least 150 full, 40 validation, and 40 holdout observations; positive
  residual net40 and naked net20 in development, validation, and holdout; positive full naked
  net30; positive bootstrap lower bound; superiority to reversed and delayed controls; positive
  month contribution concentration at most 35%; and worst period mean no worse than -40 bp.

Passing grants forward-shadow candidacy only. No PaperLive, leverage, or live-order permission is
granted.

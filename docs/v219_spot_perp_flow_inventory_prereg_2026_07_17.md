# v21.9 Spot-Perpetual Flow-Inventory Preregistration

Status: `frozen_before_return_reveal`.

Candidate-feature SHA256:
`50D24B224D45F1498626D8C044CEE4C608D9AD6471FC62457B7F34477D11BDB3`.

## Hypothesis

Aggressive spot demand is more inventory-like than aggressive perpetual demand.
After controlling each venue's symbol-specific history, names with unusually
strong spot taker flow relative to perpetual flow should outperform names whose
apparent demand is concentrated in leveraged perpetuals.

## Frozen candidates

- `SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE`: at 00/12 UTC, long up to eight symbols
  whose causal `spot flow z - perpetual flow z` score is at least +1 and short up
  to eight whose score is at most -1; require at least five names per leg.
- `SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE`: within each forward-frozen monthly
  graph community, long the highest and short the lowest score when the high is at
  least +0.5, the low at most -0.5, and their gap at least 1.5; require at least
  four community pairs.

Both venue z-scores use only the preceding 20-30 days.  Both hourly quote volumes
must be at least half their strictly prior seven-day median.  The feature is known
at the hourly close; entry waits the following complete one-hour bar.

## Portfolio and execution

- Trade Binance USD-M perpetual returns; spot data are information only.
- Allocate +0.5 equally to the long leg and -0.5 equally to the short leg.
- Estimate BTC beta from perpetual one-hour returns in the 30 calendar days
  strictly before the event month, requiring at least 480 paired observations.
- Add a BTC hedge for residual beta and normalize total gross notional to one.
- Primary holding period is 12 hours, so the 00/12 UTC schedules do not overlap
  after the one-hour execution delay.
- Primary/stress round-trip book costs are 20/40 bp on unit gross.

## Frozen controls

- Additional one-hour entry delay.
- Four-hour and 24-hour holding horizons.
- Reversed long/short direction.
- The same names and direction shifted +24 hours.
- 500 same-event random controls: globally random disjoint legs with identical
  counts for SFI1; independent high/low orientation flips within each exact graph
  community pair for SFI2.
- 2,000 UTC-day block-bootstrap paths.
- Month positive-PnL concentration and maximum symbol selection share.

## Eligibility gates

Each candidate separately requires:

1. At least 300 realized events, at least 60 in development, validation, and
   holdout, and all ten active months.
2. Mean gross return above 20 bp and 20 bp net return positive in all three periods.
3. Delayed-entry 20 bp net positive and reversed-direction 20 bp net negative.
4. Relevant random-control percentile at least 0.95 and day-block bootstrap lower
   95% net bound positive.
5. The +24-hour shifted placebo 20 bp net non-positive.
6. No month contributes more than 35% of positive aggregate net PnL and no symbol
   appears in more than 35% of event legs.

Passing only permits natural-forward research consideration.  It does not change
PaperLive, live, application, leverage, remote, or order permissions.

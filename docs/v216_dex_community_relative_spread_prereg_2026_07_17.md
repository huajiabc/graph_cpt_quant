# v21.6 DEX Community Relative-Spread Preregistration

Status: `frozen_second_stage_before_spread_return_reveal`.

Source feature SHA256:
`2079E2D0B8915E4B6601189F2B8002496EDD1E6DC324DD431A575D61C3761B5F`.

This is a bounded second-stage diagnostic motivated by the already revealed v21.3
laggard-versus-all difference.  Therefore, the same historical sample cannot
establish independent alpha evidence or authorize deployment even if every gate
passes.

## Frozen signal and portfolio

- Use the v21.2 DEX source event, CEX confirmation, feature close, community
  cooldown, forward-frozen monthly membership, and source-return direction without
  alteration.
- At the feature close, rank all available peers by their observed one-hour return
  in the source direction.
- Put the slowest half in the catch-up leg and the fastest equal-sized half in the
  leader leg; discard an odd middle name.  Require at least two names per leg.
- Raw weights are +0.5 times the source direction across laggards and -0.5 times
  the source direction across leaders.  This is dollar neutral before hedging.
- Estimate symbol BTC beta from the 30 calendar days strictly before the event
  month, requiring 2,000 paired 15-minute observations.  Add a BTC hedge for the
  remaining beta exposure and normalize to unit gross.
- Enter one complete 15-minute bar after the feature close; hold 12 hours.
- Charge 20 bp round-trip primary and 40 bp stress book cost on unit gross.

## Frozen sample and controls

Development is Aug-Nov 2025, validation Mar-Apr 2026, and holdout May 2026 onward.
The Dec 2025-Feb 2026 DEX-vendor transition remains visible but excluded from the
eligibility estimand.

Controls are:

- 15-minute additional entry delay;
- four-hour and 24-hour horizons;
- reversed spread direction;
- the same frozen spread shifted +24 hours;
- 500 same-event random peer-rank assignments preserving community, leg size,
  direction, beta hedge, timing, and cost;
- 2,000 UTC-day block-bootstrap paths;
- month and source-token positive-PnL concentration.

## Economic-interest gates

The diagnostic is only worth natural-forward observation if all gates pass:

1. At least 200 realized events, 30 per eligible chronological period, and eight
   active months.
2. Mean gross return above the 20 bp hurdle and primary net positive in all three
   periods.
3. Delayed-entry primary net positive and reversed-direction primary net negative.
4. Random-rank percentile at least 0.95 and bootstrap lower 95% primary-net bound
   positive.
5. The +24-hour shifted placebo primary net is non-positive.
6. No month or source token exceeds 35% of positive aggregate primary net PnL.

Passing yields only `research_only_requires_new_natural_forward`.  No PaperLive,
live, application, leverage, remote, or order permission is implied.

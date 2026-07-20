# v14.5 Bearish Convergence Horizon-Extension Preregistration

Date frozen: 2026-07-15, after inspecting v14.3/v14.4 and before computing any 18-hour or 24-hour
v14.5 portfolio return.

## Adaptive status and fixed mechanism

This is an adaptive horizon extension, not independent discovery. It reuses v14.3 exactly:

- the causal monthly volatility-receiver graph;
- source return z at most -2, volatility z at least 1, negative breadth at least 65%, and five
  members;
- quiet receiver absolute return z at most 0.75, volatility z at most 0, and five members;
- at least two quiet graph followers, selecting the strongest source opportunity and two strongest
  receivers;
- a 50/50 spread long the bearish-shocked source community and short the receiver communities.

No event threshold, graph rule, or ranking rule changes.

## Frozen candidates

1. `BCH1_BEARISH_CONVERGENCE_18H`, with an 18-hour cooldown;
2. `BCH2_BEARISH_CONVERGENCE_24H`, with a 24-hour cooldown.

Primary return is BTC-residual net40; naked net20/net30 are secondary. The question is whether the
4h-to-12h accumulation continues far enough to cover conservative cost.

## Controls and gates

- Fifty random-follower graphs preserve monthly leader outdegree and edge weights.
- Reversed edges and a 24-hour delayed source signal are mandatory controls.
- The random family maximum spans both horizons; require the 95th percentile.
- Two-thousand entry-day block bootstrap draws use residual net40.
- Provisional passage requires at least 50 full, 12 validation, and 12 holdout observations;
  positive residual net40 and naked net20 in development, validation, and holdout; positive full
  naked net30; positive bootstrap lower bound; superiority to reversed and delayed controls;
  positive month concentration at most 35%; and worst period mean no worse than -40 bp.

Because the horizon choice is adaptive, a historical pass grants forward-shadow candidacy only.
No PaperLive, leverage, or live-order permission is granted.

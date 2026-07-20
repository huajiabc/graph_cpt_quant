# v21.0 Funding-Settlement Rebound Preregistration

Status: `frozen_before_post_settlement_return_reveal`.

## Frozen inputs and event timing

- v20.9 funding-symbol features SHA256:
  `04BFBE67828D8CAC0C6771750C9F3C04BA3B42296A3DAD2431084ECCA03B8DC7`.
- v20.9 candidate events SHA256:
  `074C9D0D78B208D12180D7C4A5625A099B2FEB2883EEB362B755A943DA537934`.
- FSE1 has 813 frozen feature events; FSE2 has 505.
- A synchronized funding settlement occurs at 00/08/16 UTC with all 45
  research-universe alts present.
- The funding rate is observed at settlement time `t`. Primary entry is the
  fully closed 15-minute bar at `t+15m`; primary exit is `t+75m`.
- No funding cash flow is credited because the observed settlement precedes
  entry and the next settlement is outside the primary holding window.

## Frozen candidates

### FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND

At each frozen event, long every alt whose just-settled Binance USD-M funding
rate is negative. Require at least five alts.

### FSE2_NEW_NEGATIVE_ONSET_REBOUND

Long every alt whose just-settled rate is negative and whose immediately prior
settled rate was non-negative. Require at least five alts.

For both candidates, equal-weight the selected alts, add a BTC hedge that
offsets the basket's causal prior-30-day beta, then normalize total gross
notional including BTC to one. Monthly beta requires at least 2,000 prior
15-minute observations.

## Frozen economics and robustness

- Primary/stress round-trip book costs: 20/40 bp.
- Fixed cost frontier: 5/10/20/40 bp.
- One-bar additional entry delay: enter `t+30m`, hold 60 minutes.
- Holding controls from the primary entry: 30 and 120 minutes.
- Reversed-direction control for every event.
- 500 same-settlement random-basket iterations. Each random basket draws the
  same number of alts from the 45-name universe, then applies the same causal
  BTC-beta hedge and gross-one normalization.
- 2,000 settlement-day block bootstrap draws, sampling UTC days with
  replacement so the three within-day settlements are not treated as fully
  independent.
- Seed: `21000`.
- No severity, breadth, hour, hedge-ratio, entry, or holding-period grid is
  allowed.

## Decision gates

Each candidate must independently pass all gates:

1. At least 400 events overall, 100 in each frozen period, and 10 active months.
2. Mean gross return exceeds the 20 bp primary cost.
3. Primary net mean is positive in development, validation, and holdout.
4. Primary net mean is positive separately at 00, 08, and 16 UTC settlements.
5. One-bar additionally delayed primary net mean is positive.
6. Observed gross mean is at or above the 95th percentile of the frozen
   same-settlement random-basket distribution.
7. Reversed-direction primary net mean is negative.
8. The settlement-day bootstrap 95% lower bound for primary net mean is
   positive.

Passing every gate creates only an offline research candidate requiring
untouched natural-forward evidence. It does not authorize live, PaperLive,
leverage, application, or order changes.

At preregistration time, no post-settlement return or candidate PnL had been
calculated.

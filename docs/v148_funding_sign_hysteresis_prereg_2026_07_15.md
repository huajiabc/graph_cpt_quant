# v14.8 Funding-Sign Hysteresis Preregistration

## Motivation

v14.7 FSS1 passed every economic and robustness gate but missed the frozen mean
one-way turnover ceiling by 0.0095 (0.7595 versus 0.75). This follow-up changes
only the causal sign-transition rule; it does not search funding thresholds,
cross-sectional ranks, or return-conditioned filters.

## Frozen candidate

`FSS2_TWO_WEEK_SIGN_CONFIRMATION`

- Weekly Monday entry and seven-day holding horizon.
- A newly eligible symbol immediately takes its contemporaneous seven-day
  funding-score sign.
- An existing symbol changes side only after two consecutive weekly observations
  with the opposite non-zero sign.
- Same-sign and zero observations reset a pending flip; zero retains the current
  side. Missing or ineligible symbols exit immediately.
- Long all retained negative-side symbols and short all retained positive-side
  symbols, equal weight within each side, with 50/50 raw alternative-asset legs.
- Add the exact causal BTC-beta hedge and rescale total gross notional to one.
- Require at least four names on each side.

## Costs and evaluation

- Primary one-way turnover cost: 20bp; stress cost: 40bp.
- Include initial opening, weekly changes, gaps, and terminal close in turnover.
- Use the unchanged development/validation/holdout labels and contracted/broad
  negative-funding breadth states.
- Four-week moving-block bootstrap with 2,000 draws.
- 1,000 random full-universe portfolios preserving each week's observed long and
  short breadth and charging the candidate's observed turnover.

## Frozen promotion gates

- At least 45 weeks, 11 months, and 10 validation and holdout weeks.
- Mean one-way turnover no greater than 0.75.
- Positive mean funding carry, 40bp stress return, all three time splits, both
  breadth states, and 95% moving-block lower bound.
- At least the 95th random-null percentile.
- Positive-month concentration no greater than 35% and worst split at least
  -40bp/week.
- Maximum absolute residual BTC beta and gross-notional drift no greater than
  1e-12.

No PaperLive, leverage, or real-order permission is granted by this preregistration.

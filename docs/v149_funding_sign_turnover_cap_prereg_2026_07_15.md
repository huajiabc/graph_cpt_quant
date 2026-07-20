# v14.9 Funding-Sign Turnover-Cap Preregistration

## Motivation

v14.7 FSS1 passed every economic and robustness gate except the mean turnover
ceiling (0.7595 versus 0.75). v14.8 showed that changing the signal to require
two-week sign confirmation destroys validation performance. This follow-up keeps
the v14.7 signal and target portfolio exactly unchanged and tests one execution
rule only.

## Frozen candidate

`FSS3_CURRENT_SIGN_070_TURNOVER_CAP`

- The weekly target is exactly v14.7 FSS1: long every negative seven-day funding
  score, short every positive score, equal weight within sign, 50/50 raw
  alternative-asset legs, at least four names per side.
- Add the exact causal BTC-beta hedge and normalize total gross notional to one.
- On each continuous weekly transition, move as far as possible from the previous
  executed portfolio toward the current target while limiting full L1 portfolio
  turnover to 0.70. The path is a convex blend of previous and target
  alternative-asset weights followed by a current-beta BTC hedge and gross-one
  normalization.
- The 0.70 cap is frozen as an operational buffer below the 0.75 mean-turnover
  gate; it is not selected from return results and no alternative cap is tested.
- Initial opening, terminal close, mandatory exits, and data gaps are not hidden by
  the cap and remain fully charged in reported turnover.
- Weekly Monday entry and seven-day holding horizon.

## Costs and null

- Primary one-way turnover cost: 20bp; stress cost: 40bp.
- Four-week moving-block bootstrap with 2,000 draws.
- 1,000 random full-universe paths preserve each week's observed negative and
  positive breadth, use the identical 0.70 execution rule and BTC hedge, and pay
  their own realized turnover. There is one frozen candidate, so no post-hoc
  parameter-family maximum is needed.

## Frozen promotion gates

- At least 45 weeks, 11 months, and 10 validation and holdout weeks.
- Mean fully charged turnover no greater than 0.75; all cap-applicable transitions
  no greater than 0.70 plus 1e-10 numerical tolerance.
- Positive mean funding carry, 40bp stress return, development, validation,
  holdout, contracted and broad states, and 95% moving-block lower bound.
- At least the 95th random-null percentile.
- Positive-month concentration no greater than 35%; worst time split at least
  -40bp/week.
- Maximum absolute residual BTC beta, gross-notional drift, and cap breach within
  numerical tolerance.

No PaperLive, leverage, or real-order permission is granted by this preregistration.

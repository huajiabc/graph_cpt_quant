# v11.8 Crowding-Unwind Volatility Transmission Preregistration

Date: 2026-07-15

Status before outcome inspection: `PREREGISTERED`.

## Question

Does an independently measured position-crowding unwind propagate through the
frozen monthly 8-by-9 price communities strongly enough to trade follower
returns after costs?

## Frozen sources and timing

- Bybit hourly long/short account ratios from
  `/v5/market/account-ratio`.
- Existing Bybit 15-minute klines and open-interest-derived feature panel.
- Frozen v11.0 monthly 8-by-9 community memberships; they are not rebuilt from
  the new outcome.
- An account-ratio observation may be used only at or after its timestamp and
  only when its age is no more than 90 minutes.
- Entry is after the completed signal hour; outcome is the next four-hour
  return already defined by the common research panel.

## Frozen state variables

For each symbol and hour:

- `crowding = log(long_account_ratio / short_account_ratio)`;
- crowding z-score from the preceding 30 calendar days, shifted one hour;
- one-hour return z-score from the preceding 30 days, shifted one hour;
- existing one-hour OI-value-delta z-score.

Minimum rolling history is 20 days. Winsorization is fixed at plus/minus five
z-score units.

## Frozen candidates

`CU1_CROWDED_LONG_UNWIND_SHORT` leader condition:

- crowding z-score at least +2.0;
- return z-score at most -1.5;
- OI delta z-score at most -1.0.

Short the other members of the same frozen community for four hours.

`CU2_CROWDED_SHORT_SQUEEZE_LONG` mirrors the condition:

- crowding z-score at most -2.0;
- return z-score at least +1.5;
- OI delta z-score at most -1.0.

Long the other members for four hours.

If multiple leaders fire in a community-hour, their follower set is counted
once. Portfolios require at least three finite followers, use equal weights,
and enforce a four-hour per-candidate global cooldown. BTC is not a follower.

## Frozen evaluation

- Development: feature times before 2026-01-01.
- Validation: 2026-01-01 through 2026-03-31.
- Holdout: 2026-04-01 onward.
- Focal raw round-trip cost: 20 bp; stress cost: 30 bp.
- BTC-residual result uses betas estimated only from each preceding 30-day
  history and charges 40 bp for the two-leg diagnostic.
- Controls: 50 exact-size random 8-by-9 partitions per month, one-day shifted
  account-ratio signal, reversed direction, chronological fifths, and
  day-block bootstrap.

## Promotion gate

A candidate is rejected unless it has at least 100 full observations, 20
validation observations, and 20 holdout observations; positive raw net20 in
all three periods; positive residual net40 in validation and holdout; positive
bootstrap lower bound; at least the 95th random-partition percentile; and no
single month or community above 35% of positive PnL.

No threshold refinement, model fitting, leverage, or PaperLive deployment is
allowed after seeing these outcomes. Any clue that fails the gate is only a
new forward hypothesis.

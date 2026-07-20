# v23.1 Book-Vacuum Synthetic Straddle Preregistration

Status: `FROZEN_BEFORE_POST_ENTRY_OUTCOME_LOAD`.

## Question

The v22.4 book-vacuum events raised subsequent BTC realized variance, but that is
not automatically tradable alpha because option markets charge implied
volatility, theta, spreads, and fees. This test asks a narrower question: does
the post-event BTC move, by itself, reprice a causal-IV ATM straddle enough to
clear a fixed premium-return hurdle?

## Frozen input

- Feature artifact:
  `reports/v23_0_book_vacuum_implied_variance_feature_audit/causal_implied_variance_features.parquet`.
- SHA-256:
  `88A24C37339AEB9F26A272E14794375789026D2426A2730949EFA2B45B178C0D`.
- Events: the 123 v22.4 events passing the v23.0 causal surface gate.
- Deribit ATM IV is taken from the latest completed daily trade surface known by
  entry, no more than 72 hours old, using the quality-passing 7--45 DTE row
  closest to 21 DTE.
- Entry BTC is the Bybit completed-hour close at the event time.
- No event is selected using any post-entry option trade, mark, BTC move, or
  outcome availability.

## Frozen synthetic contract and repricing

For each event at time `t`:

- strike equals BTC spot at `t`;
- expiry is the causal surface row's actual expiry;
- volatility is the causal ATM IV and remains constant through each horizon;
- the contract is one inverse-BTC call plus one inverse-BTC put;
- the entry and exit BTC option values use the repository's deterministic
  inverse Black--Scholes implementation, with remaining maturity reduced by the
  elapsed horizon;
- BTC option values are converted to USD using the contemporaneous BTC spot;
- gross premium return is `exit USD straddle / entry USD straddle - 1`.

This is deliberately a movement-sufficiency diagnostic. It does not claim that
historical option bids or asks were available, and it gives no credit for a
post-event IV increase. Conversely, stale daily IV can understate the true
event-time hurdle.

## Frozen horizons, friction hurdles, and variance diagnostic

- Primary horizon: 4 hours.
- Secondary horizons: 1 and 8 hours.
- Primary friction hurdle: 1.00% of entry premium round trip.
- Stress friction hurdle: 2.00% of entry premium round trip.
- For each horizon, also report the sum of squared hourly BTC log moves divided
  by the causal implied-variance budget `IV^2 * hours / (365.25 * 24)`.
- Report all/development/validation/holdout and positive-/negative-pressure
  strata without changing the frozen event set.

The premium hurdles are research stress assumptions, not reconstructed
historical spreads or commissions.

## Frozen matched-time control

Construct an hourly BTC universe over the event sample with the same causal
surface and trailing-24-hour variance features. A control time must:

- be in the same calendar month and exact UTC hour as its event;
- have a surface no more than 72 hours old and a valid 8-hour future BTC path;
- lie more than 8 hours from every frozen v22.4 event;
- have the same causal information definitions as the event.

Within each event's eligible set, rank controls by
`abs(log(control IV / event IV)) + abs(log(control prior variance / event prior variance))`
and freeze the nearest 10. At least five controls are required for a matched
event. Draw one control per event for 1,000 deterministic random paths. The
event/control comparison uses only matchable events.

## Frozen uncertainty and decision rule

- Resample entry months with replacement for 5,000 month-block bootstrap means.
- Seed: `20260717`.
- A movement-sufficiency result is positive only if:
  1. mean 4-hour premium return after the 1% hurdle is positive overall and in
     development, validation, and holdout;
  2. the month-block 95% lower bound is above zero;
  3. the matched-event mean is at or above the 90th percentile of the 1,000
     matched random-time path means; and
  4. mean 4-hour realized/implied variance ratio exceeds one.

Failure of any gate rejects the movement-sufficiency claim. Passing all gates
still yields only a research result: absent historical bid/ask and synchronized
two-leg execution, this branch cannot be promoted to PaperLive or live options.

No live, PaperLive, leverage, remote, application, or order state may change.

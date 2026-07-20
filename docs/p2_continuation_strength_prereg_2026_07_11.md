# P2 Continuation Strength - Pre-Registration

## Question

Does a fixed, continuous measure of P2 continuation strength show a monotonic
relationship with later portfolio net20 returns when the independent unit is a
burst rather than an individual token trade?

## Frozen Population

- Portfolio: `P2_MAX8_BASELINE` only.
- Candidate pool: CIC1 + CIC2, deduplicated by `trade_id`.
- Entry and exit rules remain unchanged.
- No O6, CP60, Protect_A, router, order-flow, or token-attention action is used.

## Fixed Score

The score is the equal-weight mean of four components, each clipped to `[0, 1]`:

1. Market breadth: `(volume_impulse_density_at_signal - 0.10) / 0.20`.
2. Beta continuation: `(beta_extension_score_at_signal - 80) / 20`.
3. Local shock: `(local_volume_shock_strength_at_signal - 2) / 4`.
4. Reclaim speed: `1 - minutes(pullback_time, reclaim_time) / 120`.

At least three components must be present. No weights, clipping bounds, or bin
edges may be optimized after results are observed.

Fixed score bins: `[0,.2)`, `[.2,.4)`, `[.4,.6)`, `[.6,.8)`, `[.8,1]`.

## Independent Unit And Outcome

- Independent unit: existing as-of one-hour `burst_id`.
- Burst score: mean score across selected core trades.
- Burst outcome: sum of core `net_return_20bp` divided by max positions (8).
- The final burst size is not an input.

## Periods

- Search reference: entry time before 2026-02-01.
- Validation reference: 2026-02-01 through 2026-04-30.
- Legacy holdout reference: entry time on or after 2026-05-01. This period has
  already been observed and cannot approve the hypothesis.
- Decision sample: only rows from the cumulative ledger with
  `timely_forward_observation=true` first observed after this pre-registration.

## Acceptance Gate

No promotion decision before both conditions hold:

- at least 100 timely forward core trades; and
- at least 30 timely forward bursts.

At the decision point, require:

- positive mean burst net20;
- positive score-vs-return slope;
- Spearman correlation above zero;
- bootstrap 95% interval for the slope not crossing zero;
- permutation p-value <= 0.10;
- no single month contributing more than 35% of positive net;
- positive 30bp cost stress.

Failure at the decision sample demotes the continuous score to diagnostic-only.
It must not trigger a new threshold search.

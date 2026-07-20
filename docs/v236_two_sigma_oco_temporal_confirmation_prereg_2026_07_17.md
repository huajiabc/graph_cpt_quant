# v23.6 Two-Sigma OCO Temporal Confirmation Preregistration

Status: `FROZEN_BEFORE_TWO_SIGMA_HOLDOUT_LOAD`.

## Selection record

The v23.4 preregistered 1.0-sigma OCO failed. Its frozen secondary widths showed
improvement from -9.27 bp/event at 0.75 sigma and -7.96 bp at 1.0 sigma to -0.89
bp at 1.25 sigma over the full sample. A post-v23.4 parameter-discovery scan was
then restricted to development and validation only, using widths 1.25, 1.50,
1.75, 2.00, 2.25, and 2.50.

Two adjacent widths formed a plateau:

- 2.00 sigma: +1.30 bp/event in development, +15.12 bp in validation, 67
  triggers across the two periods;
- 2.25 sigma: +3.27 bp/event in development, +12.37 bp in validation, 60
  triggers.

The 2.00-sigma rule is frozen because it is the lower-complexity, higher-coverage
member of that plateau. The 1.50--2.50-sigma holdout outcomes were not loaded
during selection. This is a temporal confirmation, not a fully untouched
end-to-end preregistration, because v23.4 had already reported holdout at widths
through 1.25 sigma.

## Frozen feature and execution

- Reuse the v23.3 feature artifact with SHA-256
  `20E29DC3BCF5E8702E46AE3B21B8F900BD26C42A6E43F7D7B488C847250DD828`.
- Use all 159 events; no pressure-sign or direction filter.
- Symmetric barriers are `spot * exp(+2 * causal hourly sigma)` and
  `spot * exp(-2 * causal hourly sigma)`.
- Reuse the exact v23.4 first-trigger OCO, pessimistic gap fill, pessimistic
  same-bar ambiguity, 4-hour exit, and zero return/cost when untriggered.
- Primary/stress round-trip costs remain 10/20 bp of traded notional.

## Frozen controls and uncertainty

- Reuse the v23.4 feature-only matched pools: same month, exact UTC hour,
  nearest trailing causal sigma, and more than eight hours from every event.
- Re-simulate both event and control paths at 2.00 sigma.
- Produce 1,000 matched random paths and 5,000 month-block bootstrap paths with
  seed `20260717`.
- Report development and validation as selection-period diagnostics, but no
  longer count them as new confirmation evidence.

## Frozen confirmation gates

The two-sigma candidate is supported only if:

1. at least 20 holdout events trigger;
2. holdout mean primary net return per event is positive;
3. full-sample mean primary net return per event is positive;
4. full-sample month-block 95% lower bound is above zero;
5. holdout event mean reaches at least the 90th percentile of holdout matched
   random-time means;
6. full-sample event mean reaches at least the 90th percentile of full-sample
   matched random-time means;
7. ambiguous-trigger fraction is no more than 10%; and
8. full-sample primary return exceeds the reversed-direction control.

Failure of any gate rejects the two-sigma candidate. Passing remains
research-only and does not authorize PaperLive, live, leverage, remote,
application, or order changes.

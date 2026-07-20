# Execution Plan - 2026-07-11

## Objective

Turn the current research stack into an auditable forward experiment before
opening any new alpha search. Real-live and canary-live remain disabled.

## Execution Order

### P0 - Workspace and source-of-truth cleanup

- [x] Preserve existing governance documents and server-loop edits.
- [x] Remove reproducible Python and pytest caches only.
- [x] Keep historical replay outputs separate from live forward outputs.
- [x] Make the governance benchmark and the executable primary distinction
  explicit in status artifacts.

### P1 - Trustworthy forward evidence

- [x] Add a cumulative forward ledger with stable signal/trade identifiers.
- [x] Preserve first-observed and last-observed timestamps.
- [x] Mark bootstrap/backfilled observations separately from genuinely timely
  forward observations.
- [x] Write per-run immutable snapshots and a run manifest.
- [x] Ensure promotion audits can consume the cumulative live ledger without
  reading a rolling seven-day replay by accident.

### P2 - Operational reliability

- [x] Resolve Python from the project virtual environment when available and
  fall back to the active system Python otherwise.
- [x] Stop dependent jobs when the primary refresh fails.
- [x] Remove duplicate S2 shadow refreshes; keep all short work diagnostic-only.
- [x] Fail closed when prepared market data is stale.
- [x] Expose rolling-return and baseline-lift stop decisions as explicit live
  gate artifacts, activating them only after their configured sample window.

### P3 - Validation

- [x] Add regression tests for replay/live namespace separation.
- [x] Add tests for cumulative deduplication, first-observed timestamps, and
  timely-forward eligibility.
- [x] Add tests for stale-data and sample-gated pause decisions.
- [x] Run the full test suite.

Validation result: `397 passed`; remaining warnings are four existing NumPy
constant-series correlation warnings in the short Path-C test.

### P4 - Alpha research after P0-P3

Run one pre-registered hypothesis at a time, in this order:

1. Continuous P2 continuation-strength score, evaluated at burst/day level.
2. CP60/F3/F5 state-dependent risk management with opportunity-cost accounting.
3. One-minute reclaim-window execution quality using truly as-of microstructure.
4. One orthogonal information source, beginning with token-attention context.
5. Portfolio correlation and burst exposure budgeting.

Do not reopen broad threshold mining, static order-book ranking, or automated
short research without a new information source and a pre-registered test.

Current status:

- [x] Item 1 pre-registered, implemented, and run on isolated historical replay.
- [ ] Item 1 promotion decision pending 100 timely trades and 30 timely bursts.
- [x] Item 4 is implemented as counterfactual-only token context with as-of
  watermarks, placebo controls, canonical-network mapping governance, and a
  cumulative forward ledger. Promotion remains sample-blocked.
- [x] Item 5 is implemented as fixed EW/VOL/BETA/CORR shadows with strictly
  as-of correlation clusters and exposure accounting. Promotion remains
  sample-blocked.
- [ ] Items 2-3 action-rule work remains blocked until the cumulative forward
  ledger has fresh, timely completed trades; implementing those rules before
  then would recreate the multiple-testing problem this plan is intended to
  stop.

## Hard Decision Gates

- Core P2: initial decision at 100 timely forward trades; stronger review at 200.
- CP60: initial decision at 50 timely exits; stronger review at 100.
- O6: initial decision at 30 timely overflow trades; stronger review at 50.
- Protect_A: initial decision at 30 timely protected exits; stronger review at 50.
- Demote a component when its pre-registered forward delta is non-positive at
  the relevant sample threshold.
- Real-live remains disabled until execution realism and the risk envelope pass.

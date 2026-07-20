# Orthogonal Source and Portfolio-Layer Findings (2026-07-11)

Status: historical diagnostic plus forward-counterfactual logging only. No live or canary permission is granted. P2_EW max-8 remains the primary forward paper ledger.

## Frozen evaluation scope

- Portfolio universe: deduplicated P2 candidate pool, first-come ordering, maximum eight concurrent positions.
- Cost views: 10/20/30 bp, with 20 bp as the reference view.
- Token attention: entry-time as-of visibility, 65-minute publication delay, 48-hour stale watermark, 7-day shifted placebo, and deterministic same-chain random-token control.
- Parameters were frozen before reading the replay output; the results below are not used to retune them.

## Portfolio-layer replay

| Arm | Selected | Skipped | Position units | Net20 / 8-unit capital | Max exposure | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| P2_EW | 147 | 48 | 147.0 | 10.87% | 8.00 | Frozen reference |
| P2_VOL | 147 | 48 | 124.5 | 9.36% | 6.67 | Preserved 86% of EW net20 while using 85% of position units |
| P2_BETA | 106 | 89 | 99.63 | 2.15% | 5.37 | The beta cap is binding and removes too much return in this replay |
| P2_CORR | 147 | 48 | 147.0 | 10.87% | 8.00 | Initially input-blocked; the 2026-07-12 as-of rebuild below shows the frozen cap is non-binding |

These figures are historical diagnostics, not forward evidence. In particular, P2_VOL is only promising enough to keep logging; it is not eligible to replace P2_EW.

### P2_CORR as-of cluster follow-up (2026-07-12)

The pre-registered monthly correlation membership is now implemented. Overall
membership coverage is 77.2%, and selected-trade cluster coverage is 69.4%.
The first month remains uncovered because no seven-day pre-month history exists.

`P2_CORR` still selected the same 147 trades as `P2_EW`, but it is now an
evaluable no-op rather than a missing-input no-op: no historical opportunity
exceeded the frozen maximum of two simultaneous positions in one covered
cluster. The threshold and cap were not changed after observing this result.

## Token-attention replay

The deduplicated P2 comparison contains 195 entries. Mapping coverage is 46.2%, and 72.8% of entries are stale under the frozen 48-hour watermark. Therefore the usable covered-and-fresh subset is only 53 entries.

Within those 53 entries:

| Split | Event n / no-event n | Mean net20 with event | Mean net20 without event | Raw spread |
|---|---:|---:|---:|---:|
| Real token prior-24h | 42 / 11 | 1.362% | 1.010% | +0.352 pp |
| 7-day shifted placebo | 30 / 23 | 1.176% | 1.437% | -0.260 pp |
| Same-chain random token | 11 / 42 | 1.440% | 1.250% | +0.190 pp |

The sign pattern is mildly encouraging because the real-event spread is larger than the random-token spread and the shifted placebo has the opposite sign. It is still not decision-grade: the usable sample is small, the real-event base rate is high, and these are unadjusted conditional means rather than an independent forward test.

## Current decisions

- Keep P2_EW as the primary forward paper portfolio.
- Keep P2_VOL as a counterfactual shadow; require new forward observations before judging it.
- Keep P2_BETA as a diagnostic of concentration, not a promotion candidate under the frozen cap.
- Keep P2_CORR as a non-binding shadow. Its input is no longer blocked, but the
  frozen concentration cap generated zero constrained decisions in replay.
- Keep token attention diagnostic-only and explicitly log mapping gaps, stale watermarks, shifted placebo, and same-chain random controls.

## Live-universe token coverage follow-up (2026-07-12)

- Conservative A/B coverage for the current 50-symbol live universe increased
  from roughly 15 symbols to 26 symbols (52%).
- Automatic B promotion now requires a registered canonical network, exact
  token symbol, at least USD 25,000 top-pool 24h volume, and at least 3x
  dominance over the next eligible same-symbol token candidate.
- An initial broader rule incorrectly surfaced wrapped or unrelated cross-chain
  assets for BCH, DASH, LTC, NEAR, ORDI, WLD, and ZEC. Those rows were audited,
  rejected, and downgraded before deployment.
- The incremental OHLCV refresh targeted 33 mapped historical/live symbols:
  32 returned data and OP remained an explicit HTTP-error row. SOL returned
  only five recent rows and therefore remains subject to the 48-hour freshness
  guard rather than being treated as a clean no-event observation.

## Forward monitoring follow-up (2026-07-12)

The cumulative ledger now also persists token context and risk-shadow skips.
The monitoring report separates timely P2 opportunities into observed, market
gate passed, pullback, entry, portfolio accepted, and completed stages, and
tracks every pre-registered sample gate without changing any strategy rule.

## Validation

- Full test suite: 407 passed.
- Ruff checks: passed.
- Paper-live PowerShell parser check: passed.
- Remote deployment: completed after the SSH tunnel recovered. The scheduled loop completed token refresh, v07d2, and health checks with return code 0; the cumulative manifest was migrated to the 19-column schema and is readable.

### Remote closure (2026-07-12)

- The cumulative manifest is now 23 columns and readable after adding token
  context and risk-skip ledgers.
- The scheduled cycle completed `v93`, `v65`, `v07d2`, and the health check
  with return code 0.
- July live correlation-cluster coverage is 100% across the 49 symbols with
  current-month prepared data.
- The forward funnel currently contains six timely unique P2 opportunities;
  all six stopped at the frozen market gate, so timely completed P2 trades
  remain 0/100.
- API probe and data freshness passed, actionable entries are zero, and exactly
  one scheduled loop process is running.

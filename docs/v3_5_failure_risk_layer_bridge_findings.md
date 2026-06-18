# v3.5 Failure Risk Layer Bridge — findings

**Status: research only, no live wiring.** A100 production run completed against
the v0.9D capacity trade cache; this doc captures the actual numbers and the
v3.6 hand-off list. v3.7 (re-investigate short) does NOT need to run — every
F-cell that wins on the managed stack wins via long-side discipline, not by
exposing a new short edge.

## Setup

- Pool: `P2_CIC1_CIC2_COMBINED`, max_positions=8.
- Universe: top-30 dynamic rank symbols (v1.2s2 / v3.3 convention).
- Motif set: `S1`, `S3`, `S5` (S2 excluded — mistimed in v1.2s).
- Cooldown: 48 bars (12h).
- O6 policy: `O6_late9_slots4_cic1_050_cic2_025`.
- CP60: 4 bars (1h) pre-signal window, |close[i]/close[i-4]-1| ≤ 0.5%.
- Protect_A cap2: at most 2 concurrent `gate_Protect_A=True` longs across baseline + overflow.
- Strict as-of: a long is only gated by confirmations with `feature_time ≤ signal_time`.
- Underlying `_epoch_ns` patched to force ns precision (pandas 3.x defaults to
  `datetime64[us, UTC]` and would have widened cooldown windows by 1000×).

## Headline

The current managed long stack (B1 = P2 max8 + O6 ≡ B3 = + Protect_A cap2 + O6;
Protect_A is unbound in this universe, see §F4 below) — the v3.5 picture:

| Action | B0 P2 max8 | B1 + O6 | B2 + CP60 + O6 | B3 + Protect_A cap2 + O6 |
|---|---|---|---|---|
| F0 (baseline) | net 10.93%, dd −4.08%, ret/dd 2.68 | 12.25%, −4.28%, 2.86 | 7.10%, −5.22%, 1.36 | 12.25%, −4.28%, 2.86 |
| F1 S1/S3/S5 sym no-long | 11.92%, −2.50%, 4.77 (37 gated) | 12.90%, −2.50%, **5.17** (37) | 9.00%, −3.66%, 2.46 (28) | 12.90%, −2.50%, **5.17** (37) |
| F2 S1-only sym no-long | 11.21%, −3.84%, 2.92 (4 gated) | 12.54%, −4.04%, 3.10 (4) | 7.15%, −4.98%, 1.44 (3) | 12.54%, −4.04%, 3.10 (4) |
| F3 no-overflow only | 10.93%, −4.08%, 2.68 (37) | 12.60%, −4.28%, 2.94 (37) | 7.05%, −5.22%, 1.35 (28) | 12.60%, −4.28%, 2.94 (37) |
| F4 disable Protect_A | 10.93%, −4.08%, 2.68 (0 flagged) | 12.25%, −4.28%, 2.86 (0) | 7.10%, −5.22%, 1.36 (0) | 12.25%, −4.28%, 2.86 (0) |
| F5 CIC2-only no-long | 11.51%, −2.75%, 4.18 (13) | 12.83%, −2.95%, 4.35 (13) | 8.21%, −3.77%, 2.18 (9) | 12.83%, −2.95%, **4.35** (13) |

(All percentages are portfolio net20 / max-drawdown proxy on max8 sizing.)

## Verdicts (8-point pass criteria)

- **F1 on B1/B3** — `shadow`. Net20 +0.65%, max_dd −42% (−4.28% → −2.50%),
  ret/dd jumps to 5.17. 37 gated longs across S1+S3+S5; gated_realized_net_mean
  is *positive* (+0.40%), i.e. the gate cuts some winners too — the DD win comes
  from cutting concentration, not loser-deletion. Sample size adequate
  (criteria 7 OK).
- **F5 on B1/B3** — `shadow`. Net20 +0.58%, max_dd −31% (−4.28% → −2.95%),
  ret/dd 4.35. Only 13 gated longs and the cut hits exactly the noisy CIC2
  branch — `gated_realized_net_mean = −0.35%`, loss_share 54%. The cleanest cut
  in the matrix: smallest sample, smallest gated_realized hit, almost the same
  net20 / dd payoff as F1. This is the v3.6 default candidate.
- **F3 on B1/B3** — `shadow` for the overflow sleeve. Net20 +0.34% with
  *zero* drawdown change vs baseline — the overflow sleeve catches a few
  failure-recent longs that don't move drawdown but do trim net20 when allowed
  in. Free money on the overflow channel; ships alongside F5 without conflict.
- **F2 (S1 only)** — `risk_mode_option`, **not shadow** despite the auto-stamp.
  Only 4 gated longs across the whole sample; net20 + dd both move marginally.
  Criterion 7 (sample size) fails. Hold F2 as a B0-only reference data point;
  the real S1 result is the S1 row inside the F1 motif breakdown — gated_count
  4, avg_net −0.57%, loss_share 75%. F1 already captures the S1 effect.
- **F4 (Protect_A diagnostic)** — `inconclusive`. `protect_a_flagged=0` across
  every cell. Either (a) the `gate_Protect_A` feature is not exercised on the
  top-30 universe (high-beta protection skips low-beta names), (b) the
  per-symbol feature column is absent in this parquet, or (c) the as-of lookup
  has zero overlap with failure_recent. Action-only diagnostic, no PnL effect;
  treat as not-yet-evaluated for v3.6.
- **B2 (CP60 prefilter)** — `reject`. CP60 removes 25 of 146 entries (~17%) and
  drops baseline net20 from 12.25% → 7.10% (−42%), without improving DD. Every
  F-cell on B2 still underperforms the F0 baseline on B1/B3. CP60 in its
  current shape (|ret_1h| ≤ 0.5%) is a bad long-side prefilter; either widen
  the window beyond 1h, raise the stagnation threshold, or drop the layer.

## Motif decomposition (F1 = umbrella, F5 = surgical)

```
F1  S1: 4 gated,  avg_net −0.57%, loss 75%  ← clean losers
F1  S3: 5 gated,  avg_net +2.06%, loss 20%  ← gate is cutting winners
F1  S5: 28 gated, avg_net +0.25%, loss 43%  ← bulk volume, neutral on PnL
F5  S5+CIC2: 6 gated, avg_net −1.72%, loss 67%  ← the cleanest signature
F5  S3+CIC2: 4 gated, avg_net +1.52%, loss 25%  ← noise even when CIC2-narrow
```

The lesson lines up with the v3.3 GA's S1 preference: S1 confirmations on the
same symbol are the cleanest stand-alone failure signal. But once you cut
through CIC type, S5+CIC2 emerges as the single best gate target — bigger
average loss than S1 and three times the sample. F5 picks this up
automatically by narrowing on CIC2; F1 picks it up via the umbrella S1+S3+S5
set at the cost of also blocking S3 winners.

## CIC decomposition (F1 over-gates CIC1)

```
F1  CIC1_beta_extreme: 24 gated, avg_net +0.81%, loss 38%  ← strong-conviction trades being blocked
F1  CIC2_beta_broad:   13 gated, avg_net −0.35%, loss 54%  ← noisy trades being blocked
F5  CIC2_beta_broad:   13 gated, avg_net −0.35%, loss 54%
```

F1's headline win lives despite F1 over-gating the high-conviction CIC1 path.
F5 explicitly skips that over-gating. This is why F5 is the v3.6 default even
though F1 wins on raw net20 by a hair — F1 is leaving CIC1 alpha on the table.

## v3.6 hand-off

1. **Shadow on the live stream**:
   - F5 (CIC2-only no-long, 48-bar cooldown, S1+S3+S5 motifs).
   - F3 (no-overflow only, 48-bar cooldown, S1+S3+S5 motifs).
   - F1 (umbrella) as a comparator only — it's strictly worse than F5+F3 on
     the CIC1 cohort.
2. **Drop B2 entirely**. CP60 in its current 1h-stagnation shape is a bad
   prefilter. Re-engineer or remove before any future use.
3. **Investigate Protect_A coverage** before F4 is meaningful. Confirm
   `gate_Protect_A` is in the v0.9D feature parquet and check whether top-30
   ranks just don't fire it.
4. **Skip v3.7**. v3.4 already closed every short sleeve as no_value; v3.5
   shows the long-side failure overlay is a real edge. The next round is
   v3.6 (live counterfactual on F5+F3), not a re-fight of v3.7.

## Discipline

- Strict as-of: a long is only gated by confirmations with `feature_time ≤ signal_time`.
- O6 simulator backfills freed slots first-come; the SKIP_OVERFLOW channel
  lets a row claim the baseline slot but never an overflow slot.
- Protect_A cap2 is enforced *across* baseline + overflow sleeves; in this
  universe the cap never binds (`protect_a_flagged = 0`).
- No paper-live / real-live permission changes; nothing here touches the
  execution path. This is forward-ledger research only.

## Provenance

- Code: `src/pressure_graph/reports/v3_5_failure_risk_layer_bridge.py`
- CLI: `pressure-graph run-v3-5-failure-risk-layer-bridge --config configs/v0_3.yaml`
- Pipeline: `pipeline.run_v3_5_failure_risk_layer_bridge_from_features`
- Tests: `tests/test_v3_5_failure_risk_layer_bridge.py` (12 cases, all green
  locally and on the A100 conda `quant` env)
- Upstream patch: `_epoch_ns` in `v12s2_long_risk_off_overlay.py` forces ns
  precision; same fix benefits v12s2/v12s3/v3.3 under pandas 3.x.
- Reports: `reports/v3_5_failure_risk_layer_bridge/` — seven CSVs +
  candidate_notes.md, ~9 KB compressed.
- A100 box: see [[a100-ssh-access]] memory note for the SSH chain.

# v3.5 Failure Risk Layer Bridge — findings

**Status (research only, no live wiring):** code committed and unit-tested
locally; full A100 numbers backfilled after the production run finishes against
the v0.9D capacity trade cache. Cell-level verdicts below — placeholders to be
replaced by the post-run snapshot — assume the published v3.3 GA winner
(`S1 / symbol / cooldown=48 / apply_core`) and the v1.2s3 stack shape
(P2 max8 + O6 + CP60 + Protect_A) as starting reference.

The bridge is the answer to "should we keep iterating short alpha, or go deeper
on long risk?" — v3.4 closed every standalone short sleeve as `no_value`, and
v3.3 found a GA-best long-side chromosome. v3.5 stress-tests the question of
**where in the stack** the failure flag should cut, not whether the flag fires.

## Setup

- Long pool: `P2_CIC1_CIC2_COMBINED` (v0.9D capacity trade cache).
- Universe: top-30 dynamic rank symbols (v1.2s2 / v3.3 convention).
- Motif set: `S1`, `S3`, `S5` (S2 stays excluded — mistimed in v1.2s).
- Cooldown: 48 bars (12h) — v3.3 SA confirmed a 40–64 bar plateau.
- O6 policy: `O6_late9_slots4_cic1_050_cic2_025`.
- CP60: 4 bars (1h) pre-signal window, |close[i]/close[i-4]-1| ≤ 0.5% → "would exit".
- Protect_A cap2: at most 2 concurrent `gate_Protect_A=True` longs across baseline + overflow sleeves.
- Strict as-of: a long is only gated by confirmations with `feature_time <= signal_time`.

## Matrix shape

24 cells (6 actions × 4 baselines), all dumped into a single
`reports/v3_5_failure_risk_layer_bridge/failure_action_summary.csv` plus six
companion CSVs for stack/motif/CIC attribution, drawdown overlay, and
skipped-trade audit.

|       | B0 P2 max8 | B1 + O6 | B2 + CP60 + O6 | B3 + Protect_A cap2 + O6 |
|-------|------------|---------|----------------|--------------------------|
| F0 record_only          | (baseline) | (baseline) | (baseline) | (baseline) |
| F1 symbol no-long S1S3S5 | TBD | TBD | TBD | TBD |
| F2 symbol no-long S1 only | TBD | TBD | TBD | TBD |
| F3 no-overflow only      | n/a* | TBD | TBD | TBD |
| F4 disable Protect_A     | diag-only | diag-only | diag-only | diag-only |
| F5 CIC2-only no-long    | TBD | TBD | TBD | TBD |

`*` F3 at B0 is a no-op because B0 has no overflow sleeve.

## Pass criteria (8 points)

A cell graduates to `shadow` only if **all** of:

1. net20 does not drop vs the matching `F0` baseline.
2. drawdown proxy improves.
3. worst burst / worst month does not worsen.
4. skipped-trade avg net is negative or weaker than kept trades.
5. core profitable months are not materially trimmed.
6. validation / holdout direction matches.
7. signal does not depend on a handful of samples.
8. rule is simple and strictly as-of.

Cells meeting only (2) + (3) — drawdown wins without preserving net — graduate
to `risk_mode_option`, never to a default overlay. Cells that drop both net
and drawdown are `reject`.

## Findings (placeholder until A100 run)

The cells are stamped by the writer at run time based on the above criteria.
After the production run, the `failure_action_summary.csv` rows will be
backfilled into the table above and the headline narrative goes here.

Three hypotheses going in (testable, not declared):

- **H1 (instructment §F3)**: F3 (no-overflow only) at B1/B3 should beat F1
  globally because O6 is an additive risk sleeve — disabling overflow on
  failure-recent symbols is a smaller cut that preserves the P2 core entry.
- **H2 (instructment §F5)**: F5 (CIC2-only no-long) at B1/B2/B3 should beat F1
  because CIC2_beta_broad is the noisier branch, while CIC1_beta_extreme is
  the high-conviction one we don't want to gate.
- **H3 (instructment §F4)**: F4 is metric-only — the simulator can't replay
  the Protect_A protected-exit path — so F4 ships as a count-only diagnostic
  rather than a net20-mover. The signal to watch is the `protect_a_flagged`
  count vs the would-be loss share inside `failure_skipped_trade_attribution.csv`.

## v3.6 hand-off

Cells stamped `shadow` or `risk_mode_option` become the v3.6 live-counterfactual
targets:

- v3.6 takes the F1/F2/F3/F5 winners and feeds them into the
  `risk_off_signal_shadow.csv` writer on the *live* signal stream (still
  research tier — no execution-path wiring).
- v3.7 (re-investigate short) only runs if at least one F-cell is strictly
  better than the v3.3 broad symbol-no-long; otherwise the v3.3 finding
  stands and we move on to other long-side surface area.

## Discipline

- Strict as-of: a long is only gated by confirmations with `feature_time <= signal_time`.
- O6 simulator backfills freed slots first-come; the SKIP_OVERFLOW channel
  lets a row claim the baseline slot but never an overflow slot.
- Protect_A cap2 is enforced *across* the baseline + overflow sleeves (not
  per-sleeve), so the cap binds the concurrent protected-long count itself.
- No paper-live / real-live permission changes.

## Provenance

- Code: `src/pressure_graph/reports/v3_5_failure_risk_layer_bridge.py`
- CLI: `pressure-graph run-v3-5-failure-risk-layer-bridge`
- Tests: `tests/test_v3_5_failure_risk_layer_bridge.py` (12 cases, all green locally)
- Pipeline: `pipeline.run_v3_5_failure_risk_layer_bridge_from_features`

Run on the A100 box (per `[[a100-ssh-access]]`) once the v0.9D cache is fresh.

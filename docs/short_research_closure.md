# Short Research Closure — graph_cpt_quant short-v12s

**Status: closed. Do not reopen without satisfying §"Reopen criteria" below.**

This document terminates the multi-version short-side research thread on the
`short-v12s` branch. Five sleeves were investigated across v12s, v3.3, v3.4,
v4S, and v6S; the verdict per sleeve is locked here so a future contributor
does not re-fight the same questions. The only artefact of that work that
ships forward is the `F3 / F5` long-stack risk layer — a *long*-side action,
not a short.

---

## TL;DR

- **Standalone short alpha**: rejected (v3.4 SS1A..SS3B, all six, no_value).
- **Motif-led short** (S1/S3/S5 → reclaim → breakdown): rejected (v3.4 + v4S Path B).
- **CIC-failure short** (CIC long → no follow-through → CP60 → breakdown): rejected (v4S Path A).
- **Crowded-stall short** (crowded funding + OI + stall + BTC weak + Swing exit): demoted to `risk_off_only` discretionary action (v6S — month concentration 85%, no hedge value).
- **F3 / F5 long risk layer**: retained as the only failure-aware shadow candidates (v3.5).
- Future short research must satisfy the 5 reopen criteria below before any re-investigation.

---

## Per-sleeve verdict

### 1. Standalone short alpha — REJECTED

| where | sleeve set | verdict |
|-------|-----------|---------|
| v3.4 SS1A failed_reclaim_breakdown btc_not_up | motif | `no_value` |
| v3.4 SS1B failed_reclaim_breakdown low_coimpulse | motif | `no_value` |
| v3.4 SS2A crowded_long_stall btc_down | motif (S3-anchored) | `no_value` |
| v3.4 SS2B crowded_long_stall low_coimpulse | motif (S3-anchored) | `no_value` |
| v3.4 SS3A cic_failure_breakdown density_fading | CIC-led | `no_value` |
| v3.4 SS3B cic_failure_breakdown no_protect_a | CIC-led | `no_value` |

All six sleeves shipped `verdict=no_value` in `reports/v3_4_true_short_sleeve/
short_vs_no_long_comparison.csv`. Reference: commits `b4715e8` + `70f384b`
plus `docs/v3_4_true_short_sleeve_findings.md`. SA over the cooldown grid
{8,16,24,32,40,48,56,64,96} bars produced the same plateau verdict.

### 2. Motif-led short (Path B in v4S) — REJECTED

Even with N=1028 short observations across the full top-30 universe under
both Fast and Swing exit rules, normal_short produces:

- Fast: mean −0.48%, win 34.7%
- Swing: mean −0.31%, win 43.8%

Reference: `docs/v4s_failure_state_graph_findings.md` Path B section; the
v4S atlas at `reports/v4s_failure_state_graph/failure_state_action_atlas.csv`
(filter to `path = B_failed_reclaim_breakdown`).

### 3. CIC-failure short (Path A in v4S) — REJECTED

When the v3.4 SS3 conditions fire (failed follow-through + CP60 weak +
break below entry/pullback low), the underlying CIC long *still recovers*:

- `allow_long` N=31, mean +1.21%, win 67.7% → the original long is right.
- `normal_short` N=31, mean −1.89% (Fast) / −2.55% (Swing).
- `exit_existing_long` N=31, mean −0.07% — even *exiting* the long is
  worse than letting it run.

The existing CP60 / Protect_A / O6 management already does the right thing
for these CIC longs. Adding a short overlay strictly costs PnL.

Reference: `docs/v4s_failure_state_graph_findings.md` Path A section.

### 4. Crowded-stall short (Path C in v4S, validated in v6S) — DEMOTED

v4S identified `crowded stall + BTC weakness + Swing` as the one cell that
beat `no_long` (N=63 mean +1.52% win 61.9% at 20 bps focal). v6S applied
the seven-criterion discipline battery and the candidate landed at
**`risk_off_only`** (5 of 7 pass):

- ✓ min_samples (63)
- ✓ beats_no_long (+1.54% vs 0)
- ✓ cost_stress_robust (positive across 4 × 3 cost/slippage grid; +0.74% at 50bp+10bp slip)
- ✓ symbol_stability (max symbol contribution 13.7%, below 35%)
- ✓ clean_short_squeeze_ok (squeeze rate 15.87%, below 20%)
- ✗ **month_stability — best-month contribution 85.06%**, one month does the work
- ✗ **hedge_value — Pearson corr vs long-stack monthly net −0.167 (weak); in long stack's worst month, S-C1 also lost −5.71%**

Net interpretation: the alpha is a single-month regime event that does not
hedge the long book. `risk_off_only` means an operator may *manually* open
a small short when the combo gate fires under BTC weakness, but the
framework will never auto-trigger it. Reference:
`docs/v6s_path_c_short_validation_findings.md`.

### 5. F3 / F5 long risk layer (v3.5) — RETAINED

The only artefact of the failure-motif research that survives discipline is
the *long-side* risk overlay:

- **F5 CIC2-only no-long** (S1/S3/S5, 48-bar cooldown) — on the B3 managed
  stack (P2 max8 + O6 + Protect_A cap2): net20 +0.58%, max_dd −31%,
  ret/dd 4.35, 13 gated longs. The cleanest cut: smallest sample,
  gated_realized_net −0.35%, loss share 54%.
- **F3 no-overflow only** — ships alongside F5 as free upside on the
  overflow channel: net20 +0.34% with zero drawdown change.
- **F1 umbrella** kept as comparator only (over-gates CIC1_beta_extreme).
- F2/F4 retired (F2 fails criterion 7; F4 is metric-only and the
  `gate_Protect_A` column never fires on the top-30 universe).
- B2 CP60 prefilter retired entirely.

Reference: `docs/v3_5_failure_risk_layer_bridge_findings.md`. These ship
to the live counterfactual shadow stream — see the companion module
`failure_overlay_shadow` for the running ledger.

---

## Why this is closed

Every failure-side angle has now been tried, and the cumulative discipline
shows the same pattern:

1. **The long stack already manages most of the risk.** CIC longs recover
   even after failure motifs; the Protect_A / CP60 / O6 chain already
   tightens before the cliff. Short overlays compete with that protection,
   not add to it.
2. **Standalone short alpha lacks distribution.** Every candidate either
   fires too rarely (Path C) or in regimes that already coincide with long
   drawdown (Path B); concentration or hedge failure kills the deployment
   case.
3. **The instructment5 framing is correct.** Failure information is
   *long-stack risk context*, not a short signal. The two pieces that
   survived (F3, F5) act on the long book, not against it.

Future short experiments should not iterate on TP/SL bands, inverted
gates, or alternative motif combinations — those have been exhausted.

---

## Reopen criteria

To reopen short research, a new candidate must satisfy these FIVE
conditions *before* the first production run:

1. **≥3-month-distributed sample** — no single month contributes ≥ 35% of
   the alpha. v6S' Path C carried 85% in one month; that pattern is
   forbidden.
2. **Hedge sign correct** — Pearson correlation vs long-stack monthly
   net ≤ −0.3, OR positive PnL in the long stack's worst month. v6S
   Path C failed both legs.
3. **Cost survives 30 bp + 5 bp slip** — not just focal 20 bps. The
   v6S cost-stress grid is the minimum bar.
4. **Strictly beats `no_long`** — not just beats 0. The B counterfactual
   must show short > no_long > 0; if `no_long` already pays for itself
   the short is redundant.
5. **Squeeze margin ≥ 20%** — clean-short label
   `short_squeeze_before_hit` ≤ 20% across the candidate sample. v6S
   Path C is at 15.87%, so the bar is achievable; any candidate above
   25% is rejected on principle.

Failure to satisfy any of the five = no run. Do not "explore" first.

---

## Provenance

- v3.4 sleeves + verdicts: `docs/v3_4_true_short_sleeve_findings.md`,
  commits `b4715e8` (feat) + `70f384b` (docs).
- v3.5 F-cells: `docs/v3_5_failure_risk_layer_bridge_findings.md`,
  commits `7206c06` + `e1920f4`.
- v4S 3 paths × 7 actions: `docs/v4s_failure_state_graph_findings.md`,
  commit `f8e1cbb`.
- v6S Path C discipline: `docs/v6s_path_c_short_validation_findings.md`,
  commit `f33c50f`.
- A100 production environment: see [[a100-ssh-access]] memory note.
- Live F3/F5 shadow recorder: `src/pressure_graph/reports/failure_overlay_shadow.py`
  and the companion ledger in `reports/failure_overlay_shadow/`.

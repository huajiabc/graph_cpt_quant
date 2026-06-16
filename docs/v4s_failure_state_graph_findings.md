# v4S Failure State Graph — findings

**Status: research only.** A100 production run completed against the
v0.9D capacity trade cache (top-30 universe, conda quant py3.11).
Reports under `reports/v4s_failure_state_graph/` (gitignored per
v3.3/v3.4/v3.5 convention); 2,244 failure-state observations × 7
actions = 15,708 atlas rows. The instructment5 question — *“failure
after, best action?”* — has a clean answer per path.

## Setup

- Pool: `P2_CIC1_CIC2_COMBINED` (v0.9D capacity trade cache).
- Universe: top-30 dynamic-rank symbols.
- Paths: A CIC failure (CIC long → no follow-through → CP60-weak → breakdown), B failed reclaim breakdown (S1/S3/S5 motif → reclaim → fail → breakdown), C crowded long stall (funding≥70 pct ∧ OI≥60 pct ∧ ret_4h ≤55 pct ∧ failed follow-through ∧ (BTC_down ∨ low coimpulse)).
- Actions: `allow_long, no_long, disable_overflow, disable_protect, exit_existing_long, small_short, normal_short`.
- Cost: 20 bps focal (round-trip = 40 bps). Short sizing: small=0.5×, normal=1.0×.
- Short execution: Fast (TP +3%, SL −2%, max 16 bars) and Swing (TP +5%, SL −3%, max 48 bars).
- Long-match lookback: 12h cooldown against the v0.9D long index.

## Headline

| path | best action | sample | mean | win | verdict |
|------|------------|--------|------|-----|---------|
| **A_cic_failure** | `allow_long` | N=31 | +1.21% | 67.7% | **shadow** — long is right, don't override |
| **A_cic_failure** | `small_short`, `normal_short` | N=31 | −0.95% / −1.89% | 9.7% | **reject** — CIC longs recover |
| **A_cic_failure** | `exit_existing_long` | N=31 | −0.07% | 0% | **reject** — exit books a small loss vs holding to the +1.21% mean |
| **B_failed_reclaim_breakdown** | `allow_long` | N=20 | −1.94% | 15% | reject (small N) — no CIC long aligned with these motif observations |
| **B_failed_reclaim_breakdown** | `small_short`, `normal_short` (fast) | N=1028 | −0.24% / −0.48% | 34.7% | **reject** — even on the largest path-B sample short loses 0.5% net at 20 bps |
| **B_failed_reclaim_breakdown** | `small_short`, `normal_short` (swing) | N=1028 | −0.16% / −0.31% | 43.8% | reject — swing improves win rate but still negative |
| **C_crowded_long_stall (fast)** | `small_short`, `normal_short` | N=63 | −0.13% / −0.26% | 39.7% | reject — Fast TP/SL clips winners on this slow regime |
| **C_crowded_long_stall (swing)** | `normal_short` | N=63 | **+1.52%** | **61.9%** | **shadow** ✓ |
| **C_crowded_long_stall (swing)** | `small_short` | N=63 | +0.76% | 61.9% | **shadow** ✓ |

The narrative is clean:

1. **Don't fight CIC longs** (path A). When the v3.4 "CIC failure breakdown"
   conditions fire (failed follow-through + CP60 would exit + price breaks
   below entry/pullback low), the long *still* recovers on average. 31 of
   them are real CIC longs from the v0.9D cache; 21 ended in net profit
   despite the breakdown signal. The right action is `allow_long`. Anything
   else — exit, disable_overflow, short — costs PnL on this path.

2. **Motif-led breakdown is not a short** (path B). With N=1028 short
   observations (vs N=20 with a matched long, because S1/S3/S5 motifs
   rarely coincide with a CIC long in the 12h window), the short net is
   −0.24% to −0.48% at 20 bps cost. Swing exit improves win rate (35 → 44%)
   but still loses net. This is the same `reject` verdict v3.4 SS1A/B
   already produced, now restated under v4S's seven-action framework.

3. **Crowded long stall + Swing exit is a real short edge** (path C). The
   combo gate (funding crowded + OI crowded + price stalling + failed
   follow-through + BTC weakness) fires 63 times across the universe.
   Fast rule clips winners (−0.13% net); Swing rule (48-bar max hold)
   captures the slow unwind — +1.52% net at 20 bps cost on `normal_short`,
   61.9% win rate. `small_short` at half size gives +0.76% with the same
   win rate. This is the *only* path × action × execution cell that
   meets the instructment5 §结果 test (short > no_long).

## v6S hand-off (next round)

Shadow the Path C swing-short on the live stream — research only,
no live wiring yet. Validate the +1.52% / 61.9% finding against the
discipline checks the v4S MVP deferred:

1. **Cost-grid sensitivity** — re-run at 30 and 50 bps. If +1.52% at
   20 bps decays to ≈0 at 50 bps, the edge is execution-dependent and
   not robust enough to ship.
2. **Month-cap** — cap monthly contribution at 35%. If +1.52% comes
   from one quarter, kill it.
3. **Leave-one-month / leave-one-symbol** — does any single month or
   symbol carry the average? At N=63 this matters a lot.
4. **Random / shuffled control** — randomise feature times by symbol;
   verify the +1.52% drops to ~0% on random gates.
5. **Forward sample** — at least one out-of-sample month before any
   paper-live / shadow-live promotion.

If all five hold, Path C swing-short ships into v6S as an *additive
risk-off action* (open small_short on crowded stall + BTC_down), NOT
as a standalone short sleeve.

## What v4S settles for good

- **Standalone short alpha** on S1/S3/S5 motif breakdowns: dead. v3.4 said
  this; v4S restates it under a seven-action lens. No more SS1A/SS1B
  tuning.
- **CIC failure breakdown as a short trigger**: dead. Even when the v3.4
  SS3A/SS3B conditions fire, the underlying CIC long still recovers
  enough that any non-`allow_long` action costs PnL. `exit_existing_long`
  is the closest counterfactual and it still loses.
- **The instructment5 question** — *failure 后最佳动作* — has a clean
  per-path answer:
  - Path A: `allow_long` (keep the CIC long; the system already manages
    the exit through Protect_A / CP60 / O6).
  - Path B: `no_long` (don't override; failure motifs don't tell us
    anything actionable for the long stack here).
  - Path C: `normal_short` with Swing rule (the one cell that wins
    cleanly).

## Discipline

- Strict as-of: every predicate reads features at idx ≤ signal_time.
- Counterfactual against `no_long` mean 0%: each action's "vs_no_long"
  is in `failure_vs_no_long.csv`.
- Short executions use the standard Fast/Swing rules from
  `pressure_graph.backtest.short_execution.simulate_short_exit`.
- No paper-live / real-live wiring. Tier: research only.
- Cost-grid / month-cap / leave-one-month / random / forward-sample
  checks deferred to v6S follow-up (see hand-off rules above).

## Provenance

- Code: `src/pressure_graph/reports/v4s_failure_state_graph.py`
- CLI: `pressure-graph run-v4s-failure-state-graph --config configs/v0_3.yaml`
- Pipeline: `pipeline.run_v4s_failure_state_graph_from_features`
- Tests: `tests/test_v4s_failure_state_graph.py` (11 cases, all green
  locally + on the A100 conda `quant` env).
- Reports: `reports/v4s_failure_state_graph/` — 6 CSVs (~4.6 MB atlas,
  430 KB vs_no_long pivot, others small) + `candidate_notes.md`.
- Related: [[v3-5-failure-risk-layer-bridge]] (where the F5+F3 hand-off
  came from), [[v3-3-v3-4-failure-search-and-true-short]] (v3.4's
  standalone-short-is-dead verdict that v4S restated).
- A100 access: [[a100-ssh-access]].

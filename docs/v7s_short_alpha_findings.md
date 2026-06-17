# v7S Short Alpha Exploration — findings

**Status: Direction E first-run complete. Verdict `no_value` on both
candidate codes — empirically confirms the closure doc's hypothesis
that CIC failure does NOT produce a tradable short even under the
strict 4-gate cascade specified by `short_instructment6 (v7s).docx`.**

> Lane opened per the docx mandate to explore short alpha orthogonal
> to the closed v12s / v3.4 / v4S / v6S motif thread. The closure doc
> (`docs/short_research_closure.md`) prohibited iterating on failed
> motifs and locked five reopen criteria; v7S satisfies the "orthogonal
> new lane" path. Direction E is the only direction with code wired in
> this commit.

## Five directions and scope

| Direction | Question | Data needed | Status |
|-----------|----------|-------------|--------|
| **A** Cross-exchange lag | Binance/OKX sell impulse → Bybit lag → short | Binance UM + Bybit linear aggTrades | Stubbed |
| **B** Liquidation continuation | Long-liquidation spike → failed reclaim → continuation | Liquidation tape | Deferred (no source) |
| **C** Crowded unwind v2 | funding+OI high + taker-buy exhaustion + CVD divergence | aggTrades CVD | Stubbed |
| **D** Relative-value pair | overextended beta vs leader → mean revert | Multi-symbol features | Stubbed |
| **E** CIC-failure confirmed (strict) | v4S Path A + beta_high gone + sell flow confirms | Local CIC + v11 orderflow_history | **Run complete — `no_value`** |

A/B/C/D directions remain `NotImplementedError`-stubbed in
`cfg.enabled_directions`; follow-up commits will wire each.

## Direction E run — gate cascade and verdict

### Universe and inputs

- Feature parquet: `data/processed/v0_3/perp_pressure_features_all_eligible.parquet` (80 symbols, 12.4M bars).
- CIC long index: 191 entries (152 within v7S top-30 universe).
- Orderflow event parquet: `data/orderflow_history/binance_um/cic_event_orderflow.parquet` (loaded ok).

### Gate cascade (per candidate)

| Stage | E1 break-entry | E2 break-pullback |
|---|---|---|
| 1. CIC longs in universe | 152 | 152 |
| 2. Breakdown found within 12 bars | 144 | 17 |
| 3. + CP60 weak follow-through | 32 | 1 |
| 4. + Protect_A not active (col missing → fail-open) | 32 | 1 |
| 5. + beta_high environment gone (btc_vol_regime / btc_market_state) | 11 | 1 |
| 6. + sell flow confirms (orderflow at break bar) | **0 (data gap)** | **0 (data gap)** |

The sell-flow gate matched zero events because the v11
`cic_event_orderflow.parquet` was backfilled around CIC LONG entry
timestamps, not breakdown timestamps. Every Direction E breakdown
returned `orderflow_unmatched_event`. To get a meaningful sell-flow
read for Direction E, the orderflow_history backfill must be re-run
with break-bar event windows.

### Fail-open ablation (sell-flow gate waived)

To produce numbers under the 5-gate strict chain (without the data-gap
sell-flow gate), the lane was re-run with `--sell-flow-fail-open`.
Eleven E1 events + one E2 event pass the 5-gate cascade. Numbers
(focal cost 20 bps, funding 30 % APR):

| Candidate | Execution | N | mean_gross | mean_net20 | win_rate | verdict | gate failures |
|-----------|-----------|---|-----------|-----------|---------|---------|---------------|
| E1_cic_break_entry_strict | fast | 11 | -1.35 % | -1.74 % | 9.1 % | `no_value` | 1,2,3,5,6,9 |
| E1_cic_break_entry_strict | swing | 11 | -2.53 % | -2.92 % | 9.1 % | `no_value` | 1,2,3,5,6,8,9,10 |
| E2_cic_break_pullback_strict | fast | 1 | +2.08 % | +1.70 % | 100 % | `no_value` | 6,7,10 (N=1) |
| E2_cic_break_pullback_strict | swing | 1 | +2.53 % | +2.17 % | 100 % | `no_value` | 6,7,10 (N=1) |

**E1 result.** 11 strict CIC-failure shorts produced -1.74 % mean net20
on Fast and -2.92 % on Swing, win rate 9.1 %. The CIC long recovered
in 10 of 11 events; `short_beats_no_long_pct = 9.1 %`. Failing gates:

- gate1: mean_net20 ≤ 0
- gate2: net30 < 0.5 × net20 (also negative)
- gate3: hit_down_3pct < 0.35 (shorts rarely hit a 3 % drop)
- gate5/6: monthly net negative under capping and leave-worst
- gate9: short does NOT beat no_long > 0 (closure §reopen criterion 4)
- gate10 (swing only): hedge corr did not pass the −0.30 bar

**E2 result.** Only 1 event passes — N too small to interpret. Even
the single trade fails gate6 / gate7 / gate10 mechanically.

### Read

The closure doc's TL;DR §3 said:

> CIC-failure short (CIC long → no follow-through → CP60 → breakdown):
> rejected (v4S Path A).

v7S Direction E asked: *if we add `beta_high_gone` + `sell_flow_confirms`
on top of v4S Path A's chain, does the answer change?* Empirically:

1. `beta_high_gone` (btc_vol_regime / btc_market_state proxy) prunes
   the population by ~3× (32 → 11 for E1; 1 → 1 for E2).
2. Despite the additional gate, the 11 surviving events show CIC longs
   recovering in 10/11 (90.9 %). The short still loses 1.7-2.9 % mean.

The v3.4 SS3 rejection ledger continues: CIC longs have a
self-recovery property that the existing CP60 / Protect_A / O6
management already exploits. Adding a short overlay competes with
that protection and loses.

The hypothesis the docx wanted to test — "wait for failure THEN
confirmation of inability to recover" — does not survive contact with
the data on this universe and window.

## Ten-gate acceptance reference

| # | Gate | Pass condition |
|---|------|----------------|
| 1 | net20+slip > 0 | `mean_net20 > 0` |
| 2 | net30 holds | `mean_net30 > 0` AND `mean_net30 ≥ 0.5 × mean_net20` |
| 3 | clean_short_hit lifts | `hit_down_3pct ≥ 0.35` |
| 4 | squeeze controllable | `short_squeeze_before_hit ≤ 0.20` |
| 5 | month_cap35 still positive | `month_capped_net > 0` |
| 6 | leave-one-month not collapsing | `leave_worst_net > 0` |
| 7 | max_symbol_share < 35 % | `max_symbol_share ≤ 0.35` |
| 8 | matched random strictly worse | `candidate_mean > random_mean` |
| 9 | short > no_long > 0 | `mean_C_short > mean_B_no_long > 0` |
| 10 | hedge complementary | `hedge_corr ≤ -0.30` OR `short_in_long_worst_month > 0` |

Verdict logic:

- `promote` — all 10 gates pass.
- `risk_off_only` — gates 1-8 pass; exactly one of 9/10 fails.
- `no_value` — any of gates 1-8 fail (or both 9 and 10 fail).

## Reproduction

A100 (production):

```bash
ssh -L 12222:10.106.200.247:2222 root@10.115.7.6 -p 25711
ssh root@localhost -p 12222            # second terminal
cd /opt/data/private/Wangjb/graph
source /opt/conda/etc/profile.d/conda.sh && conda activate quant
PYTHONPATH=src python scripts/v7s_short_alpha_once.py --config configs/v0_3.yaml
# ablation:
PYTHONPATH=src python scripts/v7s_short_alpha_once.py --config configs/v0_3.yaml \
  --sell-flow-fail-open --report-root reports/v7s_short_alpha_fail_open
# gate cascade audit:
PYTHONPATH=src python scripts/v7s_direction_e_gate_cascade.py
```

Local (without features): writes empty CSV stubs and
`candidate_notes.md = "No data"`. Useful only for harness sanity:

```bash
PYTHONPATH=src python -m pytest tests/test_v7s_short_alpha.py -q
```

## Outputs

Production outputs are committed under
`reports/v7s_short_alpha/E_cic_failure_confirmed/` (strict, N=0) and
`reports/v7s_short_alpha/E_cic_failure_confirmed_failopen/` (sell-flow
waived, N=12). Layout matches the ten docx-mandated files plus the
`gate_cascade_counts.csv` audit dropped by the diagnostic script.

## Notes for the next commit

1. **Re-backfill orderflow for breakdown events.** The Direction E
   strict run is *data-blocked*, not signal-blocked. If a follow-up
   commit re-runs `pressure_graph.orderflow_history` over break-bar
   event windows instead of CIC entry windows, the sell-flow gate
   becomes meaningful and the strict run can be re-evaluated.
2. **Direction D priority.** Relative-value pair short uses only
   multi-symbol features the long stack already builds — no new data
   plumbing needed. Cleanest next target.
3. **Direction A** is the next-highest priority and is data-ready on
   A100 (per memory note `v11-orderflow-burst-ranking`) — but it needs
   Bybit aggTrades alignment work to compare with Binance lead.
4. **Beta-high gate semantics** are now market-regime proxied via
   `btc_vol_regime` transitions. If a future feature build adds
   `gate_beta_already_extended` to the v0.3 parquet, the gate
   automatically falls back to the strict v07c semantic.

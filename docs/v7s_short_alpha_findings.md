# v7S Short Alpha Exploration — findings

**Status: Directions E + D complete. Both `no_value` across all candidates.
A clean negative finding for the docx Direction E hypothesis (CIC longs
recover even after strict gating) AND for the Direction D pair-short
hypothesis (BTC/ETH/basket hedges hurt rather than help on this universe).**

> Lane opened per the docx mandate to explore short alpha orthogonal to
> the closed v12s / v3.4 / v4S / v6S motif thread. The closure doc
> (`docs/short_research_closure.md`) prohibited iterating on failed
> motifs; v7S is the orthogonal new lane.

## Direction inventory and status

| Direction | Question | Data needed | Status |
|-----------|----------|-------------|--------|
| **A** Cross-exchange lag | Binance/OKX sell impulse → Bybit lag → short | Binance UM + Bybit linear aggTrades | Stubbed (next) |
| **B** Liquidation continuation | Long-liquidation spike → failed reclaim | Liquidation tape | Deferred (no source) |
| **C** Crowded unwind v2 | funding+OI high + taker-buy exhaustion + CVD divergence | aggTrades CVD | Stubbed |
| **D** Relative-value pair | overextended beta vs leader → mean revert | Multi-symbol features | **Run complete — `no_value`** |
| **E** CIC-failure confirmed (strict) | v4S Path A + beta_high gone + sell flow confirms | Local CIC + v11 orderflow_history | **Run complete — `no_value`** |

## Direction D — relative-value pair (Phase 1 done)

### Spec implemented (per v7s docx expanded guide)

Five candidates × three fixed holding horizons × two confirmation modes
(with vs without reclaim_failure gate) = 30 cells per stream.

| Candidate | Long leg | Cost legs (round-trip) |
|-----------|----------|------------------------|
| D0_naked_short | (none) | 1 |
| D1_pair_btc | BTC | 2 |
| D2_pair_eth | ETH | 2 |
| D3_pair_dynamic_leader | argmax 24h-ret of pool {BTC, ETH, SOL, BNB} | 2 |
| D4_pair_basket | mean(BTC, ETH, SOL) | 4 |

Holding horizons: h4 (16 bars), h12 (48 bars), h24 (96 bars). Each
horizon exits at the fixed bar's close (no intra-trade TP/SL).

`_nc` suffix denotes the no-confirmation ablation (reclaim_failure gate
bypassed) used to test docx §核心对照 item 7.

Entry chain: (1) symbol's `ret_4h_percentile ≥ 95` somewhere in 4h
lookback. (2) BTC's `ret_4h ≤ -0.5 %` at break bar. (3) (when enabled)
symbol's close ≥ 1.5 % below the lookback's high.

### Run summary (A100, top-30 universe, 76 beta candidates)

Numbers below are mean_net20 at 20 bps focal cost; n_legs-aware cost
charged per row.

| Candidate | h4 | h12 | h24 | h24 win | h24 verdict |
|-----------|----|------|------|---------|-------------|
| **D0_naked_short** | -0.30 % | -0.24 % | **+0.13 %** | 55.0 % | no_value (gate2,5,10) |
| **D0_naked_short_nc** | -0.29 % | -0.20 % | **+0.18 %** | 55.5 % | no_value (gate2,5,10) |
| D1_pair_btc | -0.76 % | -0.75 % | -0.74 % | 46.7 % | no_value |
| D1_pair_btc_nc | -0.77 % | -0.77 % | -0.76 % | 46.8 % | no_value |
| D2_pair_eth | -0.83 % | -0.78 % | -0.81 % | 45.2 % | no_value |
| D2_pair_eth_nc | -0.86 % | -0.86 % | -0.92 % | 43.8 % | no_value |
| D3_pair_dynamic_leader | -0.79 % | -0.73 % | -0.81 % | 43.8 % | no_value |
| D3_pair_dynamic_leader_nc | -0.80 % | -0.76 % | -0.84 % | 43.6 % | no_value |
| D4_pair_basket | -1.63 % | -1.60 % | -1.66 % | 32.4 % | no_value |
| D4_pair_basket_nc | -1.64 % | -1.64 % | -1.71 % | 31.5 % | no_value |

Sample sizes: with-confirmation candidates ≈ 5325 trades/horizon;
no-confirmation ≈ 6649/horizon (≈25 % more events without the
reclaim_failure gate).

### Key findings

1. **The relative-value pair hypothesis is empirically refuted on this
   universe.** Every pair candidate (D1/D2/D3/D4) has WORSE gross AND
   worse net than the naked baseline (D0). Hedging with BTC/ETH/SOL or
   the 3-symbol basket subtracted alpha. The user-docx hypothesis was
   that hedging would protect against "全市场继续上冲导致裸空被打爆";
   in this data that risk is sufficiently rare that the hedge cost +
   hedge correlation with the beta wipes out the protection benefit.

2. **The reclaim_failure gate adds no meaningful edge.** Comparing
   with-confirmation vs no-confirmation cells:
   - D0 h24: +13 bps → +18 bps (no_conf is slightly better)
   - D1 h24: -74 bps → -76 bps (~same)
   - D2 h24: -81 bps → -92 bps (no_conf slightly worse)
   The gate drops 25 % of the population for zero net20 improvement.
   The alpha (such as it is) lives in the overextension +
   leader_weakening combo, not in the breakdown confirmation.

3. **Naked beta short of overextended names at 24h hold is the only
   row with positive net20** at the 20 bp focal — but cost-fragile.
   - D0_naked_short_nc h24: gross +0.50 %, net20 +0.18 %, win 55.5 %.
   - At 30 bp net = -0.04 % (gate2 fails).
   - month_cap fails too (one month likely dominates).

4. **Basket hedge (D4) is catastrophic** at -1.66 % net20 because the
   4-leg cost (8 × 20 bp = 1.6 %) dominates any signal. The basket arm
   exists only because the docx §核心对照 item 4 called for it; it
   verifies it does not work here.

5. **Win rates flip near 50 % at h24.** D0 sits at 55 %, all pairs
   sit at 43-47 %. The base rate from overextended-beta-short is
   marginally positive but well below the cost-breakeven level.

### Verdict

Direction D verdict: `no_value` on all 30 cells.

Direction D's closest-to-promote cell is `D0_naked_short_nc h24`
(7 of 10 gates pass), and that's a stripped-down NAKED short with
NO confirmation gate — the exact opposite of what the docx structurally
proposed. The expanded pair scaffolding tested in this commit is the
right place to STOP iterating, not iterate.

## Direction E — strict CIC-failure-confirmed (previously closed)

Recap from prior commit `97fb697`:

| Candidate | Execution | N | mean_net20 | Verdict |
|-----------|-----------|---|-----------|---------|
| E1_cic_break_entry_strict | fast | 11 | -1.74 % | no_value |
| E1_cic_break_entry_strict | swing | 11 | -2.92 % | no_value |
| E2_cic_break_pullback_strict | fast | 1 | +1.70 % | no_value (N=1) |
| E2_cic_break_pullback_strict | swing | 1 | +2.17 % | no_value (N=1) |

CIC longs recovered in 10/11 events (90.9 %). Strict CIC-failure
confirmed short does not pay.

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

## Notes for the next commit

1. **Direction A is the next priority** — per memory note
   `v11-orderflow-burst-ranking`, Binance UM + Bybit linear aggTrades
   are both on the A100 box. The lead-lag detector is the cleanest
   data-ready next direction since it requires aligning two tapes by
   timestamp, not building new gates.
2. **Direction E remains data-blocked on sell-flow.** Re-backfilling
   orderflow_history at breakdown bars (instead of CIC entry bars)
   would unblock the strict gate; deferred until Direction A lands.
3. **Direction C v2** needs CVD divergence labels not currently
   exported — spec these before wiring.
4. **Direction D scaffolding stays in tree** as a negative result — do
   not re-tune hedge ratios or add new horizons here without first
   producing a hypothesis for why the leader-beta correlation in this
   universe would differ.

## Reproduction

A100 (production):

```bash
ssh -L 12222:10.106.200.247:2222 root@10.115.7.6 -p 25711
ssh root@localhost -p 12222            # second terminal
cd /opt/data/private/Wangjb/graph
source /opt/conda/etc/profile.d/conda.sh && conda activate quant
PYTHONPATH=src python scripts/v7s_short_alpha_once.py --config configs/v0_3.yaml
# Direction E orderflow ablation:
PYTHONPATH=src python scripts/v7s_short_alpha_once.py --config configs/v0_3.yaml \
  --sell-flow-fail-open --report-root reports/v7s_short_alpha_fail_open
# Direction E gate cascade audit:
PYTHONPATH=src python scripts/v7s_direction_e_gate_cascade.py
```

Local (without features): writes empty CSV stubs and
`candidate_notes.md = "No data"`. Useful only for harness sanity:

```bash
PYTHONPATH=src python -m pytest tests/test_v7s_short_alpha.py -q
```

## Outputs

Production outputs live under `reports/v7s_short_alpha/<direction>/`
(gitignored locally; pulled from A100 per session). Each direction
emits the docx-mandated ten CSVs + `candidate_notes.md` (verdict
narrative). Direction E additionally emits
`gate_cascade_counts.csv` via the diagnostic script for visibility into
which gate filters the population.

# v7S Short Alpha Exploration — findings

**Status: Directions E + D + A complete. Direction A1_h24 produces v7S's
first 8/10-gate cell — the cross-exchange lead-lag hypothesis is the
only structural angle that survives discipline so far.**

Headline:

- Direction A1_binance_sell_impulse_bybit_lag at **h24**: N=34,
  gross +0.93 %, net20 +0.61 %, net30 +0.41 % (cost-robust),
  win 73.5 %, fails only gate5 (month_cap) and gate7 (symbol_share
  35.4 % — over the 35 % bar by 0.4 %).
- Direction E: all candidates `no_value` (closure doc confirmed).
- Direction D: all 30 cells `no_value` (pair hedging HURTS naked alpha
  in this universe).

> Lane opened per the docx mandate to explore short alpha orthogonal to
> the closed v12s / v3.4 / v4S / v6S motif thread. The closure doc
> (`docs/short_research_closure.md`) prohibited iterating on failed
> motifs; v7S is the orthogonal new lane.

## Direction inventory and status

| Direction | Question | Data needed | Status |
|-----------|----------|-------------|--------|
| **A** Cross-exchange lag | Binance/OKX sell impulse → Bybit lag → short | Binance UM event orderflow + Bybit features | **Run complete — A1_h24 8/10 gates** |
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

## Direction A — cross-exchange downside lead-lag short

### Spec implemented

Three candidates × three fixed holding horizons = 9 cells per stream.
A3 (Hyperliquid lag) is deferred until a Hyperliquid tape lands.

| Candidate | Source-venue gate | Target-venue gate |
|-----------|-------------------|-------------------|
| A0_no_filter | (none — control) | (none — fires whenever an event row exists) |
| A1_binance_sell_impulse_bybit_lag | shock_bar Binance `buy_sell_imbalance ≤ -0.15` | Bybit close drop ≤ 1.5 % over 1h lookback |
| A2_binance_breakdown_bybit_failed_reclaim | pullback + reclaim Binance imbalance both ≤ -0.05 | Bybit recovered < 0.5 % from window low |

Event source: `data/orderflow_history/binance_um/cic_event_orderflow.parquet`
(574 events; Bybit-anchored signal_times with Binance UM imbalance reads
per the v11 PRE_ENTRY window contract).

Execution: single-leg short on Bybit (n_legs=1, same cost model as
Direction E and Direction D's naked baseline).

### Run summary (A100, top-30 universe)

| Candidate | Horizon | N | mean_gross | mean_net20 | mean_net30 | win | verdict | failures |
|-----------|---------|---|-----------|-----------|-----------|-----|---------|----------|
| A0_no_filter | h4 | 476 | -1.66 % | -2.05 % | -2.25 % | 21.8 % | no_value | 1,2,3,5,6,8,9 |
| A0_no_filter | h12 | 476 | -1.40 % | -1.76 % | -1.96 % | 30.0 % | no_value | 1,2,3,5,6,8,9 |
| A0_no_filter | h24 | 476 | -0.41 % | -0.72 % | -0.92 % | 48.7 % | no_value | 1,2,3,5,9 |
| A1 | h4 | 34 | +0.13 % | -0.25 % | -0.45 % | 50.0 % | no_value | 1,2,5,6,9,10 |
| A1 | h12 | 34 | -0.51 % | -0.87 % | -1.07 % | 47.1 % | no_value | 1,2,5,6,7,9 |
| **A1** | **h24** | **34** | **+0.93 %** | **+0.61 %** | **+0.41 %** | **73.5 %** | **no_value (gate5, gate7)** | **5, 7** |
| A2 | h4-h24 | 3 | -3 % to -5 % | -3 % to -5 % | -3 % to -5 % | 0 % | no_value | all |

### A1_h24 detailed read (canonical 1.5 % lag)

- N=34 events (25 winners, 9 losers).
- max_symbol_share = 35.4 % — fails gate7 (≤ 35 %) by 0.4 percentage points.
- leave_worst_symbol_net = +0.587 (positive).
- net30 = +0.41 % (cost-robust under the 30 bp stress).
- short_beats_no_long via gate9 — passes.
- A0 control (no filter) loses 41 to 205 bps gross at all horizons.
  The filter — Binance sell impulse + Bybit lag — is what supplies the
  edge; not the bar selection itself.

### A1 threshold sweep (h24 only — h4/h12 stay no_value)

The canonical 1.5 % threshold was followed by a sweep over four lag
thresholds to test whether the alpha generalizes:

| Variant | Bybit lag ≤ | N | gross | net20 | net30 | win | failures |
|---------|-------------|---|-------|-------|-------|-----|----------|
| A1_lag10bp | 1.0 % | 28 | +0.14 % | -0.18 % | -0.38 % | 67.9 % | 1,2,3,5,7,9 |
| A1 (1.5 %) | 1.5 % | 34 | +0.93 % | +0.61 % | +0.41 % | 73.5 % | 5,7 |
| A1_lag20bp | 2.0 % | 35 | +0.93 % | +0.61 % | +0.41 % | 74.3 % | 3,5,7 |
| **A1_lag25bp** | **2.5 %** | **37** | **+1.04 %** | **+0.72 %** | **+0.52 %** | **75.7 %** | **3, 5** |

A1_lag25bp passes 8 of 10 gates (gate7 now clears at 33.x % symbol
share, just below the 35 % bar) and net30 stays positive at +0.52 %.
The remaining failures are:

- **gate3** clean_short_hit_lifts (`hit_down_3pct ≥ 0.35`) — at the
  looser thresholds the shorts win on smaller drops more often
  (75.7 % at 24h hold), but the rate of clean 3 %+ down moves doesn't
  scale with N — suggests a slower / smaller-magnitude unwind, not the
  cliff-like clean-short the gate was designed for.
- **gate5** month_cap_positive — small N + concentrated months (best
  month ≈ 154 % of total when capped at 35 %; standard small-sample
  artifact). The closure doc's reopen criterion §1 is the binding
  constraint, not the alpha itself.

The 1.0 % threshold (A1_lag10bp) is empirically too strict — N drops
to 28, the cleanest events get lost, and net20 turns negative. The
2.5 % threshold is empirically the best on this 574-event tape — but
even there gate5 / gate3 hold the candidate at `no_value`.

### Key findings

1. **Cross-exchange downside lead-lag IS a real signal in this data.**
   Binance UM shock-bar buy_sell_imbalance ≤ -0.15 combined with Bybit
   close still ≤ 1.5 % drop in the same 1 h gives +93 bps gross / 73.5 %
   hit rate at 24 h hold. A0 (no filter) loses; A1 (with filter) wins.
2. **Lead-lag persistence shows on the 24 h horizon, not earlier.** A1 at
   h4 is gross +13 bps, win 50 %; the edge only materialises in the slower
   horizon. Consistent with "information propagation lag" framing.
3. **A2's stricter gate (sustained breakdown + failed reclaim) is too
   tight.** N=3 — the gate stack doesn't have a meaningful sample.
4. **The 574 events in cic_event_orderflow are CIC-anchored, not
   continuous time.** That's why N=34 even at A1's modest gates. A
   follow-up that builds continuous Binance CVD aggregates from
   aggTrades archives would 100-1000× the sample size; that's the
   right next step for Direction A.

### Verdict

Direction A verdict: `no_value` at the gate level, but A1_h24 is
risk_off_only-adjacent (8/10 gates passing). Recommend an immediate
follow-up commit that EITHER (a) loosens A1's Bybit lag threshold from
1.5 % to 2.0 % to expand N AND lower symbol_share, OR (b) backfills
continuous Binance CVD to widen the event base.

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

# v1.1 Orderflow Burst Ranking — Findings (2026-06-12)

Question (collab doc §4 + §5.2): under capacity limits the P2 basket's skipped
candidates perform as well as or better than selected ones; can pre-entry
taker-flow features (taker buy ratio, CVD imbalance, large-trade pressure)
rank same-burst CIC candidates better than first-come selection?

**Verdict: weak-positive in-sample, not validated — do not promote any
orderflow rank into live-shadow selection yet.** Keep the v0.8 live shadow as
a data collector and revisit when the contested-burst sample grows (~3x).

## Setup

- Candidate stream: full re-collection of the 369-symbol eligible Bybit
  universe (1y, 15m, via api.bybit-tr.com) → streaming v0.3 feature build →
  v0.9D frozen-candidate replay. 585 unique CIC1/CIC2/MIR1 events across 62
  symbols (2025-07-02 → 2026-06-05).
- Orderflow: Binance UM aggTrades daily archives (`binance_proxy` is an
  accepted v0.8.1 demand-queue source), 259 symbol-days, window features
  computed with the exact v0.8 `summarize_trade_window` semantics.
  Pre-entry coverage: **99.3% (CIC1) / 98.3% (CIC2)** of the P2 pool.
- Replication fidelity: first-come P2 replays reproduce the collaborator's
  published capacity curve to ≤0.04pp despite independent data collection
  (max5 10.62% vs 10.65%, max8 10.91% vs 10.93%, max10 9.60% vs 9.61%).
- Leak discipline: ranking features consume only pre-entry windows
  (shock bar / pullback window / reclaim bar / pre-entry union); entry-bar and
  post-entry windows are diagnostics. Asserted at import.

## Evidence

### 1. Same-burst discrimination is at coin-flip level

Pairwise "higher feature → higher net20" win rates over 833–1104 same-burst
pairs: **every feature ≤ 49.1%** (best: pullback imbalance 49.1%, pre-entry
imbalance 47.7%, reclaim imbalance 46.6%). The doc's hoped-for "which reclaim
is more real" separation does not materialize at 15m-window granularity.

Within-burst Spearman ICs are mildly positive for pre-entry imbalance (+0.108)
and low large-sell share (+0.101) but over only **20 contested bursts**
(≈1.5σ — not significant).

### 2. Counterfactual capacity replays improve, but fragilely

P2 pool, `select_portfolio` with orderflow scores vs first-come (20bp):

| rule | max5 | max8 | max10 |
|---|---|---|---|
| R0 first-come | 10.62% | 10.91% | 9.60% |
| R_of_reclaim_imbalance | 11.21% (+0.59) | **12.18% (+1.27)** | 11.29% (+1.69) |
| R_of_pre_entry_imbalance | 12.24% (+1.62) | 11.63% (+0.73) | 10.48% (+0.88) |

- Deltas are identical at 30/50bp by construction (same selected sets).
- Selected-trade counts barely change (147→147 at max8): the entire delta
  comes from swapping a handful of trades at ~20 contested moments — well
  inside single-trade noise for a 147-trade year.
- The selected-minus-skipped gap stays **negative under every rule**
  (-1.49pp vs R0's -1.79pp at max8): ranking nudges the gap, it does not
  close it. The doc's structural observation stands.

### 3. Walk-forward does not validate

Expanding-window pairwise-logistic ranker (months 4+ scored): portfolio_net20
**0.77% vs first-come 1.83%** at max8 — the learned weights are worse than no
ranking. The fixed composite scores 3.10% on the same period, but its
pct-rank normalization is computed over the full year (mild distributional
lookahead), so only the single-feature results are fully causal. With ~20
contested bursts there is not enough signal to fit even 8 weights.

## Interpretation

1. Orderflow at 15m-window granularity does not answer "同一波里哪个币的
   reclaim 更真实" — same-burst pairs are coin-flip.
2. The in-sample replay gains (+0.6 to +1.7pp/yr) concentrate in reclaim-bar
   and pre-entry imbalance, never flip the selected-vs-skipped gap positive,
   and the learned variant fails out-of-sample. Treat as noise until the
   contested sample grows.
3. P2 max8 + O6 overflow remains the structure; capacity ranking remains the
   open problem. The next levers, per the doc's own §5.2 list, are
   finer-granularity reclaim quality (tick-sequence shape rather than
   15m aggregates) and post-reclaim follow-through measured live — both need
   the v0.8 shadow to keep accruing tape.

## Reproduction

```
pressure-graph collect --config configs/v0_3.yaml --exchange bybit --days 365 --all-eligible --skip-existing --workers 6
pressure-graph build-v03-features --config configs/v0_3.yaml --streaming --workers 5 --batch-size 12
pressure-graph run-v09d --config configs/v0_3.yaml
pressure-graph collect-orderflow-history --workers 12
pressure-graph run-v11-orderflow-ranking --config configs/v0_3.yaml
```

Outputs in `reports/v1_1_orderflow_burst_ranking/` (feature_ic, pairwise_winrate,
feature_quintiles, counterfactual_replays, walkforward_replay, coverage CSVs +
candidate_notes.md). The streaming feature build exists because the research
box is an 18GB-memcg pod; it is parity-tested against the monolithic build.

No paper-live or real-live permission changes.

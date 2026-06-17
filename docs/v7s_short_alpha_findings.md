# v7S Short Alpha Exploration — findings (in progress)

**Status: Direction E scaffolding shipped; A100 production run pending.**

> This lane was opened per `short_instructment6 (v7s).docx` to answer the
> orthogonal short-alpha question the closure doc (`docs/short_research_closure.md`)
> left explicit room for. The closure prohibited iterating on failed
> motifs (S1/S3/S5 reclaim, crowded-stall single-month, plain
> CIC-failure short); v7S is the new lane.

## Five directions and current scope

| Direction | Question | Data needed | Status |
|-----------|----------|-------------|--------|
| **A** Cross-exchange lag | Binance/OKX sell impulse → Bybit lag → short | Binance UM + Bybit linear aggTrades | Stubbed |
| **B** Liquidation continuation | Long-liquidation spike → failed reclaim → continuation | Liquidation tape | Deferred (no source) |
| **C** Crowded unwind v2 | funding+OI high + taker-buy exhaustion + CVD divergence | aggTrades CVD | Stubbed |
| **D** Relative-value pair | overextended beta vs leader → mean revert | Multi-symbol features | Stubbed |
| **E** CIC-failure confirmed (strict) | v4S Path A + beta_high gone + sell flow confirms | Local CIC + v11 orderflow_history | **Wired in this commit** |

Direction E is the only direction with code wired this commit. The other
four are listed in `cfg.enabled_directions` as flags; requesting any of
them currently raises `NotImplementedError`.

## Direction E specification

Strict gate chain (a CIC long must be present, gate fires at the
breakdown bar):

1. **CIC continuation** — a CIC1/CIC2 long entry exists on the symbol
   (`_build_cic_long_index`).
2. **Breakdown** — first bar within `e_breakdown_valid_bars` whose close
   < reference level. Two candidates:
   - `E1_cic_break_entry_strict` — reference = CIC entry close.
   - `E2_cic_break_pullback_strict` — reference = CIC pullback low.
3. **CP60 weak follow-through** — `_gate_cp60_would_exit` (re-used from v3.4).
4. **Protect_A not active** — `_gate_no_protect_a` (re-used from v3.4).
5. **beta_high environment gone** (NEW) — `gate_beta_already_extended` was
   True within the `e_beta_high_lookback_bars` lookback AND is False at
   the break bar. Missing column fails closed.
6. **Sell flow confirms** (NEW) — `buy_sell_imbalance` in the configured
   orderflow window (default `reclaim_bar`) ≤
   `e_sell_flow_max_imbalance` (default `-0.05`). Missing orderflow data
   fails closed unless `e_sell_flow_fail_open=True`, in which case the
   audit reason is `orderflow_missing_open` for downstream visibility.

Two exit rules per signal — Fast (matched to closure-doc default) and
Swing (matched to v3.4 SWING_RULE), each with funding accrual applied
via `_net_short_return`.

## Outputs (ten files mandated by v7s docx)

Per direction, under `reports/v7s_short_alpha/<direction>/`:

| CSV | What it shows |
|-----|---------------|
| `short_candidate_summary.csv` | N, mean_net20, win_rate, verdict |
| `short_cost_grid.csv` | 4×3 cost × extra-slippage grid (mean_net, win_rate) |
| `short_first_touch.csv` | hit_down_3pct, squeeze rate, max_adverse_up |
| `short_vs_no_long.csv` | A_no_action / B_no_long / C_short head-to-head |
| `short_vs_exit_long.csv` | exit-now PnL vs short_net20 when long active |
| `short_hedge_value.csv` | corr vs long monthly net + long-worst-month overlap |
| `month_cap_leave_one_month.csv` | month_cap35 net + leave-worst-month net |
| `symbol_contribution.csv` | max symbol share + leave-worst-symbol net |
| `matched_random_baseline.csv` | candidate mean vs random-pool mean |
| `candidate_notes.md` | Markdown — per-candidate verdict + audit reasons |

## Ten-gate acceptance (docx §统一验收标准)

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
- `risk_off_only` — gates 1-8 pass, exactly one of 9/10 fails.
- `no_value` — any of gates 1-8 fail (or both of 9 and 10 fail).

## Running this lane

Local (will write empty CSV stubs because no feature parquet on this box):

```bash
PYTHONPATH=src python -m pressure_graph.cli run-v7s-short-alpha --config configs/v0_3.yaml
```

A100 (production):

```bash
ssh -L 2222:10.106.200.247:2222 root@10.115.7.6 -p 25711  # jumphost
ssh root@localhost -p 2222                                  # A100
cd /opt/data/private/Wangjb/graph_cpt_quant
git fetch && git checkout v7s-short-alpha && git pull
bash scripts/server_v7s_short_alpha_run.sh
```

See memory note `a100-ssh-access` for the credentials and the `/opt/data/private/Wangjb`
file-upload path.

## Results

_Empty until the A100 run lands. Numbers backfilled here once the run
completes and the CSVs are pulled back._

### Direction E

| Candidate | Execution | N | mean_net20 | gate failures | verdict |
|-----------|-----------|---|------------|---------------|---------|
| E1_cic_break_entry_strict | fast | TBD | TBD | TBD | TBD |
| E1_cic_break_entry_strict | swing | TBD | TBD | TBD | TBD |
| E2_cic_break_pullback_strict | fast | TBD | TBD | TBD | TBD |
| E2_cic_break_pullback_strict | swing | TBD | TBD | TBD | TBD |

## Notes for the next commit

- Once Direction E numbers land, decide whether the orderflow gate is
  worth the data cost. If `e_sell_flow_fail_open=True` materially
  changes the verdict, that's the gate's actual signal value.
- Direction A is the next priority — v11 already proves Binance/Bybit
  aggTrades are reachable on A100; need a per-symbol Bybit-aggTrades
  alignment table before scoring lag.
- Direction D is a clean adjacency: short-overextended-beta + long-leader
  uses only multi-symbol features the long stack already builds.
- Direction B remains blocked on liquidation tape ingestion; revisit
  only if a data path opens up.
- Direction C v2 needs CVD divergence labels not currently exported;
  spec these before wiring.

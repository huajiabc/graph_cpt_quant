# Sell-Pressure Propagation Map — Design (Phase 1)

> Source: goal docx 2026-06-18. "卖压传播型空头 alpha" / "Downside Propagation Graph".
> Deliverable is a **map**, not a strategy. No S1/S3/S5, no CIC failure short, no breakdown-by-price short, no flipped-long, no "rose too much".

## Question

Given that an active sell-pressure event occurs on a **source** symbol-bar `t`, does a
**target** symbol-bar in a defined **lag window** follow through downward with a stable,
non-random magnitude? Where (which source → target edges)? On what lag? Which months?

## Inputs (Phase 1 — local data)

Continuous Binance USDT-M aggTrades-derived CVD parquet, already on disk:

- Path: `data/orderflow_history/binance_um/continuous/<SYM>/<bar_size>/<YYYY-MM>.parquet`
- Bar size: **5min** (the goal doc names 5m/15m/1h/4h as primary lag scales).
- Symbols (intersection): AVAXUSDT, DOGEUSDT, ETHUSDT, INJUSDT, NEARUSDT, ORDIUSDT, SEIUSDT, WLDUSDT.
- Overlap window: **2025-06-01 → 2025-11-15** (5.5 months; ETH extends to 2026-02-06 — used only for forced-unwind probe).
- Per-bar features used: `cvd_delta_volume`, `taker_buy_ratio`, `buy_sell_imbalance`,
  `large_sell_count`, `large_buy_count`, `volume`, `turnover`, `coverage_ratio`, `source_quality`.
- Bar VWAP (= turnover / volume) is used as the price reference for return measurement
  at this resolution — no separate OHLCV is needed.

## Out of scope this phase

- **Cross-exchange propagation.** Local data is single-exchange. Documented as data gap
  in [§Phase 2 — Cross-Exchange](#phase-2--cross-exchange-data-gap) below; not a
  silent skip.
- **BTC as leader.** No continuous BTC CVD locally. ETH is the only liquid leader on
  disk. Adding BTC continuous CVD is the first Phase-2 backfill.
- **Strategy/PnL.** No execution, no fees, no slippage. We are measuring edges, not
  trading them.

## Event types (source sell-pressure event)

A bar `t` on a source symbol is tagged with at least one of the following events. All
features are computed per source over the symbol's own history; thresholds are
*rolling* over the last `W = 288` bars (= 24h at 5min) so they self-adapt to regime.

| code | name | rule on bar `t` |
|------|------|-----------------|
| E1 | CVD breakdown | `cvd_delta_volume` z-score over `W` ≤ −3.0 |
| E2 | Taker sell imbalance | `taker_buy_ratio ≤ 0.30` AND `volume ≥ 2 × W-median` |
| E3 | Large-sell cluster | `large_sell_count − large_buy_count` ≥ 95th-pctile over `W`, and ≥ 3 |
| E4 | Sustained sell continuation | three consecutive bars `t−2..t` with negative `cvd_delta_volume`, each with z ≤ −1.5 over `W` |

A bar may carry multiple event codes; events are evaluated independently and combined
later. `E_any` = OR of all four. Coverage gate: bars require
`coverage_ratio ≥ 0.8` AND `source_quality == 'complete'`; otherwise no event is
emitted.

## Lag windows

For each event at source time `t`, the target's response is measured over
non-overlapping windows after `t`:

| code | window |
|------|--------|
| L05 | (t, t+5min]   — i.e. the next 1 target bar    |
| L15 | (t, t+15min]  — next 3 target bars             |
| L30 | (t, t+30min]  — next 6 target bars             |
| L60 | (t, t+60min]  — next 12 target bars            |
| L240 | (t, t+240min] — next 48 target bars (4h)      |

Target return: VWAP(end of window) / VWAP(t) − 1. **A short profits when this is
negative.**

## Per-edge outputs

For each `(source, target, event_type ∈ {E1..E4, E_any}, lag_window)`:

| column | meaning |
|--------|---------|
| `n_events` | count of source events satisfying the gate |
| `target_response_mean` | mean target VWAP return over the lag window |
| `target_response_median` | median target VWAP return |
| `short_return_mean` | `−target_response_mean` (annotation for downstream readers) |
| `adverse_upsample` | P(target return ≥ +10 bps in lag window) — squeeze proxy |
| `target_cvd_followthrough_rate` | P(target's own `cvd_delta_volume` z ≤ −1 inside the lag window) |
| `month_distribution_max_share` | max single-month share of events (concentration check) |
| `month_distribution_active_months` | number of distinct months with ≥1 event |
| `shuffled_null_mean` | mean target return at same number of randomly-placed timestamps |
| `bootstrap_ci_low` | 5th pctile of (observed − shuffled) over 1000 resamples |
| `bootstrap_ci_high` | 95th pctile of (observed − shuffled) |
| `edge_strength` | (observed_short_return − shuffled_short_return) / shuffled_std; standardized improvement over random |

`edge_strength` > 0 and `bootstrap_ci_low` > 0 (i.e. observed strictly more negative
than shuffled) is the bar for "edge exists". The closure doc gates from
[short-research-closure.md](short_research_closure.md) still apply *before any
promotion to a strategy*; this phase only produces the map.

## Three explored paths

1. **Leader → beta** — every directed `source → target` pair with `source ≠ target`
   among the 8 symbols (56 edges) × 4 event codes × 5 lag windows = 1,120 rows. The
   goal is to find ordered pairs where the source event has a stable downward
   follow-through on the target.
2. **Forced-unwind continuation** — `source == target`, i.e. does the same symbol
   keep falling after its own sell-pressure event? 8 symbols × 4 events × 5 lags
   = 160 rows. Compares event-positive vs same-symbol shuffled-null.
3. **Cross-exchange (Phase 2 — data gap).** Documented below; not run this phase.

## Significance — shuffled null

For each (source, event_type) we draw `n_events` random timestamps uniformly from the
source's covered period, then run the same target-response measurement at those random
times. We do this 1,000 times and take the mean / 5-95 pctile of the difference
(observed − shuffled). The bootstrap CI is over the *difference* of paired draws, not
the observed alone — this is the correct test of "is this stronger than a random
edge?". An edge with `bootstrap_ci_low ≤ 0` is treated as not-distinguishable from
random and dropped from the headline map.

## Phase 2 — cross-exchange (data gap)

We do **not** have OKX, Bybit, Hyperliquid aggTrades on disk. The cross-exchange
question in the goal doc cannot be answered with what's local. Phase-2 work to enable
it:

1. Add a Bybit linear-perp aggTrades backfill mirroring
   `binance_continuous_cvd.py`. Bybit publishes per-day trade dumps freely.
2. Add an OKX swap trades backfill (paginated REST or zip dump if available).
3. Extend `sell_pressure_propagation` to take an `(exchange, symbol)` tuple as both
   source and target. The event detector, lag-window response, shuffled null and
   bootstrap CI all generalize without change.

Until Phase 2 lands, the cross-exchange row of the map is explicitly blank rather than
silently empty. **No silent caps.**

## Files

| path | role |
|------|------|
| `src/pressure_graph/reports/sell_pressure_propagation.py` | core module: events, response, null, aggregation |
| `tests/test_sell_pressure_propagation.py` | pytest tests with synthetic data |
| `reports/sell_pressure_propagation/edge_map.parquet` | long-form per-edge table |
| `reports/sell_pressure_propagation/edge_map.csv` | mirror of parquet, human-readable |
| `reports/sell_pressure_propagation/summary.md` | human-readable map summary |

## Success criteria (per goal docx)

> 找到至少一个"卖压传播边": source sell pressure 出现后, target 在固定延迟窗口内有
> 稳定跟随下跌, 并且这个关系不是 random / shuffled edge.

Concretely, this phase passes if **at least one directed edge** has:
- `n_events ≥ 30` (stat-meaningful),
- `month_distribution_active_months ≥ 3` (not a one-month artefact),
- `month_distribution_max_share ≤ 0.50` (not a single-month spike),
- `bootstrap_ci_low > 0` (observed short return strictly above shuffled null),
- `adverse_upsample ≤ 0.40` (squeeze proxy not dominating).

If none pass, the map is still the deliverable — but the headline is "no propagation
edge survives shuffled-null on this universe at this resolution; recommend Phase 2
data expansion before deciding."

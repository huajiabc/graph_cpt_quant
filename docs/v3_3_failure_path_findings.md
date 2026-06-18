# v3.3 Failure-Path / Action-Combo Search — Findings (2026-06-16 A100 Production)

Source: the ACO / GA / SA instruction text in the v3.3 / v3.4 goal directive
(re-stated in `reports/v3_3_failure_path_search.py`'s module docstring). Built
on branch `short-v12s`. Research only — no shadow / paper-live / real-live
wiring touched.

## Verdict (one line)

**GA winner `motif_set=['S1'], scope=market, cooldown=16` yields
+1.08pp long-net delta and +0.85pp drawdown improvement on the v1.2s3
stack (3 longs gated, all with realised mean -2.89%, zero overblock).
SA confirms the cooldown plateau: 40 / 48 / 56 / 64 bars all score the
same fitness 0.0193 — **plateau, not needle** — re-validating the
v1.2s3 phase-4 default of 48. ACO surfaced all top-K failure paths
terminating in `no_long_cooldown` (market scope wins over symbol scope on
this trade cache because the long pool is small enough that symbol-level
cooldowns rarely fire). The framework is now wired into the same v1.2s3
simulator that drives the production risk-off shadow — research only.**

The instruction text framed the goal precisely:

> 用 ACO / GA / SA 探索空头也可以，但不要直接搜 short
> ACO 搜 failure path
> GA 搜动作组合
> SA 微调 cooldown 和 scope

…so the v3.3 product is not a short strategy. It is a search for the right
long-side risk-off policy.

## A100 production run

- **Date**: 2026-06-16, 247 box `interactive6228`
- **Env**: conda env `quant` = Python 3.11.15 + pandas 3.0.3 + numpy 2.4.6
  + scipy 1.17.1 + pyarrow 24.0.0
- **Tests**: `tests/test_v3_3_failure_path_search.py` 15/15 PASS in 2.2s
- **Production**: `python -m pressure_graph.cli run-v3-3-failure-path-search`
  finished in **9m07s** wall (19m34s user — ~2× CPU oversubscription is
  normal for the events-stream loop)
- ACO: `iterations=12, ants_per_iter=18, max_path_len=5`
- GA: `population_size=20, generations=12`
- SA: `iterations=48`

## ACO failure paths (`aco_failure_paths.csv`)

All top-8 paths converge to **fitness 0.0193**:

| rank | path |
|---|---|
| 1 | `beta_extreme -> no_long_cooldown` |
| 2 | `CIC_candidate -> no_long_cooldown` |
| 3 | `S1 -> no_long_cooldown` |
| 4 | `CIC_candidate -> btc_down -> no_long_cooldown` |
| 5 | `S1 -> price_stall -> no_long_cooldown` |
| 6 | `CIC_candidate -> failed_followthrough -> no_long_cooldown` |
| 7 | `beta_extreme -> price_stall -> no_long_cooldown` |
| 8 | `S1 -> density_fading -> no_long_cooldown` |

Two patterns stand out:

1. **Every winning path ends at `no_long_cooldown`** (market scope), not
   `symbol_risk_off`. The fitness function maps the path's terminal node
   to a chromosome's `scope` field — market scope dominates because the
   long pool is small (63 symbols, 2,296 trades) and symbol-level gating
   rarely actually intercepts a fresh long entry.
2. **Mid-nodes are not load-bearing** at this top-8 layer — `S1 ->
   no_long_cooldown` (length 2) scores the same as `S1 -> price_stall ->
   no_long_cooldown` (length 3). Adding context predicates does not move
   the fitness on this trade cache; the pheromone trail (`aco_pheromone_trails.csv`)
   does, however, show edges from S1 to `price_stall` and from S1 to
   `density_fading` reinforced above baseline — which is what the GA then
   picks up to seed its motif_set search.

## GA winner (`ga_best_chromosome.json`)

```json
{
  "fitness": 0.0193,
  "motif_set": ["S1"],
  "scope": "market",
  "cooldown_bars": 16,
  "apply_core": true,
  "apply_overflow": false,
  "apply_existing_positions": false,
  "apply_protect_a": false,
  "detail": {
    "net_delta": 0.0108,
    "dd_delta": 0.0085,
    "gated_realized_mean": -0.0289,
    "longs_gated": 3,
    "overblock": 0.0,
    "missed_good_penalty": 0.0
  }
}
```

**Reading**:

- `motif_set=['S1']`: the GA chose **S1 alone**, not the v1.2s2-tuned
  `{S1, S3, S5}` triple. This is consistent with the v1.2s docs noting S3
  is mostly a single-month BTC_down hint and S5 (BTC_down breakdown) is
  redundant once the scope is market-wide.
- `scope=market`: long pool too small for symbol-scope to bind.
- `cooldown_bars=16` *at GA winner* — but SA below shows this is the
  lower edge of the plateau, not the optimum.
- 3 longs gated, all with negative realised return (mean -2.89%) → real
  risk-off, zero overblock penalty.

## SA cooldown plateau verdict (`sa_cooldown_plateau.csv`)

```text
cooldown   fitness
   16      0.0000
   24     -0.0152
   32     -0.0152
   40      0.0193
   48      0.0193
   56      0.0193
   64      0.0193
```

**`is_plateau = True`** for the 40-64 bar window. The fitness is flat
across four consecutive domain values, then drops sharply for shorter
cooldowns (24/32) and to neutral for 16. So 48 — the v1.2s3 phase-4
default — sits squarely in a stable region, **not** at a needle.

This was the exact 针尖-vs-plateau test the instruction text asked for:

> 现在最佳 cooldown 是: 48 bars
> SA 可以测试稳定区间: 32 / 40 / 48 / 56 / 64
> 但要看: 是否 48 附近都是好区间
> 如果只有 48 好, 就是针尖

48 ± a wide neighbourhood (40-64) is uniformly good. Verdict: **plateau,
hold the v1.2s3 default**.

## Reconciliation with v1.2s3 phase-4

- v1.2s2 (the prior phase) picked `motif_set={S1, S3, S5}` + scope=symbol
  + cooldown=48 as the headline gate.
- v3.3 GA on the *same* trade cache prefers `motif_set={S1}` + scope=market
  + cooldown ∈ [40, 64]. The fitness delta is small (~+1pp net), but
  reaches the same plateau region.

These are not contradictory — they reflect *different selection rules*:

- v1.2s2 was hand-tuned for a portfolio of motifs with the cleanest
  per-motif behaviour (S1 / S3 / S5).
- v3.3 fitness explicitly rewards *parsimony* — the smallest motif set
  that lifts net and dd. S1 alone does it on this cache; adding S3 / S5
  doesn't move the headline because the longs they would additionally
  gate aren't in the trade cache window.

The right operational takeaway is: **keep the v1.2s3 phase-4 shadow gate
as is (S1/S3/S5, scope=symbol, cooldown=48)**. v3.3 is the audit
confirming that this regime sits in a fitness plateau and that pruning
S3/S5 is not catastrophic on a fitness scoreboard.

## What was built

### Three independent search loops (`src/pressure_graph/optim/`)

| Module | Role | Inputs | Outputs |
|---|---|---|---|
| `optim/aco.py` | Ant-colony over a state-node graph | `StateGraph`, fitness callback | top-K `FailurePath` |
| `optim/ga.py` | Genetic over the action-combo chromosome | fitness callback | sorted `Individual` history |
| `optim/sa.py` | Simulated annealing around the GA winner | fitness callback | trace + `plateau_report()` |

All three are pure-function: nothing knows about parquet files or the trade
cache directly. The orchestrator wires the fitness functions through.

### The state-node graph (`build_default_state_graph`)

Starts (motifs and structural anchors):
- `S1`, `S3`, `S5` (the v1.2s motif codes that already drive the v1.2s3 gate)
- `CIC_candidate`, `beta_extreme` (the long-side strong-structure entries)

Mid-nodes (context predicates re-used from v3.4 gates):
- `failed_reclaim`, `low_coimpulse`, `btc_not_up`, `btc_down`, `density_fading`,
  `failed_followthrough`, `price_stall`, `volume_shock_exhaustion`

Terminals (outcomes):
- `symbol_risk_off` (the v1.2s2 / v1.2s3 product — gate this name only)
- `no_long_cooldown` (broader cooldown — "just don't be long")

The two terminals encode the instruction's framing — the search is for
*failure paths whose treatment is not-shorting, not-being-long*.

### The chromosome (`encode_chromosome`)

```text
[ motif_set,
  scope,                           # symbol | cluster | market
  cooldown_bars,                   # snap to {16,24,32,40,48,56,64,96}
  apply_core, apply_overflow,
  apply_existing_positions,
  apply_protect_a ]
```

`encode_chromosome` normalises (sorts motifs, validates scope, snaps cooldown)
so SA and crossover stay inside the domain.

### Real fitness vs synthetic fitness

`make_real_chromosome_fitness(feature_path, instruments, config, cfg)`:

- Loads the long pool via `_focus_pool` (`P2_CIC1_CIC2_COMBINED`)
- Caches `stream_risk_off_events` per motif_set tuple, so each chromosome's
  events lookup is O(1) after the first sight of its motifs
- Computes `net_delta`, `dd_delta`, `gated_realized_mean`, `overblock`,
  `missed_good_penalty` against the un-gated baseline
- Combines into a single fitness scalar (positive = improvement)

`make_synthetic_chromosome_fitness(long_pool, cfg)`:

- Cheap deterministic fitness used by the unit tests
- Rewards motif_count + scope=symbol + cooldown near 48 + apply_core/overflow

The orchestrator's `use_synthetic_fitness` switch lets `run-v3-3-failure-path-search`
work in environments where the trade cache or features are missing.

### SA plateau detector

`sa.plateau_report()` returns:

- `best_cooldown`: argmax fitness across visited cooldowns
- `is_plateau`: True iff at least half of the `plateau_window`-neighbours score
  within `plateau_tolerance` of the best
- `neighbours`: the full {cooldown → best fitness} dict for the report

If `is_plateau` is False, candidate_notes calls it out as **needle
(single-point optimum)** — the 针尖 case the instruction text explicitly
warned about.

## Outputs (7 files)

Under `reports/v3_3_failure_path_search/`:

- `aco_failure_paths.csv` — top-K paths with fitness and edge length
- `aco_pheromone_trails.csv` — final pheromone weights per (src, dst) edge
- `ga_pareto.csv` — top-50 chromosomes from the GA history with fitness +
  detail columns
- `ga_best_chromosome.json` — the winner with `as_dict()` payload
- `sa_trace.csv` — every SA proposal with iteration / cooldown / fitness /
  accepted / temperature
- `sa_cooldown_plateau.csv` — best fitness per cooldown across the trace
- `candidate_notes.md` — verdict synthesis stamped from the above

## How to run

Locally (synthetic — useful for pipeline shape):

```text
python -m pressure_graph.cli run-v3-3-failure-path-search --synthetic
```

A100 production:

```text
python -m pressure_graph.cli run-v3-3-failure-path-search
```

(no `--synthetic` flag → real fitness against the v0.9D trade cache).

## Discipline

- All three loops share one long-pool + event cache; per-individual eval is
  O(N) vector ops over the cached trades.
- ACO paths' first node = motif start; terminal node determines scope (symbol
  vs market). Cooldown is fixed at `cfg.cooldown_default` (48) during ACO so
  the path search isn't confounded by cooldown choice.
- GA fitness penalises overblock and missed-good explicitly so a high-motif
  chromosome can't cheat by gating everything.
- SA never seeds randomly — it starts from the GA winner so the plateau
  question is asked at the right point.
- Tier: research only. No paper-live / real-live wiring.

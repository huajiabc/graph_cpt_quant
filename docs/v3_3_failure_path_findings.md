# v3.3 Failure-Path / Action-Combo Search — Framework + Pending A100 Run

Source: the ACO / GA / SA instruction text in the v3.3 / v3.4 goal directive
(re-stated in `reports/v3_3_failure_path_search.py`'s module docstring). Built
on branch `short-v12s`. Research only — no shadow / paper-live / real-live
wiring touched.

## Verdict (provisional)

**Framework landed. ACO + GA + SA loops run end-to-end against the v1.2s3
current-stack simulator (`make_real_chromosome_fitness`) or a synthetic
fitness for unit tests. Locally — with the trade cache absent — the
synthetic smoke run produces all seven expected outputs (4 CSV, 1 JSON, 1
trace, 1 notes). The production verdict — whether the GA winner's cooldown
sits in a plateau (per the instruction text's针尖 vs plateau test) — depends
on the A100 run with the real v0.9D trade cache. This document will be
updated once that run produces real CSVs.**

The instruction text framed the goal precisely:

> 用 ACO / GA / SA 探索空头也可以，但不要直接搜 short
> ACO 搜 failure path
> GA 搜动作组合
> SA 微调 cooldown 和 scope

…so the v3.3 product is not a short strategy. It is a search for the right
long-side risk-off policy.

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

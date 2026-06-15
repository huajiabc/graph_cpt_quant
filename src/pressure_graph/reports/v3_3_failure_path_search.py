"""v3.3 Failure-Path / Action-Combo Search.

Orchestrates ACO + GA + SA on top of the v1.2s3 current-stack risk-off
simulator. The single objective behind all three loops is the long book —
we are NOT searching for short signals, we are searching for the right risk-off
policy that improves the long book's net20 / drawdown / sample efficiency.

Stages:

1. **ACO failure-path mining.** Walks a state-node graph (motifs as start
   nodes, context predicates as middle nodes, "no-long" / "symbol risk-off"
   as terminal nodes) and reinforces paths that empirically reduce long-book
   loss. Output: top-K paths suitable as motif_set candidates for the GA.

2. **GA action-combo search.** Chromosome ``[motif_set, scope, cooldown,
   apply_core, apply_overflow, apply_existing_positions, apply_protect_a]``.
   Fitness = long_net20 + dd_improvement − overblock − missed_good. Output:
   ranked individuals + the best chromosome JSON.

3. **SA cooldown/scope plateau check.** Seeded from the GA winner, perturbs
   cooldown ±1 step (16/24/32/40/48/56/64/96) and scope (symbol/cluster/market)
   to confirm 48 (or whatever the GA chose) sits in a plateau, not a needle.

All three loops share the same long-pool + risk-off-events cache so a single
end-to-end run is cheap (per-chromosome eval ≈ vector ops over O(N) trades).

Tier: research only. No paper-live / real-live wiring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.optim import (
    ACOConfig,
    AntColonyOptimizer,
    Chromosome,
    GAConfig,
    GeneticOptimizer,
    SAConfig,
    SimulatedAnnealer,
    StateGraph,
    encode_chromosome,
)

REPORT_ROOT = Path("reports/v3_3_failure_path_search")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
DEFAULT_LONG_POOL = "P2_CIC1_CIC2_COMBINED"


# --------------------------------------------------------------------------------------
# State graph — frozen at module level so ACO runs are reproducible.
# --------------------------------------------------------------------------------------


def build_default_state_graph() -> StateGraph:
    """The v3.3 failure-path graph: motif -> context -> outcome.

    Two outcome terminals encode the goal of the search:
      * ``symbol_risk_off`` — gate longs on this name only
      * ``no_long_cooldown`` — gate longs name-wide for a cooldown window

    The "no_long" terminal carries the instruction text's framing — the search
    is for failure paths whose treatment is *not opening short*, but *not
    being long*.
    """
    starts = ("S1", "S3", "S5", "CIC_candidate", "beta_extreme")
    mids = (
        "failed_reclaim",
        "low_coimpulse",
        "btc_not_up",
        "btc_down",
        "density_fading",
        "failed_followthrough",
        "price_stall",
        "volume_shock_exhaustion",
    )
    terminals = ("symbol_risk_off", "no_long_cooldown")
    nodes = starts + mids + terminals
    adjacency: dict[str, tuple[str, ...]] = {}
    for start in starts:
        adjacency[start] = mids + terminals
    for mid in mids:
        # Mids can chain into other mids or terminate.
        adjacency[mid] = tuple(m for m in mids if m != mid) + terminals
    for term in terminals:
        adjacency[term] = ()
    return StateGraph(
        nodes=nodes,
        adjacency=adjacency,
        start_nodes=starts,
        terminal_nodes=terminals,
    )


# --------------------------------------------------------------------------------------
# Long-pool + event-cache loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class V33Config:
    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    long_pool_name: str = DEFAULT_LONG_POOL
    top_n: int = 30
    aco: ACOConfig = field(default_factory=lambda: ACOConfig(iterations=12, ants_per_iter=18, max_path_len=5))
    ga: GAConfig = field(default_factory=lambda: GAConfig(generations=12, population_size=20))
    sa: SAConfig = field(default_factory=lambda: SAConfig(iterations=48))
    overblock_penalty_scale: float = 0.3
    missed_good_penalty_scale: float = 0.5
    cooldown_default: int = 48


def _load_long_pool(cfg: V33Config) -> pd.DataFrame:
    """Load the focal long pool. Empty frame if the trade cache is missing."""
    if not cfg.trade_cache_path.exists():
        return pd.DataFrame()
    # Deferred import to keep this module importable when v06a1 / scipy is broken.
    from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool

    trades = read_parquet(cfg.trade_cache_path)
    pool = _focus_pool(trades, cfg.long_pool_name)
    if pool.empty:
        return pool
    pool = pool.copy()
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["signal_time"]).reset_index(drop=True)
    pool["net_return"] = pd.to_numeric(pool["net_return"], errors="coerce").fillna(0.0)
    return pool


# --------------------------------------------------------------------------------------
# Synthetic fitness — used when the real cache / event stream is unavailable.
# Real fitness (against v12s3 simulator) is plugged in by ``make_real_fitness``.
# --------------------------------------------------------------------------------------


def make_synthetic_path_fitness(long_pool: pd.DataFrame) -> Callable[[tuple[str, ...]], float]:
    """A cheap fitness for failure paths: longer + more context-rich = better,
    *if* the long pool has any losing rows that share the path's motif anchor.

    This is intentionally simple — the real orchestrator plugs in a fitness
    that runs the v12s3 stack simulator. The synthetic version is enough for
    unit-testing the ACO loop without the trade cache.
    """
    avg_loss = 0.0
    if not long_pool.empty:
        nets = pd.to_numeric(long_pool["net_return"], errors="coerce")
        avg_loss = float((-nets[nets < 0]).mean()) if (nets < 0).any() else 0.0

    def _fit(path: tuple[str, ...]) -> float:
        if len(path) < 2:
            return 0.0
        # Path bonus: motif start + at least one context mid + terminal outcome.
        starts = {"S1", "S3", "S5", "CIC_candidate", "beta_extreme"}
        terminals = {"symbol_risk_off", "no_long_cooldown"}
        has_start = path[0] in starts
        has_terminal = path[-1] in terminals
        score = 0.0
        score += 1.0 if has_start else 0.0
        score += 1.0 if has_terminal else 0.0
        score += 0.2 * sum(1 for n in path[1:-1] if n not in starts and n not in terminals)
        score += avg_loss * 10.0 * (1.0 if has_terminal else 0.0)
        return float(score)

    return _fit


def make_synthetic_chromosome_fitness(
    long_pool: pd.DataFrame,
    cfg: V33Config,
) -> Callable[[Chromosome], tuple[float, dict]]:
    """A simple chromosome fitness: motifs reinforce, scope/cooldown shape it.

    Used when the v12s3 simulator can't be invoked locally (synthetic envs / unit
    tests). The real orchestrator overrides with ``make_real_chromosome_fitness``.
    """

    avg_loss = 0.0
    if not long_pool.empty:
        nets = pd.to_numeric(long_pool["net_return"], errors="coerce")
        avg_loss = float((-nets[nets < 0]).mean()) if (nets < 0).any() else 0.0

    def _fit(chromosome: Chromosome) -> tuple[float, dict]:
        # Motif count, scope preference (symbol > cluster > market for cleanliness),
        # cooldown preference (48 ± plateau), apply flags reward.
        motif_score = 0.4 * len(chromosome.motif_set)
        scope_score = {"symbol": 0.6, "cluster": 0.3, "market": 0.1}.get(chromosome.scope, 0.0)
        cooldown_score = max(0.0, 1.0 - abs(chromosome.cooldown_bars - cfg.cooldown_default) / 32.0)
        apply_score = 0.0
        apply_score += 0.5 * float(chromosome.apply_core)
        apply_score += 0.3 * float(chromosome.apply_overflow)
        apply_score += 0.1 * float(chromosome.apply_existing_positions)
        apply_score += 0.05 * float(chromosome.apply_protect_a)
        bonus = avg_loss * 10.0 * float(len(chromosome.motif_set))
        score = motif_score + scope_score + cooldown_score + apply_score + bonus
        return float(score), {
            "motif_score": motif_score,
            "scope_score": scope_score,
            "cooldown_score": cooldown_score,
            "apply_score": apply_score,
            "loss_bonus": bonus,
        }

    return _fit


# --------------------------------------------------------------------------------------
# Real fitness — couples to v12s3 stack simulator.
# --------------------------------------------------------------------------------------


def make_real_chromosome_fitness(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V33Config,
) -> Callable[[Chromosome], tuple[float, dict]]:
    """Build a chromosome fitness function backed by the v12s3 simulator.

    The fitness combines:
      * net20 lift vs un-gated baseline (positive contribution)
      * drawdown reduction (positive contribution)
      * overblock penalty (negative when many gated longs were winners)
      * missed-good penalty (negative when good longs were suppressed)

    Deferred imports keep this module importable in synthetic-test envs where
    the v3.4 feature stack can't load.
    """
    # Deferred: pulls v06a1 -> scipy through the feature reader.
    from pressure_graph.reports.v06c import _rank_inputs
    from pressure_graph.reports.v12s2_long_risk_off_overlay import (
        RiskOffConfig,
        _apply_market_gate,
        _apply_symbol_gate,
        _mode_metrics,
        stream_risk_off_events,
    )

    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n][
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    long_pool = _load_long_pool(cfg)
    events_cache: dict[tuple[str, ...], pd.DataFrame] = {}

    def _events_for(motifs: tuple[str, ...]) -> pd.DataFrame:
        if motifs in events_cache:
            return events_cache[motifs]
        cfg_events = RiskOffConfig(
            motifs=motifs,
            symbol_cooldown_bars=cfg.cooldown_default,
        )
        events = stream_risk_off_events(
            feature_path, rank30, rank90, symbols, config, cfg_events
        )
        events_cache[motifs] = events
        return events

    def _baseline_metrics() -> dict:
        if long_pool.empty:
            return {"portfolio_net20": 0.0, "max_drawdown_proxy": 0.0}
        no_gate = pd.Series(False, index=long_pool.index)
        return _mode_metrics(long_pool, no_gate, "baseline", max_positions=8)

    baseline = _baseline_metrics()

    def _fit(chromosome: Chromosome) -> tuple[float, dict]:
        if long_pool.empty or not chromosome.motif_set:
            return 0.0, {"reason": "empty_pool_or_motifs"}
        events = _events_for(chromosome.motif_set)
        run_cfg = RiskOffConfig(
            motifs=chromosome.motif_set,
            symbol_cooldown_bars=chromosome.cooldown_bars,
        )
        gate_fn = (
            _apply_symbol_gate
            if chromosome.scope == "symbol"
            else (_apply_market_gate if chromosome.scope == "market" else _apply_symbol_gate)
        )
        gated = gate_fn(long_pool, events, run_cfg)
        metrics = _mode_metrics(long_pool, gated, "v3_3_search", max_positions=8)
        net_delta = float(metrics["portfolio_net20"]) - float(baseline["portfolio_net20"])
        dd_delta = float(metrics["max_drawdown_proxy"]) - float(baseline["max_drawdown_proxy"])
        dd_improvement = max(0.0, dd_delta)  # less-negative dd is improvement
        gated_realized_mean = float(metrics.get("gated_realized_net_mean", 0.0) or 0.0)
        overblock = max(0.0, gated_realized_mean) * cfg.overblock_penalty_scale
        missed_good_count = int(metrics.get("longs_gated", 0)) if gated_realized_mean > 0 else 0
        missed_good_penalty = missed_good_count * cfg.missed_good_penalty_scale / 100.0
        score = net_delta + dd_improvement - overblock - missed_good_penalty
        return float(score), {
            "net_delta": net_delta,
            "dd_delta": dd_delta,
            "gated_realized_mean": gated_realized_mean,
            "longs_gated": int(metrics.get("longs_gated", 0)),
            "overblock": overblock,
            "missed_good_penalty": missed_good_penalty,
        }

    return _fit


def make_real_path_fitness(
    chromosome_fitness: Callable[[Chromosome], tuple[float, dict]],
    cfg: V33Config,
) -> Callable[[tuple[str, ...]], float]:
    """Map ACO paths onto Chromosome evaluations so both loops share the simulator."""

    motif_universe = {"S1", "S3", "S5"}

    def _fit(path: tuple[str, ...]) -> float:
        # First node is the motif start; restrict to known motifs.
        motifs = tuple(sorted(set(n for n in path if n in motif_universe))) or ("S1",)
        # Path includes a scope hint (symbol_risk_off -> "symbol"; no_long_cooldown -> "market").
        scope = "symbol"
        if "no_long_cooldown" in path:
            scope = "market"
        chromosome = encode_chromosome(
            motif_set=motifs,
            scope=scope,
            cooldown_bars=cfg.cooldown_default,
            apply_core=True,
            apply_overflow=False,
            apply_existing_positions=False,
            apply_protect_a=False,
        )
        score, _ = chromosome_fitness(chromosome)
        return float(score)

    return _fit


# --------------------------------------------------------------------------------------
# Output writers — five CSVs + JSON best chromosome + candidate_notes.md
# --------------------------------------------------------------------------------------


def _write_aco_outputs(report_root: Path, optimizer: AntColonyOptimizer, paths: list) -> dict[str, Path]:
    paths_df = pd.DataFrame(
        [
            {"rank": i + 1, "fitness": float(fp.fitness), "len": len(fp), "path": " -> ".join(fp.nodes)}
            for i, fp in enumerate(paths)
        ]
    )
    pheromone_df = pd.DataFrame(
        [{"src": a, "dst": b, "pheromone": w} for a, b, w in optimizer.pheromone_snapshot()]
    )
    paths_csv = report_root / "aco_failure_paths.csv"
    pheromone_csv = report_root / "aco_pheromone_trails.csv"
    paths_df.to_csv(paths_csv, index=False)
    pheromone_df.to_csv(pheromone_csv, index=False)
    return {"aco_failure_paths": paths_csv, "aco_pheromone_trails": pheromone_csv}


def _write_ga_outputs(report_root: Path, history: list) -> dict[str, Path]:
    pareto_df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "fitness": float(ind.fitness),
                **ind.chromosome.as_dict(),
                **{f"detail_{k}": v for k, v in ind.detail.items()},
            }
            for i, ind in enumerate(history[:50])
        ]
    )
    pareto_csv = report_root / "ga_pareto.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    best_json = report_root / "ga_best_chromosome.json"
    if history:
        best = history[0]
        best_json.write_text(
            json.dumps({"fitness": float(best.fitness), **best.chromosome.as_dict(), "detail": best.detail}, indent=2, default=str),
            encoding="utf-8",
        )
    return {"ga_pareto": pareto_csv, "ga_best_chromosome": best_json}


def _write_sa_outputs(report_root: Path, sa: SimulatedAnnealer) -> dict[str, Path]:
    trace_df = pd.DataFrame(
        [
            {
                "iteration": step.iteration,
                "cooldown_bars": int(step.chromosome.cooldown_bars),
                "scope": step.chromosome.scope,
                "fitness": float(step.fitness),
                "accepted": bool(step.accepted),
                "temperature": float(step.temperature),
            }
            for step in sa.trace
        ]
    )
    plateau = sa.plateau_report()
    plateau_df = pd.DataFrame(
        [
            {"cooldown_bars": cd, "fitness": score}
            for cd, score in plateau.get("neighbours", {}).items()
        ]
        + (
            [{"cooldown_bars": plateau["best_cooldown"], "fitness": plateau.get("best_fitness", float("nan"))}]
            if plateau.get("best_cooldown") is not None
            else []
        )
    )
    trace_csv = report_root / "sa_trace.csv"
    plateau_csv = report_root / "sa_cooldown_plateau.csv"
    trace_df.to_csv(trace_csv, index=False)
    plateau_df.to_csv(plateau_csv, index=False)
    return {"sa_trace": trace_csv, "sa_cooldown_plateau": plateau_csv}


def _write_notes(report_root: Path, paths: list, history: list, sa: SimulatedAnnealer) -> Path:
    plateau = sa.plateau_report()
    lines: list[str] = [
        "# v3.3 Failure-Path / Action-Combo Search",
        "",
        "Goal: search for better long risk-off policies, not short signals.",
        "ACO mines failure paths; GA tunes the action-combo chromosome; SA",
        "verifies the cooldown sits on a plateau (not a needle).",
        "",
    ]
    if paths:
        lines.append("## Top ACO failure paths")
        for i, fp in enumerate(paths[:8]):
            lines.append(f"- #{i + 1} fitness={fp.fitness:.4f}: {' -> '.join(fp.nodes)}")
        lines.append("")
    if history:
        best = history[0]
        lines.append("## GA winner chromosome")
        lines.append(f"- fitness={best.fitness:.4f}")
        for k, v in best.chromosome.as_dict().items():
            lines.append(f"  - {k}: {v}")
        lines.append("")
    if plateau.get("best_cooldown") is not None:
        verdict = "plateau (stable)" if plateau.get("is_plateau") else "needle (single-point optimum)"
        lines.append("## SA cooldown plateau verdict")
        lines.append(
            f"- best cooldown={plateau['best_cooldown']} bars, fitness="
            f"{plateau.get('best_fitness', float('nan')):.4f}; **{verdict}**."
        )
        for cd, score in sorted(plateau.get("neighbours", {}).items()):
            lines.append(f"  - neighbour cd={cd}: fitness={score:.4f}")
        lines.append("")
    lines.extend(
        [
            "## Discipline",
            "- All three loops share one long-pool + event cache; per-individual",
            "  evaluation is O(N) vector ops over the cached trades.",
            "- Synthetic-fitness fallback exists for unit tests; the production",
            "  run uses the v1.2s3 simulator via ``make_real_chromosome_fitness``.",
            "- Tier: research only. No paper-live / real-live wiring.",
        ]
    )
    notes_path = report_root / "candidate_notes.md"
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


# --------------------------------------------------------------------------------------
# Main entry — runs ACO, then GA, then SA on the GA winner.
# --------------------------------------------------------------------------------------


def write_v3_3_failure_path_search(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V33Config = V33Config(),
    *,
    use_synthetic_fitness: bool = False,
) -> dict[str, Path]:
    """End-to-end v3.3 driver.

    When ``use_synthetic_fitness=True`` (or the trade cache is missing) the
    synthetic fitness functions run — the framework still produces all outputs
    so the orchestrator can be smoke-tested without features.
    """
    report_root = ensure_dir(cfg.report_root)
    long_pool = _load_long_pool(cfg)

    if use_synthetic_fitness or long_pool.empty or not feature_path.exists():
        chromosome_fitness = make_synthetic_chromosome_fitness(long_pool, cfg)
        path_fitness = make_synthetic_path_fitness(long_pool)
    else:
        chromosome_fitness = make_real_chromosome_fitness(feature_path, instruments, config, cfg)
        path_fitness = make_real_path_fitness(chromosome_fitness, cfg)

    graph = build_default_state_graph()
    aco = AntColonyOptimizer(graph=graph, fitness_fn=path_fitness, cfg=cfg.aco)
    paths = aco.run()

    ga = GeneticOptimizer(fitness_fn=chromosome_fitness, cfg=cfg.ga)
    history = ga.run()

    seed = history[0].chromosome if history else encode_chromosome(
        motif_set=("S1", "S3", "S5"),
        scope="symbol",
        cooldown_bars=cfg.cooldown_default,
        apply_core=True,
        apply_overflow=True,
        apply_existing_positions=False,
        apply_protect_a=False,
    )

    def _sa_fitness(c: Chromosome) -> float:
        score, _ = chromosome_fitness(c)
        return float(score)

    sa = SimulatedAnnealer(seed=seed, fitness_fn=_sa_fitness, cfg=cfg.sa)
    sa.run()

    outputs: dict[str, Path] = {}
    outputs.update(_write_aco_outputs(report_root, aco, paths))
    outputs.update(_write_ga_outputs(report_root, history))
    outputs.update(_write_sa_outputs(report_root, sa))
    outputs["candidate_notes"] = _write_notes(report_root, paths, history, sa)
    return outputs


__all__ = [
    "DEFAULT_LONG_POOL",
    "REPORT_ROOT",
    "TRADE_CACHE_PATH",
    "V33Config",
    "build_default_state_graph",
    "make_real_chromosome_fitness",
    "make_real_path_fitness",
    "make_synthetic_chromosome_fitness",
    "make_synthetic_path_fitness",
    "write_v3_3_failure_path_search",
]

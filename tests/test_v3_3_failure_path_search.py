"""Unit tests for the v3.3 metaheuristic stack (ACO + GA + SA + orchestrator).

These tests exercise the search loops with the synthetic fitness paths so
nothing requires the v0.9D trade cache or the real feature parquet. The
production run on the A100 box uses the same harness with the real fitness
plugged in via ``make_real_chromosome_fitness``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pressure_graph.optim import (
    ACOConfig,
    AntColonyOptimizer,
    Chromosome,
    GAConfig,
    GeneticOptimizer,
    SAConfig,
    SimulatedAnnealer,
    StateGraph,
    decode_chromosome,
    encode_chromosome,
)
from pressure_graph.optim.ga import COOLDOWN_VALUES, MOTIF_UNIVERSE, SCOPE_VALUES, Individual
from pressure_graph.reports.v3_3_failure_path_search import (
    V33Config,
    build_default_state_graph,
    make_synthetic_chromosome_fitness,
    make_synthetic_path_fitness,
    write_v3_3_failure_path_search,
)


# ---------- StateGraph + ACO ----------------------------------------------------------


def test_build_default_state_graph_has_starts_mids_terminals():
    g = build_default_state_graph()
    assert "S1" in g.start_nodes
    assert "no_long_cooldown" in g.terminal_nodes
    assert "low_coimpulse" in g.nodes
    # Every start has at least one outgoing edge.
    for s in g.start_nodes:
        assert g.successors(s)
    # Terminals have no outgoing edges.
    for t in g.terminal_nodes:
        assert g.successors(t) == ()


def test_aco_runs_and_returns_paths_with_terminal_bias():
    g = build_default_state_graph()
    # Synthetic fitness rewards paths that end at a terminal.

    def fit(path: tuple[str, ...]) -> float:
        return 5.0 if path and path[-1] in g.terminal_nodes else 1.0

    aco = AntColonyOptimizer(
        graph=g,
        fitness_fn=fit,
        cfg=ACOConfig(ants_per_iter=10, iterations=6, max_path_len=4, rng_seed=42),
    )
    paths = aco.run()
    assert paths
    # The best path should end at a terminal node.
    assert paths[0].nodes[-1] in g.terminal_nodes


def test_aco_pheromone_reinforces_better_edges():
    g = StateGraph(
        nodes=("A", "B", "C", "D"),
        adjacency={"A": ("B", "C"), "B": ("D",), "C": ("D",), "D": ()},
        start_nodes=("A",),
        terminal_nodes=("D",),
    )
    # Fitness rewards A -> B -> D, penalises A -> C -> D.

    def fit(path: tuple[str, ...]) -> float:
        return 10.0 if "B" in path else 0.0

    aco = AntColonyOptimizer(
        graph=g,
        fitness_fn=fit,
        cfg=ACOConfig(ants_per_iter=20, iterations=10, max_path_len=3, rng_seed=7),
    )
    aco.run()
    snapshot = dict(((a, b), w) for a, b, w in aco.pheromone_snapshot())
    assert snapshot[("A", "B")] > snapshot[("A", "C")]


def test_aco_handles_fitness_exceptions():
    g = build_default_state_graph()

    def fit(path):
        raise RuntimeError("boom")

    aco = AntColonyOptimizer(
        graph=g, fitness_fn=fit,
        cfg=ACOConfig(ants_per_iter=4, iterations=3, max_path_len=3, rng_seed=1),
    )
    paths = aco.run()
    # Exception coerced to 0.0 -> still returns a sorted list (possibly all-zero).
    assert isinstance(paths, list)


# ---------- GA -------------------------------------------------------------------------


def test_encode_chromosome_normalises_motifs_and_snaps_cooldown():
    c = encode_chromosome(
        motif_set=("S2", "S1", "BAD"),
        scope="symbol",
        cooldown_bars=50,  # not in domain -> snap to nearest (48)
        apply_core=True,
        apply_overflow=False,
        apply_existing_positions=True,
        apply_protect_a=False,
    )
    assert c.motif_set == ("S1", "S2")
    assert c.cooldown_bars == 48
    assert c.scope == "symbol"


def test_encode_chromosome_rejects_unknown_scope():
    with pytest.raises(ValueError):
        encode_chromosome(("S1",), "moon", 48, True, True, False, False)


def test_decode_chromosome_round_trips_payload():
    c = encode_chromosome(("S1", "S5"), "cluster", 32, True, True, False, True)
    payload = c.as_dict()
    out = decode_chromosome(payload)
    assert out == c


def test_ga_run_returns_unique_and_sorted_population():
    pool = pd.DataFrame({"net_return": [-0.01, -0.02, 0.03]})
    cfg = V33Config(ga=GAConfig(population_size=12, generations=6, rng_seed=11))
    fitness = make_synthetic_chromosome_fitness(pool, cfg)
    ga = GeneticOptimizer(fitness_fn=fitness, cfg=cfg.ga)
    history = ga.run()
    # Sorted descending by fitness.
    assert all(history[i].fitness >= history[i + 1].fitness for i in range(len(history) - 1))
    # Best fitness is finite.
    assert np.isfinite(history[0].fitness)
    # Best chromosome's motif_set must be valid.
    assert set(history[0].chromosome.motif_set).issubset(set(MOTIF_UNIVERSE))


def test_ga_handles_seeded_chromosomes():
    pool = pd.DataFrame({"net_return": [-0.01]})
    cfg = V33Config(ga=GAConfig(population_size=8, generations=4, rng_seed=3))
    fitness = make_synthetic_chromosome_fitness(pool, cfg)
    seed = encode_chromosome(("S1", "S3"), "symbol", 48, True, True, False, False)
    ga = GeneticOptimizer(fitness_fn=fitness, cfg=cfg.ga, seeded_chromosomes=(seed,))
    history = ga.run()
    # Seeded chromosome must appear in history.
    matched = any(ind.chromosome == seed for ind in history)
    assert matched


# ---------- SA -------------------------------------------------------------------------


def test_sa_explores_cooldown_and_scope_neighbours():
    cfg = V33Config()
    fitness = lambda c: max(0.0, 1.0 - abs(c.cooldown_bars - 48) / 32.0)  # peak at 48
    seed = encode_chromosome(("S1",), "symbol", 32, True, True, False, False)
    sa = SimulatedAnnealer(seed=seed, fitness_fn=fitness, cfg=SAConfig(iterations=30, rng_seed=5))
    trace = sa.run()
    cooldowns = {step.chromosome.cooldown_bars for step in trace}
    # SA should have visited more than one cooldown.
    assert len(cooldowns) > 1


def test_sa_plateau_report_flags_plateau_when_neighbours_close():
    seed = encode_chromosome(("S1",), "symbol", 48, True, True, False, False)

    # Flat-ish fitness around 48 -> plateau.
    def fitness(c):
        return 1.0 - 0.0001 * abs(c.cooldown_bars - 48)

    sa = SimulatedAnnealer(seed=seed, fitness_fn=fitness, cfg=SAConfig(iterations=40, rng_seed=9))
    sa.run()
    report = sa.plateau_report()
    assert report["best_cooldown"] in COOLDOWN_VALUES
    assert report["is_plateau"] is True


def test_sa_plateau_report_flags_needle_when_neighbours_far():
    seed = encode_chromosome(("S1",), "symbol", 48, True, True, False, False)

    # Sharp needle at 48 -> neighbours score far worse.
    def fitness(c):
        if c.cooldown_bars == 48:
            return 1.0
        return 0.0

    sa = SimulatedAnnealer(
        seed=seed,
        fitness_fn=fitness,
        cfg=SAConfig(iterations=60, plateau_tolerance=0.05, plateau_window=2, rng_seed=4),
    )
    sa.run()
    report = sa.plateau_report()
    assert report["is_plateau"] is False


# ---------- Orchestrator end-to-end (synthetic) ---------------------------------------


def test_write_v3_3_failure_path_search_smoke_with_synthetic(tmp_path: Path):
    cfg = V33Config(
        report_root=tmp_path / "v3_3_report",
        trade_cache_path=tmp_path / "missing.parquet",  # forces synthetic path
        aco=ACOConfig(ants_per_iter=6, iterations=4, max_path_len=4, rng_seed=21),
        ga=GAConfig(population_size=8, generations=3, rng_seed=22),
        sa=SAConfig(iterations=15, rng_seed=23),
    )
    outputs = write_v3_3_failure_path_search(
        feature_path=tmp_path / "missing_features.parquet",
        instruments=pd.DataFrame(),
        config=None,  # fitness fn never reaches this in synthetic mode
        cfg=cfg,
        use_synthetic_fitness=True,
    )
    expected_keys = {
        "aco_failure_paths",
        "aco_pheromone_trails",
        "ga_pareto",
        "ga_best_chromosome",
        "sa_trace",
        "sa_cooldown_plateau",
        "candidate_notes",
    }
    assert expected_keys.issubset(outputs.keys())
    for key in expected_keys:
        assert outputs[key].exists()
    aco_paths = pd.read_csv(outputs["aco_failure_paths"])
    assert not aco_paths.empty
    ga_pareto = pd.read_csv(outputs["ga_pareto"])
    assert not ga_pareto.empty
    notes = outputs["candidate_notes"].read_text(encoding="utf-8")
    assert "v3.3 Failure-Path" in notes


def test_synthetic_path_fitness_rewards_terminal_paths():
    pool = pd.DataFrame({"net_return": [-0.02, -0.01]})
    fit = make_synthetic_path_fitness(pool)
    path_with_terminal = ("S1", "low_coimpulse", "symbol_risk_off")
    path_no_terminal = ("S1", "low_coimpulse", "btc_not_up")
    assert fit(path_with_terminal) > fit(path_no_terminal)


def test_synthetic_chromosome_fitness_rewards_motif_count_and_cooldown_proximity():
    pool = pd.DataFrame({"net_return": [-0.01]})
    cfg = V33Config()
    fit = make_synthetic_chromosome_fitness(pool, cfg)
    c_good = encode_chromosome(("S1", "S3", "S5"), "symbol", 48, True, True, False, False)
    c_bad = encode_chromosome(("S2",), "market", 96, False, False, False, False)
    g_score, _ = fit(c_good)
    b_score, _ = fit(c_bad)
    assert g_score > b_score

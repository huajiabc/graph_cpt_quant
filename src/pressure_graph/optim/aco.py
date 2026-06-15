"""Ant-Colony Optimization over a state-node graph for failure-path mining.

The instruction text behind v3.3:

    用 ACO / GA / SA 探索空头也可以, 但不要直接搜 short
    ACO 搜 failure path
    路径类似:
        CIC / impulse state -> failed reclaim -> low co-impulse -> symbol risk-off
        beta extreme -> volume shock exhaustion -> failed follow-through -> no-long
    目标不是只找 short, 而是找 failure path

The ants walk through *state predicates* in temporal order; pheromone is
reinforced when the path empirically precedes long-book failure. The output is
a ranked list of paths suitable as new motif_set candidates for the GA.

This module is pure: it never reads the trade cache directly. The caller
supplies a ``fitness_fn`` that takes a path and returns a scalar (higher =
stronger failure signal). The caller also owns the state graph definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class StateGraph:
    """Directed graph of state predicates with allowed transitions.

    ``nodes`` is an ordered tuple of node names. ``adjacency`` maps each node
    to the tuple of nodes it can transition to (subset of ``nodes``); empty
    tuple = terminal. Callers can freeze a domain-specific graph (e.g. the v3.3
    failure-path graph) and reuse it across optimization runs.
    """

    nodes: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    start_nodes: tuple[str, ...]
    terminal_nodes: tuple[str, ...]

    def successors(self, node: str) -> tuple[str, ...]:
        return self.adjacency.get(node, ())


@dataclass(frozen=True)
class FailurePath:
    """One ACO-evaluated path with its fitness and pheromone trail summary."""

    nodes: tuple[str, ...]
    fitness: float
    edge_visits: int = 0  # how often the path was sampled in the final iter.

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class ACOConfig:
    """ACO hyperparameters. Defaults are tame; tune via the orchestrator."""

    ants_per_iter: int = 24
    iterations: int = 24
    max_path_len: int = 5
    min_path_len: int = 2
    alpha: float = 1.0  # pheromone weight
    beta: float = 1.0  # heuristic weight
    evaporation: float = 0.15
    deposit_scale: float = 1.0
    initial_pheromone: float = 0.5
    elite_keep: int = 8  # top-K paths returned at the end
    rng_seed: int = 17
    pheromone_min: float = 0.05
    pheromone_max: float = 5.0


class AntColonyOptimizer:
    """Build failure paths through a ``StateGraph`` and reinforce by fitness.

    Usage::

        graph = StateGraph(...)
        aco = AntColonyOptimizer(graph, fitness_fn, cfg)
        paths = aco.run()

    ``fitness_fn`` receives a ``tuple[str, ...]`` path and returns a float.
    Higher = better. NaN / negative is fine — they are clipped to zero before
    pheromone deposit (so they don't poison the trail).
    """

    def __init__(
        self,
        graph: StateGraph,
        fitness_fn: Callable[[tuple[str, ...]], float],
        cfg: ACOConfig = ACOConfig(),
        heuristic_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.graph = graph
        self.fitness_fn = fitness_fn
        self.cfg = cfg
        self.heuristic_fn = heuristic_fn or (lambda a, b: 1.0)
        self.rng = np.random.default_rng(cfg.rng_seed)
        self.pheromone: dict[tuple[str, str], float] = {}
        for node in graph.nodes:
            for nxt in graph.successors(node):
                self.pheromone[(node, nxt)] = cfg.initial_pheromone
        # Self-loops are not allowed (paths must advance).

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def _choose_next(self, current: str) -> str | None:
        successors = [n for n in self.graph.successors(current) if n != current]
        if not successors:
            return None
        weights = np.array(
            [
                (self.pheromone.get((current, nxt), self.cfg.initial_pheromone) ** self.cfg.alpha)
                * (self.heuristic_fn(current, nxt) ** self.cfg.beta)
                for nxt in successors
            ],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            return str(self.rng.choice(successors))
        probabilities = weights / total
        return str(self.rng.choice(successors, p=probabilities))

    def _construct_path(self) -> tuple[str, ...]:
        start = str(self.rng.choice(self.graph.start_nodes))
        path: list[str] = [start]
        visited = {start}
        for _ in range(self.cfg.max_path_len - 1):
            nxt = self._choose_next(path[-1])
            if nxt is None or nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            if nxt in self.graph.terminal_nodes and len(path) >= self.cfg.min_path_len:
                break
        if len(path) < self.cfg.min_path_len:
            return tuple(path) if path else ()
        return tuple(path)

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------

    def _evaporate(self) -> None:
        for key in list(self.pheromone.keys()):
            new_value = self.pheromone[key] * (1.0 - self.cfg.evaporation)
            self.pheromone[key] = max(self.cfg.pheromone_min, new_value)

    def _deposit(self, path: tuple[str, ...], fitness: float) -> None:
        deposit = max(0.0, float(fitness)) * self.cfg.deposit_scale
        if deposit <= 0.0 or len(path) < 2:
            return
        for a, b in zip(path[:-1], path[1:]):
            key = (a, b)
            current = self.pheromone.get(key, self.cfg.initial_pheromone)
            self.pheromone[key] = min(self.cfg.pheromone_max, current + deposit)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[FailurePath]:
        """Run the ACO loop and return the top-``cfg.elite_keep`` failure paths."""
        scored: dict[tuple[str, ...], FailurePath] = {}
        for _ in range(self.cfg.iterations):
            iter_paths: list[tuple[tuple[str, ...], float]] = []
            for _ in range(self.cfg.ants_per_iter):
                path = self._construct_path()
                if not path:
                    continue
                try:
                    fitness = float(self.fitness_fn(path))
                except Exception:
                    fitness = float("nan")
                if not np.isfinite(fitness):
                    fitness = 0.0
                iter_paths.append((path, fitness))
                existing = scored.get(path)
                if existing is None or fitness > existing.fitness:
                    scored[path] = FailurePath(nodes=path, fitness=fitness)
            self._evaporate()
            for path, fitness in iter_paths:
                self._deposit(path, fitness)
        # Bump visit counts using the final pheromone as a proxy for "use".
        finalized = [
            FailurePath(
                nodes=fp.nodes,
                fitness=fp.fitness,
                edge_visits=sum(
                    1
                    for a, b in zip(fp.nodes[:-1], fp.nodes[1:])
                    if self.pheromone.get((a, b), 0.0) > self.cfg.initial_pheromone
                ),
            )
            for fp in scored.values()
        ]
        finalized.sort(key=lambda fp: (-fp.fitness, len(fp)))
        return finalized[: self.cfg.elite_keep]

    def pheromone_snapshot(self) -> list[tuple[str, str, float]]:
        """Return the current pheromone trail as a sorted (src, dst, weight) list."""
        return sorted(
            [(a, b, float(w)) for (a, b), w in self.pheromone.items()],
            key=lambda triple: -triple[2],
        )


__all__ = [
    "ACOConfig",
    "AntColonyOptimizer",
    "FailurePath",
    "StateGraph",
]

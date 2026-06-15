"""Genetic-algorithm search over the v3.3 action-combo chromosome.

Chromosome (instruction text):

    [motif_set, scope, cooldown, applies_to_core, applies_to_overflow,
     applies_to_existing_positions, applies_to_protect_a]

Fitness (instruction text):

    improve long book net20
    reduce drawdown
    minimize missed good trades
    avoid overblocking

This module exposes:

- ``Chromosome`` dataclass + ``encode_chromosome`` / ``decode_chromosome``
- ``GAConfig`` hyperparameters
- ``GeneticOptimizer`` — tournament-selection + uniform-crossover + per-gene
  mutation, all pure-function, deterministic given the seed.

Fitness is supplied by the caller (the v3.3 orchestrator wires it to the v12s3
stack simulator). Nothing here knows about parquet files or the trade cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Gene domains — frozen at module level so encode/decode round-trip cleanly.
MOTIF_UNIVERSE: tuple[str, ...] = ("S1", "S2", "S3", "S5")
SCOPE_VALUES: tuple[str, ...] = ("symbol", "cluster", "market")
COOLDOWN_VALUES: tuple[int, ...] = (16, 24, 32, 40, 48, 56, 64, 96)


@dataclass(frozen=True)
class Chromosome:
    """A v3.3 action-combo individual.

    All fields are domain-constrained; ``encode`` ensures bool flags are bool
    and the index fields point into the canonical domain tuples.
    """

    motif_set: tuple[str, ...]
    scope: str
    cooldown_bars: int
    apply_core: bool
    apply_overflow: bool
    apply_existing_positions: bool
    apply_protect_a: bool

    def as_dict(self) -> dict:
        return {
            "motif_set": list(self.motif_set),
            "scope": self.scope,
            "cooldown_bars": int(self.cooldown_bars),
            "apply_core": bool(self.apply_core),
            "apply_overflow": bool(self.apply_overflow),
            "apply_existing_positions": bool(self.apply_existing_positions),
            "apply_protect_a": bool(self.apply_protect_a),
        }


def encode_chromosome(
    motif_set: tuple[str, ...],
    scope: str,
    cooldown_bars: int,
    apply_core: bool,
    apply_overflow: bool,
    apply_existing_positions: bool,
    apply_protect_a: bool,
) -> Chromosome:
    """Construct a Chromosome with domain validation."""
    motifs = tuple(sorted(set(m for m in motif_set if m in MOTIF_UNIVERSE)))
    if scope not in SCOPE_VALUES:
        raise ValueError(f"scope {scope} not in {SCOPE_VALUES}")
    if cooldown_bars not in COOLDOWN_VALUES:
        # Snap to the nearest domain value rather than rejecting — friendly to SA.
        cooldown_bars = min(COOLDOWN_VALUES, key=lambda v: abs(v - cooldown_bars))
    return Chromosome(
        motif_set=motifs,
        scope=str(scope),
        cooldown_bars=int(cooldown_bars),
        apply_core=bool(apply_core),
        apply_overflow=bool(apply_overflow),
        apply_existing_positions=bool(apply_existing_positions),
        apply_protect_a=bool(apply_protect_a),
    )


def decode_chromosome(payload: dict) -> Chromosome:
    return encode_chromosome(
        motif_set=tuple(payload.get("motif_set", ())),
        scope=str(payload.get("scope", "symbol")),
        cooldown_bars=int(payload.get("cooldown_bars", 48)),
        apply_core=bool(payload.get("apply_core", True)),
        apply_overflow=bool(payload.get("apply_overflow", True)),
        apply_existing_positions=bool(payload.get("apply_existing_positions", False)),
        apply_protect_a=bool(payload.get("apply_protect_a", False)),
    )


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 24
    generations: int = 16
    tournament_k: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.12
    elite_keep: int = 2
    rng_seed: int = 17
    # Fitness weights (instruction text translated into a scalar objective):
    weight_net20: float = 1.0
    weight_drawdown_reduction: float = 1.0
    weight_missed_good: float = 0.5
    weight_overblock: float = 0.3
    min_motifs: int = 1


@dataclass
class Individual:
    chromosome: Chromosome
    fitness: float = float("nan")
    detail: dict = field(default_factory=dict)


class GeneticOptimizer:
    """Tournament-selection GA, deterministic by seed.

    The caller supplies a ``fitness_fn(chromosome) -> (score, detail_dict)``.
    The optimizer treats ``score`` opaquely — combine sub-objectives upstream
    or use ``GAConfig`` weights and feed a dict to ``detail``.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Chromosome], tuple[float, dict]],
        cfg: GAConfig = GAConfig(),
        seeded_chromosomes: tuple[Chromosome, ...] = (),
    ) -> None:
        self.fitness_fn = fitness_fn
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rng_seed)
        self.seeded_chromosomes = tuple(seeded_chromosomes)

    # ------------------------------------------------------------------
    # Random individuals + initial population
    # ------------------------------------------------------------------

    def _random_chromosome(self) -> Chromosome:
        # Random non-empty motif subset (size in [min_motifs, |universe|]).
        size = int(self.rng.integers(self.cfg.min_motifs, len(MOTIF_UNIVERSE) + 1))
        motifs = tuple(
            sorted(self.rng.choice(MOTIF_UNIVERSE, size=size, replace=False).tolist())
        )
        return encode_chromosome(
            motif_set=motifs,
            scope=str(self.rng.choice(SCOPE_VALUES)),
            cooldown_bars=int(self.rng.choice(COOLDOWN_VALUES)),
            apply_core=bool(self.rng.random() < 0.7),
            apply_overflow=bool(self.rng.random() < 0.6),
            apply_existing_positions=bool(self.rng.random() < 0.3),
            apply_protect_a=bool(self.rng.random() < 0.2),
        )

    def _initial_population(self) -> list[Individual]:
        pop: list[Individual] = [Individual(chromosome=c) for c in self.seeded_chromosomes]
        while len(pop) < self.cfg.population_size:
            pop.append(Individual(chromosome=self._random_chromosome()))
        return pop[: self.cfg.population_size]

    # ------------------------------------------------------------------
    # Variation operators
    # ------------------------------------------------------------------

    def _crossover(self, a: Chromosome, b: Chromosome) -> Chromosome:
        """Uniform crossover; for motif_set, sample from the union."""
        choose_a = self.rng.random() < 0.5
        scope = a.scope if choose_a else b.scope
        cooldown = a.cooldown_bars if self.rng.random() < 0.5 else b.cooldown_bars
        union = sorted(set(a.motif_set) | set(b.motif_set))
        if not union:
            union = list(MOTIF_UNIVERSE[:1])
        size = int(self.rng.integers(self.cfg.min_motifs, len(union) + 1))
        motifs = tuple(sorted(self.rng.choice(union, size=size, replace=False).tolist()))
        return encode_chromosome(
            motif_set=motifs,
            scope=scope,
            cooldown_bars=cooldown,
            apply_core=a.apply_core if self.rng.random() < 0.5 else b.apply_core,
            apply_overflow=a.apply_overflow if self.rng.random() < 0.5 else b.apply_overflow,
            apply_existing_positions=(
                a.apply_existing_positions if self.rng.random() < 0.5 else b.apply_existing_positions
            ),
            apply_protect_a=a.apply_protect_a if self.rng.random() < 0.5 else b.apply_protect_a,
        )

    def _mutate(self, c: Chromosome) -> Chromosome:
        motifs = list(c.motif_set)
        scope = c.scope
        cooldown = c.cooldown_bars
        flags = [c.apply_core, c.apply_overflow, c.apply_existing_positions, c.apply_protect_a]
        rate = self.cfg.mutation_rate
        # motif_set: flip one motif in or out.
        if self.rng.random() < rate:
            candidate = str(self.rng.choice(MOTIF_UNIVERSE))
            if candidate in motifs and len(motifs) > self.cfg.min_motifs:
                motifs = [m for m in motifs if m != candidate]
            elif candidate not in motifs:
                motifs.append(candidate)
        # scope: rotate.
        if self.rng.random() < rate:
            scope = str(self.rng.choice(SCOPE_VALUES))
        # cooldown: step ±1 in the domain index, or pick a fresh value.
        if self.rng.random() < rate:
            idx = COOLDOWN_VALUES.index(cooldown)
            shift = int(self.rng.choice([-1, 1]))
            cooldown = COOLDOWN_VALUES[max(0, min(len(COOLDOWN_VALUES) - 1, idx + shift))]
        # flags: independent bit flips.
        for i in range(len(flags)):
            if self.rng.random() < rate:
                flags[i] = not flags[i]
        return encode_chromosome(
            motif_set=tuple(motifs),
            scope=scope,
            cooldown_bars=cooldown,
            apply_core=flags[0],
            apply_overflow=flags[1],
            apply_existing_positions=flags[2],
            apply_protect_a=flags[3],
        )

    def _tournament(self, pop: list[Individual]) -> Individual:
        contenders = self.rng.choice(len(pop), size=self.cfg.tournament_k, replace=False)
        best = max(pop[i] for i in contenders.tolist()) if False else None
        # numpy's choice returns int64; iterate manually for clear typing.
        best_ind = pop[int(contenders[0])]
        for j in contenders[1:].tolist():
            if pop[int(j)].fitness > best_ind.fitness:
                best_ind = pop[int(j)]
        return best_ind

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, individual: Individual) -> Individual:
        try:
            score, detail = self.fitness_fn(individual.chromosome)
        except Exception as exc:  # fitness should not crash the search
            score = float("nan")
            detail = {"error": repr(exc)}
        score_f = float(score) if np.isfinite(score) else float("-inf")
        return Individual(chromosome=individual.chromosome, fitness=score_f, detail=detail)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[Individual]:
        pop = [self._evaluate(ind) for ind in self._initial_population()]
        history: list[Individual] = list(pop)
        for _ in range(self.cfg.generations):
            pop.sort(key=lambda ind: -ind.fitness)
            elites = pop[: self.cfg.elite_keep]
            children: list[Individual] = list(elites)
            while len(children) < self.cfg.population_size:
                parent_a = self._tournament(pop).chromosome
                parent_b = self._tournament(pop).chromosome
                child_chromosome = (
                    self._crossover(parent_a, parent_b)
                    if self.rng.random() < self.cfg.crossover_rate
                    else parent_a
                )
                child_chromosome = self._mutate(child_chromosome)
                children.append(self._evaluate(Individual(chromosome=child_chromosome)))
            pop = children
            history.extend(pop)
        # Deduplicate by chromosome — keep the best-scoring instance.
        best_by_key: dict[tuple, Individual] = {}
        for ind in history:
            key = (
                ind.chromosome.motif_set,
                ind.chromosome.scope,
                ind.chromosome.cooldown_bars,
                ind.chromosome.apply_core,
                ind.chromosome.apply_overflow,
                ind.chromosome.apply_existing_positions,
                ind.chromosome.apply_protect_a,
            )
            existing = best_by_key.get(key)
            if existing is None or ind.fitness > existing.fitness:
                best_by_key[key] = ind
        finalized = sorted(best_by_key.values(), key=lambda ind: -ind.fitness)
        return finalized


__all__ = [
    "COOLDOWN_VALUES",
    "Chromosome",
    "GAConfig",
    "GeneticOptimizer",
    "Individual",
    "MOTIF_UNIVERSE",
    "SCOPE_VALUES",
    "decode_chromosome",
    "encode_chromosome",
]

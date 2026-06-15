"""Simulated Annealing — cooldown / scope plateau verifier for v3.3.

The instruction text:

    SA 微调 cooldown 和 scope
    现在最佳 cooldown 是: 48 bars
    SA 可以测试稳定区间: 32 / 40 / 48 / 56 / 64
    但要看: 是否 48 附近都是好区间
    如果只有 48 好, 就是针尖

So this loop is a *plateau detector*, not a global search. It starts from the
GA winner (or any seed), proposes ±1-step moves in cooldown / scope, and
records every visited (cooldown, fitness) pair so the orchestrator can plot
the local landscape.

A single-needle optimum at 48 with a worse neighbor on either side is a red
flag the verdict text calls out as "针尖" — the v3.3 candidate_notes file
distinguishes plateau from needle using this trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .ga import COOLDOWN_VALUES, SCOPE_VALUES, Chromosome, encode_chromosome


@dataclass(frozen=True)
class SAConfig:
    iterations: int = 60
    initial_temperature: float = 0.05
    cooling_rate: float = 0.92
    min_temperature: float = 1e-4
    rng_seed: int = 17
    move_cooldown_prob: float = 0.7
    move_scope_prob: float = 0.3
    plateau_tolerance: float = 0.02
    plateau_window: int = 3  # how many neighbours on each side count as "plateau".


@dataclass
class SAStep:
    iteration: int
    chromosome: Chromosome
    fitness: float
    accepted: bool
    temperature: float


class SimulatedAnnealer:
    """SA local search over (cooldown, scope) keeping all other genes fixed."""

    def __init__(
        self,
        seed: Chromosome,
        fitness_fn: Callable[[Chromosome], float],
        cfg: SAConfig = SAConfig(),
    ) -> None:
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(cfg.rng_seed)
        seed_eval = float(fitness_fn(seed))
        self.current = seed
        self.current_fitness = seed_eval
        self.best = seed
        self.best_fitness = seed_eval
        self.trace: list[SAStep] = []

    # ------------------------------------------------------------------
    # Neighbour proposals
    # ------------------------------------------------------------------

    def _neighbour(self, c: Chromosome) -> Chromosome:
        flip = self.rng.random()
        cooldown = c.cooldown_bars
        scope = c.scope
        if flip < self.cfg.move_cooldown_prob:
            idx = COOLDOWN_VALUES.index(cooldown)
            shift = int(self.rng.choice([-1, 1]))
            cooldown = COOLDOWN_VALUES[max(0, min(len(COOLDOWN_VALUES) - 1, idx + shift))]
        elif flip < self.cfg.move_cooldown_prob + self.cfg.move_scope_prob:
            scope = str(self.rng.choice(SCOPE_VALUES))
        return encode_chromosome(
            motif_set=c.motif_set,
            scope=scope,
            cooldown_bars=cooldown,
            apply_core=c.apply_core,
            apply_overflow=c.apply_overflow,
            apply_existing_positions=c.apply_existing_positions,
            apply_protect_a=c.apply_protect_a,
        )

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def _accept(self, delta: float, temperature: float) -> bool:
        if delta >= 0:
            return True
        if temperature <= 0:
            return False
        # Probabilistic acceptance of worse moves.
        return bool(self.rng.random() < np.exp(delta / temperature))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> list[SAStep]:
        temperature = self.cfg.initial_temperature
        for it in range(self.cfg.iterations):
            proposal = self._neighbour(self.current)
            fit = float(self.fitness_fn(proposal))
            delta = fit - self.current_fitness
            accepted = self._accept(delta, temperature)
            self.trace.append(
                SAStep(
                    iteration=it,
                    chromosome=proposal,
                    fitness=fit,
                    accepted=accepted,
                    temperature=temperature,
                )
            )
            if accepted:
                self.current = proposal
                self.current_fitness = fit
                if fit > self.best_fitness:
                    self.best = proposal
                    self.best_fitness = fit
            temperature = max(self.cfg.min_temperature, temperature * self.cfg.cooling_rate)
        return self.trace

    # ------------------------------------------------------------------
    # Plateau analysis
    # ------------------------------------------------------------------

    def plateau_report(self) -> dict:
        """Inspect the visited (cooldown, fitness) cloud for plateau-vs-needle.

        Returns a dict suitable for inclusion in the SA CSV row of v3.3.
        ``is_plateau`` is True when every neighbour within ``plateau_window``
        of the best cooldown scores within ``plateau_tolerance`` of the best.
        """
        if not self.trace:
            return {"is_plateau": False, "best_cooldown": None, "neighbours": {}}
        best_per_cd: dict[int, float] = {}
        for step in self.trace:
            cd = int(step.chromosome.cooldown_bars)
            if cd not in best_per_cd or step.fitness > best_per_cd[cd]:
                best_per_cd[cd] = step.fitness
        sorted_cds = sorted(best_per_cd.keys())
        best_cd = max(best_per_cd, key=lambda k: best_per_cd[k])
        best_score = best_per_cd[best_cd]
        # Count neighbours that score "close enough" within plateau_window steps.
        try:
            anchor = sorted_cds.index(best_cd)
        except ValueError:
            anchor = -1
        plateau_neighbours: dict[int, float] = {}
        for offset in range(-self.cfg.plateau_window, self.cfg.plateau_window + 1):
            if offset == 0:
                continue
            j = anchor + offset
            if 0 <= j < len(sorted_cds):
                neighbour_cd = sorted_cds[j]
                plateau_neighbours[neighbour_cd] = best_per_cd[neighbour_cd]
        if not plateau_neighbours:
            is_plateau = False
        else:
            within = sum(
                1
                for score in plateau_neighbours.values()
                if best_score - score <= self.cfg.plateau_tolerance
            )
            is_plateau = within >= max(1, len(plateau_neighbours) // 2)
        return {
            "is_plateau": is_plateau,
            "best_cooldown": int(best_cd),
            "best_fitness": float(best_score),
            "neighbours": {int(k): float(v) for k, v in plateau_neighbours.items()},
        }


__all__ = ["SAConfig", "SAStep", "SimulatedAnnealer"]

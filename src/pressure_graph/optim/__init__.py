"""Metaheuristic search primitives for v3.3 failure-path / action-combo tuning.

Three independent search loops compose into the v3.3 pipeline:

- ``aco``: ant-colony search over a state-node graph; the goal is *failure
  paths*, not short signals. A path is a temporal sequence of state predicates
  (CIC, failed_reclaim, low_coimpulse, …) whose pheromone is reinforced when
  suppressing longs after the pattern improves the long book.
- ``ga``: genetic search over the v12s3 risk-off chromosome
  ``[motif_set, scope, cooldown, apply-flags]``. Fitness blends long net20,
  drawdown reduction, missed-good-long penalty, and overblock penalty.
- ``sa``: simulated annealing around the GA winner — confirm cooldown sits in
  a plateau (32/40/48/56/64), not a single-needle optimum.

These are pure-function modules. The v12s3 stack simulator is the eval call
plugged into each loop; nothing here knows about the trade cache directly.
"""
from __future__ import annotations

from .aco import ACOConfig, AntColonyOptimizer, FailurePath, StateGraph
from .ga import Chromosome, GAConfig, GeneticOptimizer, decode_chromosome, encode_chromosome
from .sa import SAConfig, SimulatedAnnealer

__all__ = [
    "ACOConfig",
    "AntColonyOptimizer",
    "Chromosome",
    "FailurePath",
    "GAConfig",
    "GeneticOptimizer",
    "SAConfig",
    "SimulatedAnnealer",
    "StateGraph",
    "decode_chromosome",
    "encode_chromosome",
]

from __future__ import annotations

import numpy as np

from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v157_pair_shock_fragile_receiver import (
    V157Config,
    beta_neutral_v157_weights,
    select_v157_side,
)


def test_select_v157_side_retains_inside_top_four() -> None:
    candidates = [(f"S{index}", float(10 - index)) for index in range(6)]
    selected = select_v157_side(candidates, {"S2", "S3"})
    assert selected == ["S2", "S3"]


def test_beta_neutral_v157_weights_are_exact() -> None:
    longs = ["A", "B"]
    shorts = ["C", "D"]
    betas = {"A": 1.0, "B": 1.2, "C": 0.7, "D": 0.8}
    weights = beta_neutral_v157_weights(longs, shorts, betas)
    residual = sum(weights[symbol] * betas[symbol] for symbol in longs + shorts)
    residual += weights[BTC]
    assert np.isclose(sum(abs(weight) for weight in weights.values()), 1.0)
    assert abs(residual) < 1e-12


def test_v157_frozen_counts() -> None:
    cfg = V157Config()
    assert cfg.side_count == 2
    assert cfg.retention_count == 4
    assert cfg.random_iterations == 1000

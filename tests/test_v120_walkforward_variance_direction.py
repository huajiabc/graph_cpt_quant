from __future__ import annotations

import numpy as np

from pressure_graph.reports.v120_walkforward_variance_direction import (
    _directional_payoff,
    _ridge_predict,
)


def test_directional_cost_penalizes_both_sides() -> None:
    short_gross = _directional_payoff(60.0, 40.0, 1.0, 0.0)
    short_net = _directional_payoff(60.0, 40.0, 1.0, 1.0)
    long_gross = _directional_payoff(40.0, 60.0, -1.0, 0.0)
    long_net = _directional_payoff(40.0, 60.0, -1.0, 1.0)
    assert short_net < short_gross
    assert long_net < long_gross


def test_ridge_predict_learns_positive_relation() -> None:
    train_x = np.arange(1.0, 11.0).reshape(-1, 1)
    train_y = train_x[:, 0] * 2.0
    prediction, coefficients = _ridge_predict(
        train_x, train_y, np.array([12.0]), alpha=0.01
    )
    assert prediction > train_y.max()
    assert coefficients[0] > 0

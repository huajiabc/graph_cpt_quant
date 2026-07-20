from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v125_cross_venue_perpetual_carry import (
    _components,
    _top_positive_weights,
)


def test_positive_cross_venue_spread_is_selected() -> None:
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDE"),
            "score_7d": [-0.2, 0.1, 0.2, 0.3, 0.4],
            "pair_gross_return": np.zeros(5),
        }
    )
    weights, selected = _top_positive_weights(frame, "score_7d", 3)
    assert selected == ["C", "D", "E"]
    assert np.isclose(sum(weights.values()), 1.0)


def test_pair_return_is_bybit_minus_binance_plus_funding_spread() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "bybit_return": [0.10, -0.02],
            "binance_return": [0.08, -0.03],
            "price_basis_return": [0.02, 0.01],
            "funding_spread_return": [0.004, 0.002],
            "pair_gross_return": [0.024, 0.012],
        }
    )
    result = _components(frame, {"A": 0.5, "B": 0.5})
    assert np.isclose(result["price_basis_return"], 0.015)
    assert np.isclose(result["funding_spread_return"], 0.003)
    assert np.isclose(result["gross_return"], 0.018)

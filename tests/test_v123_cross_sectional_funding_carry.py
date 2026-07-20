from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v123_cross_sectional_funding_carry import (
    _direct_weights,
    _portfolio_components,
)


def test_low_funding_is_long_and_high_funding_is_short() -> None:
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDEF"),
            "score_7d": np.arange(6, dtype=float),
            "carry_adjusted_return": np.zeros(6),
            "residual_carry_adjusted_return": np.zeros(6),
        }
    )
    weights, long_symbols, short_symbols = _direct_weights(frame, "score_7d", 2)
    assert long_symbols == ["A", "B"]
    assert short_symbols == ["E", "F"]
    assert np.isclose(sum(weights.values()), 0.0)


def test_positive_funding_pays_short_position() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["LONG", "SHORT"],
            "price_return": [0.0, 0.0],
            "future_funding": [-0.001, 0.001],
            "residual_price_return": [0.0, 0.0],
        }
    )
    components = _portfolio_components(frame, {"LONG": 0.5, "SHORT": -0.5})
    assert np.isclose(components["funding_return"], 0.001)
    assert np.isclose(components["gross_return"], 0.001)

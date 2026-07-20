from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v140_equal_weight_negative_funding_state import (
    V140Config,
    _equal_negative_distribution,
)


def test_equal_negative_distribution_uses_every_negative_name() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "score_7d": [-0.1, -0.2, -0.3, -0.4, 0.1],
            "price_return": [0.0] * 5,
            "btc_beta": [1.0] * 5,
        }
    )
    distribution = _equal_negative_distribution(frame, V140Config())
    assert set(distribution) == {"A", "B", "C", "D"}
    assert np.allclose(list(distribution.values()), 0.25)

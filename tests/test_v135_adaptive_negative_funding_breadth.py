from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v135_adaptive_negative_funding_breadth import (
    V135Config,
    _select_adaptive_hold_band,
)


def test_adaptive_breadth_uses_all_six_negative_names() -> None:
    frame = pd.DataFrame(
        {
            "symbol": [f"S{index:02d}" for index in range(12)],
            "score_7d": [-0.01] * 6 + [0.01] * 6,
            "price_return": np.zeros(12),
            "btc_beta": np.ones(12),
        }
    )
    selected = _select_adaptive_hold_band(frame, [], V135Config())
    assert selected == [f"S{index:02d}" for index in range(6)]


def test_adaptive_breadth_refuses_three_name_portfolio() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "score_7d": [-0.3, -0.2, -0.1],
            "price_return": [0.0] * 3,
            "btc_beta": [1.0] * 3,
        }
    )
    assert _select_adaptive_hold_band(frame, [], V135Config()) == []

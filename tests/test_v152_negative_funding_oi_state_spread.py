from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v152_negative_funding_oi_state_spread import (
    V152Config,
    oi_state_target,
)


def _local() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "score_7d": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "oi_change_7d": [0.5, 0.2, 0.0, -0.1, -0.4],
            "price_return": [0.01, 0.02, 0.0, -0.01, -0.02],
            "future_funding": [-0.002] * 5,
            "btc_beta": [1.0] * 5,
            "btc_return": [0.01] * 5,
            "btc_future_funding": [0.0002] * 5,
        }
    )


def test_v152_uses_equal_extreme_halves_and_omits_middle() -> None:
    weights, _, longs, shorts, breadth = oi_state_target(_local(), V152Config())
    assert breadth == 5
    assert longs == ["A", "B"]
    assert shorts == ["D", "E"]
    assert "C" not in weights


def test_v152_target_is_gross_one_and_beta_neutral() -> None:
    weights, components, _, _, _ = oi_state_target(_local(), V152Config())
    assert np.isclose(sum(abs(weight) for weight in weights.values()), 1.0)
    assert np.isclose(components["residual_btc_beta"], 0.0, atol=1e-12)

from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v124_delta_neutral_spot_perp_carry import (
    _components,
    _top_positive_weights,
)


def test_top_funding_selects_only_positive_scores() -> None:
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDEF"),
            "score_7d": [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            "pair_gross_return": np.zeros(6),
        }
    )
    weights, selected = _top_positive_weights(frame, "score_7d", 3)
    assert selected == ["D", "E", "F"]
    assert np.isclose(sum(weights.values()), 1.0)
    missing, _ = _top_positive_weights(frame, "score_7d", 5)
    assert missing == {}


def test_pair_return_is_spot_minus_perp_plus_short_funding() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "spot_return": [0.10, -0.02],
            "perp_return": [0.08, -0.03],
            "basis_return": [0.02, 0.01],
            "future_funding": [0.004, 0.002],
            "pair_gross_return": [0.024, 0.012],
        }
    )
    result = _components(frame, {"A": 0.5, "B": 0.5})
    assert np.isclose(result["basis_return"], 0.015)
    assert np.isclose(result["funding_return"], 0.003)
    assert np.isclose(result["gross_return"], 0.018)

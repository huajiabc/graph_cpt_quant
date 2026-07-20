from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v136_severity_weighted_negative_funding import (
    V136Config,
    _severity_distribution,
    _weighted_components,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "score_7d": [-0.01, -0.02, -0.03, -0.20, 0.01],
            "price_return": [0.01] * 5,
            "future_funding": [-0.002] * 5,
            "btc_beta": [1.5] * 5,
            "btc_return": [0.004] * 5,
            "btc_future_funding": [0.0007] * 5,
        }
    )


def test_severity_distribution_caps_extreme_and_sums_to_one() -> None:
    distribution = _severity_distribution(_frame(), V136Config())
    assert set(distribution) == {"A", "B", "C", "D"}
    assert np.isclose(sum(distribution.values()), 1.0)
    assert distribution["D"] < 0.7


def test_weighted_portfolio_has_unit_gross_and_zero_beta() -> None:
    distribution = _severity_distribution(_frame(), V136Config())
    weights, components = _weighted_components(_frame(), distribution)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert np.isclose(components["residual_btc_beta"], 0.0)
    assert components["funding_return"] > 0

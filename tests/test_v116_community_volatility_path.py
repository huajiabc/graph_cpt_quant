from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v116_community_volatility_path import (
    _bucket_path_features,
)


def test_path_efficiency_distinguishes_trend_from_chop() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    residual = pd.DataFrame(
        {
            "TREND_A": [0.01] * 8,
            "TREND_B": [0.02] * 8,
            "CHOP_A": [0.01, -0.01] * 4,
            "CHOP_B": [0.02, -0.02] * 4,
        },
        index=index,
    )

    trend = _bucket_path_features(residual, ["TREND_A", "TREND_B"])
    chop = _bucket_path_features(residual, ["CHOP_A", "CHOP_B"])

    assert np.isclose(trend.iloc[-1]["path_efficiency_1h"], 1.0)
    assert np.isclose(chop.iloc[-1]["path_efficiency_1h"], 0.0)
    assert trend.iloc[-1]["direction_breadth_1h"] == 1.0
    assert np.isnan(chop.iloc[-1]["direction_breadth_1h"])


def test_path_direction_and_breadth_use_known_one_hour_returns() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    residual = pd.DataFrame(
        {
            "A": [-0.01, -0.01, -0.01, -0.01],
            "B": [-0.02, -0.02, -0.02, -0.02],
            "C": [0.005, 0.005, 0.005, 0.005],
        },
        index=index,
    )

    features = _bucket_path_features(residual, ["A", "B", "C"]).iloc[-1]

    assert features["direction"] == -1.0
    assert np.isclose(features["direction_breadth_1h"], 2.0 / 3.0)

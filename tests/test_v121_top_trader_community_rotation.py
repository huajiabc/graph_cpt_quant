from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v121_top_trader_community_rotation import (
    V121Config,
    _bucket_weights,
    causal_rolling_zscore,
    freeze_v121_directions_and_costs,
)


def test_causal_zscore_does_not_use_current_observation() -> None:
    ordinary = pd.Series([1.0, 2.0, 3.0, 4.0])
    shocked = pd.Series([1.0, 2.0, 3.0, 400.0])
    left = causal_rolling_zscore(ordinary, window=3, minimum_history=3)
    right = causal_rolling_zscore(shocked, window=3, minimum_history=3)
    assert np.isclose(left.iloc[-1], (4.0 - 2.0) / 1.0)
    assert np.isclose(right.iloc[-1], 5.0)


def test_bucket_weights_are_market_neutral() -> None:
    frame = pd.DataFrame(
        {
            "symbol": list("ABCDEF"),
            "future_ret_4h": np.arange(6, dtype=float),
        }
    )
    weights, high, low = _bucket_weights(
        frame, pd.Series(np.arange(6, dtype=float)), bucket_size=2
    )
    assert high == ["E", "F"]
    assert low == ["A", "B"]
    assert np.isclose(sum(weights.values()), 0.0)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)


def test_direction_is_frozen_from_development_only() -> None:
    times = pd.to_datetime(
        ["2025-12-01", "2026-02-01", "2026-05-01"], utc=True
    )
    decisions = pd.DataFrame(
        {
            "candidate": ["X"] * 3,
            "feature_time": times,
            "period": ["development", "validation", "holdout"],
            "continuation_raw_return": [-0.01, 0.02, 0.03],
            "continuation_residual_return": [-0.01, 0.02, 0.03],
            "spearman_ic": [-0.1, 0.2, 0.3],
            "_weights": [{"A": 0.5, "B": -0.5}] * 3,
        }
    )
    output, directions = freeze_v121_directions_and_costs(
        decisions,
        V121Config(focal_cost=0.004, stress_cost=0.006, one_way_cost=0.002),
    )
    assert directions.loc[0, "chosen_direction"] == "reversal"
    assert output["raw_return"].tolist() == [0.01, -0.02, -0.03]

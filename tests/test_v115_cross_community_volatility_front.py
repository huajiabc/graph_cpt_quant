from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v115_cross_community_volatility_front import (
    V115Config,
    _community_density,
    build_v115_front_state,
)


def test_community_density_counts_symbol_shock_breadth() -> None:
    shocks = pd.DataFrame(
        {"A": [2.0], "B": [0.0], "C": [3.0], "D": [4.0]}
    )
    thresholds = pd.Series({"A": 1.0, "B": 1.0, "C": 2.0, "D": 5.0})
    communities = {"C1": ["A", "B"], "C2": ["C", "D"]}

    density = _community_density(shocks, thresholds, communities)

    assert density.at[0, "C1"] == 0.5
    assert density.at[0, "C2"] == 0.5


def test_front_event_requires_btc_not_already_in_lower_tail() -> None:
    history_index = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    target_index = pd.date_range("2026-02-01", periods=2, freq="1h", tz="UTC")
    history = pd.DataFrame(
        {
            "A": [0.0] * 18 + [3.0, 3.0],
            "B": [0.0] * 18 + [3.0, 3.0],
            "C": [0.0] * 18 + [3.0, 3.0],
            "D": [0.0] * 18 + [3.0, 3.0],
            "E": [0.0] * 18 + [3.0, 3.0],
            "F": [0.0] * 18 + [3.0, 3.0],
        },
        index=history_index,
    )
    target = pd.DataFrame(4.0, index=target_index, columns=history.columns)
    history_btc = pd.Series(
        [-0.02, -0.01, 0.0, 0.01] * 5, index=history_index
    )
    target_btc = pd.Series([-0.03, 0.0], index=target_index)
    communities = {"C1": ["A", "B"], "C2": ["C", "D"], "C3": ["E", "F"]}
    cfg = V115Config(
        shock_quantile=0.8,
        community_density_quantile=0.8,
        front_breadth_quantile=0.8,
        min_active_communities=3,
        btc_lag_quantile=0.25,
    )

    state = build_v115_front_state(
        history, target, history_btc, target_btc, communities, cfg
    )

    assert bool(state.iloc[0]["event"]) is False
    assert bool(state.iloc[1]["event"]) is True

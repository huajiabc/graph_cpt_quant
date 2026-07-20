from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v118_crowding_unwind_transmission import (
    CANDIDATES,
    V118Config,
    build_v118_portfolios,
    rolling_v118_zscore,
)


def test_rolling_zscore_uses_only_prior_values() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    zscore = rolling_v118_zscore(values, window=3, minimum_history=3)
    expected = (100.0 - 2.0) / 1.0
    assert np.isnan(zscore.iloc[2])
    assert zscore.iloc[3] == 5.0
    assert expected > zscore.iloc[3]


def test_crowded_long_unwind_shorts_other_community_members() -> None:
    timestamp = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    month = pd.Timestamp("2026-01-01", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    crowding = pd.DataFrame([[3.0, 0.0, 0.0, 0.0]], index=[timestamp], columns=symbols)
    return_z = pd.DataFrame([[-2.0, 0.0, 0.0, 0.0]], index=[timestamp], columns=symbols)
    oi_z = pd.DataFrame([[-2.0, 0.0, 0.0, 0.0]], index=[timestamp], columns=symbols)
    future = pd.DataFrame([[0.0, -0.01, -0.02, -0.03]], index=[timestamp], columns=symbols)
    context = {
        "month_start": month,
        "period": "validation",
        "target_times": pd.DatetimeIndex([timestamp]),
        "communities": {"C1": symbols},
        "crowding_z": crowding,
        "return_z": return_z,
        "oi_z": oi_z,
        "raw_future_4h": future,
        "residual_future_4h": future,
    }
    events = build_v118_portfolios(
        {month: context}, V118Config(minimum_followers=3)
    )
    event = events[events["candidate"].eq(CANDIDATES[0])].iloc[0]
    assert event["leader_symbols"] == "A"
    assert event["follower_symbols"] == "B|C|D"
    assert event["raw_gross"] == 0.02

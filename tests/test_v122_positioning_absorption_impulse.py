from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v122_positioning_absorption_impulse import (
    V122Config,
    _exact_lag,
    apply_v122_costs,
)


def test_exact_lag_does_not_bridge_missing_hour() -> None:
    times = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 04:00", "2026-01-01 09:00"], utc=True
    )
    lagged = _exact_lag(pd.Series([1.0, 2.0, 3.0]), pd.Series(times), 4)
    assert np.isnan(lagged[0])
    assert lagged[1] == 1.0
    assert np.isnan(lagged[2])


def test_nonoverlapping_turnover_uses_candidate_horizon() -> None:
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 12:00"], utc=True)
    decisions = pd.DataFrame(
        {
            "candidate": ["X", "X"],
            "feature_time": times,
            "horizon_hours": [12, 12],
            "raw_return": [0.01, 0.02],
            "residual_return": [0.01, 0.02],
            "_weights": [
                {"A": 0.5, "B": -0.5},
                {"A": 0.5, "B": -0.5},
            ],
        }
    )
    output = apply_v122_costs(decisions, V122Config())
    assert output["realized_turnover"].tolist() == [1.0, 1.0]
    assert np.isclose(output.loc[0, "turnover_net_20bp_oneway"], 0.008)

import numpy as np
import pandas as pd

from pressure_graph.reports.v111_sparse_topology_continuation import (
    V111Config,
    forward_simple_returns,
    select_sparse_events,
)


def test_v111_forward_return_uses_only_t_plus_one_through_horizon() -> None:
    frame = pd.DataFrame({"A": [0.10, 0.20, -0.10, 0.05]})
    result = forward_simple_returns(frame, 2)
    assert np.isclose(result.loc[0, "A"], (1.20 * 0.90) - 1.0)
    assert np.isclose(result.loc[1, "A"], (0.90 * 1.05) - 1.0)
    assert np.isnan(result.loc[2, "A"])


def test_v111_sparse_threshold_uses_strictly_prior_months() -> None:
    events = pd.DataFrame(
        {
            "month_start": pd.to_datetime(
                ["2026-01-01"] * 4 + ["2026-02-01"] * 2, utc=True
            ),
            "break_severity": [1.0, 2.0, 3.0, 4.0, 100.0, 2.6],
        }
    )
    cfg = V111Config(min_prior_events=4, severity_quantile=0.5)
    selected = select_sparse_events(events, cfg)
    assert selected["month_start"].dt.month.unique().tolist() == [2]
    assert selected["break_severity"].tolist() == [100.0, 2.6]
    assert selected["severity_threshold"].unique().tolist() == [2.5]

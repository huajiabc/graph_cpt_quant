from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v171_option_skew_innovation_btc import (
    add_causal_innovation_zscore,
)


def test_innovation_rejects_long_gap_and_excludes_current() -> None:
    times = pd.to_datetime(
        [
            "2023-06-01 01:00:00Z",
            "2023-06-02 01:00:00Z",
            "2023-06-03 01:00:00Z",
            "2023-06-10 01:00:00Z",
            "2023-06-11 01:00:00Z",
        ]
    )
    surface = pd.DataFrame({"snapshot_time": times, "skew": [0.0, 1.0, 3.0, 100.0, 104.0]})
    result = add_causal_innovation_zscore(
        surface,
        lookback=2,
        maximum_gap_days=3,
        maximum_history_span_days=20,
    )
    assert np.isnan(result.loc[3, "skew_innovation"])
    expected = (4.0 - 1.5) / np.std([1.0, 2.0], ddof=0)
    assert np.isclose(result.loc[4, "innovation_zscore"], expected)

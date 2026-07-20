from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v170_option_skew_directional_btc import (
    add_causal_skew_zscore,
    select_25_delta_skew,
)


def test_select_25_delta_skew_uses_nearest_deltas() -> None:
    rows = []
    for option_type, deltas in (("C", [0.20, 0.26, 0.35]), ("P", [-0.20, -0.24, -0.35])):
        for index, delta in enumerate(deltas):
            rows.append(
                {
                    "snapshot_time": pd.Timestamp("2023-07-01 01:00:00Z"),
                    "expiration_time": pd.Timestamp("2023-07-31 08:00:00Z"),
                    "strike_price": 29_000 + index * 1_000,
                    "option_type": option_type,
                    "symbol": f"BTC-X-{index}-{option_type}",
                    "best_bid_price": 100,
                    "best_ask_price": 110,
                    "best_bid_qty": 2,
                    "best_ask_qty": 2,
                    "mark_iv": 0.50 + index * 0.01 + (0.05 if option_type == "P" else 0),
                    "delta": delta,
                }
            )
    selected = select_25_delta_skew(pd.DataFrame(rows))
    assert selected is not None
    assert np.isclose(selected["call_delta"], 0.26)
    assert np.isclose(selected["put_delta"], -0.24)


def test_causal_skew_zscore_excludes_current_value() -> None:
    times = pd.date_range("2023-06-01 01:00:00Z", periods=4, freq="D")
    surface = pd.DataFrame({"snapshot_time": times, "skew": [0.0, 1.0, 2.0, 100.0]})
    result = add_causal_skew_zscore(surface, lookback=3, maximum_span_days=5)
    expected = (100.0 - 1.0) / np.std([0.0, 1.0, 2.0], ddof=0)
    assert result["skew_zscore"].iloc[:3].isna().all()
    assert np.isclose(result.loc[3, "skew_zscore"], expected)

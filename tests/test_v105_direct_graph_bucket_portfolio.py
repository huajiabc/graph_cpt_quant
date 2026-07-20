import pandas as pd

from pressure_graph.reports.v105_direct_graph_bucket_portfolio import (
    CANDIDATES,
    V105Config,
    add_v105_states,
    build_v105_portfolios,
)


def test_v105_trades_neighbor_bucket_in_both_directions() -> None:
    times = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 3,
            "feature_time": times,
            "period": ["validation"] * 3,
            "entry_day": ["2026-01-01"] * 3,
            "entry_month": ["2026-01"] * 3,
            "bucket_ret_1h": [0.0, 0.01, 0.0],
            "bucket_ret_1h_rank": [0.5, 0.9, 0.5],
            "bucket_positive_breadth_1h": [0.5, 0.8, 0.5],
            "bucket_excess_ret_1h": [0.0, 0.005, 0.0],
            "bucket_ret_15m": [0.0, -0.001, 0.0],
            "bucket_future_ret_4h": [0.0, 0.012, 0.0],
        }
    )
    portfolios, _ = build_v105_portfolios(
        add_v105_states(frame), V105Config(max_buckets=1)
    )
    continuation = portfolios[portfolios["candidate"].eq(CANDIDATES[0])].iloc[0]
    reversal = portfolios[portfolios["candidate"].eq(CANDIDATES[1])].iloc[0]
    assert continuation["gross_4h"] == 0.012
    assert reversal["gross_4h"] == -0.012
    assert continuation["net_4h_20bp"] == 0.01

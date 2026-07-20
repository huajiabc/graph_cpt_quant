import pandas as pd

from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    simulate_v234_oco,
)


def test_v234_upper_breakout_enters_long_and_exits_at_four_hours() -> None:
    entry = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    features = pd.DataFrame(
        [
            {
                "entry_time": entry,
                "entry_spot": 100.0,
                "causal_hourly_sigma": 0.01,
            }
        ]
    )
    times = pd.date_range(entry, periods=16, freq="15min")
    bars = pd.DataFrame(
        {
            "bar_open_time": times,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
        }
    )
    bars.loc[1:, "high"] = 102.0
    bars.loc[15, "close"] = 103.0
    outcome = simulate_v234_oco(features, bars, V234Config())
    assert outcome.iloc[0]["triggered"]
    assert outcome.iloc[0]["trade_direction"] == 1
    assert outcome.iloc[0]["primary_net_return"] > 0

from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v174_deribit_skew_receiver_oco import (
    simulate_receiver_oco_leg,
)


def _bars(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(highs), freq="h", tz="UTC")
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=index)


def test_oco_uses_first_unique_break_and_fixed_exit() -> None:
    bars = _bars(
        [100, 101, 102, 103, 104, 105, 106, 107, 107, 107, 107, 107, *([110] * 19)],
        [95, 96, 97, 98, 99, 100, 101, 102, 102, 102, 102, 102, *([103] * 19)],
        [98, 99, 100, 101, 102, 103, 104, 105, 105, 105, 105, 105, *([109] * 19)],
    )
    signal_time = bars.index[5]
    result = simulate_receiver_oco_leg(bars, signal_time)
    assert result["filled"] is True
    assert result["side"] == "long"
    assert result["entry_price"] == 105
    assert result["exit_time"] == signal_time + pd.Timedelta(hours=24)


def test_oco_rejects_same_hour_dual_touch() -> None:
    highs = [100, 101, 102, 103, 104, 105, 106, *([106] * 24)]
    lows = [95, 96, 97, 98, 99, 100, 94, *([94] * 24)]
    closes = [98, 99, 100, 101, 102, 103, 100, *([100] * 24)]
    bars = _bars(highs, lows, closes)
    result = simulate_receiver_oco_leg(bars, bars.index[5])
    assert result["filled"] is False
    assert result["ambiguous"] is True
    assert result["status"] == "same_bar_dual_trigger"

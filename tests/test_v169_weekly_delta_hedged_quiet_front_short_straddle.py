from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v169_weekly_delta_hedged_quiet_front_short_straddle import (
    _apply_nonoverlap,
    calculate_weekly_straddle_trade,
)


def _pair(day: int, call_bid: float, call_ask: float, put_bid: float, put_ask: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "option_type": ["C", "P"],
            "best_bid_price": [call_bid, put_bid],
            "best_ask_price": [call_ask, put_ask],
            "delta": [0.55 + day * 0.01, -0.45 + day * 0.01],
        }
    )


def test_weekly_trade_uses_daily_delta_path_and_costs() -> None:
    path = [_pair(day, 100, 110, 90, 100) for day in range(8)]
    path[-1] = _pair(7, 80, 90, 70, 80)
    prices = [30_000 + day * 100 for day in range(8)]
    result = calculate_weekly_straddle_trade(path, prices)
    expected_short_hedge = sum(
        (0.10 + day * 0.02) * 100 for day in range(7)
    )
    assert np.isclose(result["short_option_pnl"], 20.0)
    assert np.isclose(result["short_hedge_pnl"], expected_short_hedge)
    assert np.isclose(result["long_hedge_pnl"], -expected_short_hedge)
    assert result["short_primary_net_return"] < result["short_gross_return"]
    assert result["short_stress_net_return"] < result["short_primary_net_return"]


def test_nonoverlap_accepts_next_entry_at_prior_exit() -> None:
    times = pd.date_range("2023-07-01 01:00:00Z", periods=15, freq="D")
    frame = pd.DataFrame(
        {
            "entry_time": times,
            "exit_time": times + pd.Timedelta(days=7),
        }
    )
    selected = _apply_nonoverlap(frame, pd.Series(True, index=frame.index))
    assert selected["entry_time"].tolist() == [times[0], times[7], times[14]]

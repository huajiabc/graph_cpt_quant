from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.v100_exact_taker_flow_alpha import (
    _extract_event,
    _price_at_or_after,
    add_v100_states,
)


def _minute_frame() -> pd.DataFrame:
    times = pd.date_range("2026-04-01 00:00Z", periods=300, freq="1min")
    frame = pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": "BTCUSDT",
            "bar_open_time": times,
            "open": np.linspace(100.0, 103.0, len(times)),
            "high": np.linspace(100.1, 103.1, len(times)),
            "low": np.linspace(99.9, 102.9, len(times)),
            "close": np.linspace(100.05, 103.05, len(times)),
            "buy_turnover": 60.0,
            "sell_turnover": 40.0,
            "turnover": 100.0,
            "trade_count": 10,
        }
    )
    return frame


def test_extract_event_excludes_signal_minute_from_flow_features() -> None:
    frame = _minute_frame()
    signal = pd.Timestamp("2026-04-01 00:15Z")
    frame.loc[frame["bar_open_time"].eq(signal), "buy_turnover"] = 100_000.0
    event = pd.Series(
        {
            "event_id": "event-1",
            "exchange": "bybit",
            "symbol": "BTCUSDT",
            "path_name": "short_squeeze",
            "signal_time": signal,
            "period": "development",
        }
    )

    row = _extract_event(event, {"BTCUSDT": frame})

    assert row is not None
    assert row["imbalance_5m"] == pytest.approx(0.20)
    assert row["turnover_5m"] == pytest.approx(500.0)
    assert row["entry_price"] == pytest.approx(
        frame.loc[frame["bar_open_time"].eq(signal), "open"].iloc[0]
    )


def test_flow_state_rules_are_frozen_and_distinct() -> None:
    panel = pd.DataFrame(
        {
            "imbalance_5m": [0.20, -0.20, 0.20],
            "imbalance_15m": [0.05, -0.05, 0.05],
            "return_5m": [0.01, 0.001, -0.001],
            "turnover_acceleration": [1.20, 0.80, 1.20],
        }
    )

    states = add_v100_states(panel)

    assert states["OF1_CONFIRM_LONG"].tolist() == [True, False, False]
    assert states["OF2_SELL_ABSORPTION_LONG"].tolist() == [False, True, False]
    assert states["OF3_AVOID_BUY_EXHAUSTION"].tolist() == [True, True, False]


def test_price_lookup_honors_forward_tolerance() -> None:
    frame = _minute_frame().drop(index=[15]).reset_index(drop=True)

    assert _price_at_or_after(frame, pd.Timestamp("2026-04-01 00:15Z")) == pytest.approx(
        frame.loc[frame["bar_open_time"].eq(pd.Timestamp("2026-04-01 00:16Z")), "open"].iloc[0]
    )
    assert np.isnan(
        _price_at_or_after(
            frame,
            pd.Timestamp("2026-04-01 05:00Z"),
            tolerance_minutes=2,
        )
    )

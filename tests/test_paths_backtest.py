from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.backtest import simulate_trades
from pressure_graph.config.models import ExecutionRule
from pressure_graph.paths import add_event_columns, apply_crowded_veto


def test_event_dedup_respects_cooldown() -> None:
    df = pd.DataFrame(
        {
            "exchange": ["bybit"] * 17,
            "symbol": ["ETHUSDT"] * 17,
            "bar_open_time": pd.date_range("2025-01-01", periods=17, freq="15min", tz="UTC"),
            "short_squeeze_signal": [True, True, True, *([False] * 13), True],
        }
    )
    out = add_event_columns(df, ["short_squeeze_signal"], cooldown_bars=16)
    assert out["short_squeeze_signal_event"].tolist() == [
        True,
        *([False] * 15),
        True,
    ]


def test_crowded_long_veto_blocks_long_signals() -> None:
    df = pd.DataFrame(
        {
            "short_squeeze_signal_raw": [True, True],
            "momentum_ignition_signal_raw": [True, False],
            "crowded_long_risk": [True, False],
        }
    )
    out = apply_crowded_veto(df)
    assert out["short_squeeze_signal"].tolist() == [False, True]
    assert out["momentum_ignition_signal"].tolist() == [False, False]


def test_same_bar_tp_sl_defaults_to_sl_first_and_cost_is_round_trip() -> None:
    df = pd.DataFrame(
        {
            "exchange": ["bybit", "bybit"],
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "bar_open_time": pd.to_datetime(["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z"]),
            "bar_close_time": pd.to_datetime(["2025-01-01 00:15:00Z", "2025-01-01 00:30:00Z"]),
            "feature_time": pd.to_datetime(["2025-01-01 00:15:00Z", "2025-01-01 00:30:00Z"]),
            "open": [100, 100],
            "high": [100, 104],
            "low": [100, 97],
            "close": [100, 101],
            "funding_time": [pd.NaT, pd.NaT],
            "funding_rate_settled": [0.0, 0.0],
            "short_squeeze_signal_event": [True, False],
        }
    )
    trades = simulate_trades(
        df,
        "short_squeeze_signal_event",
        "short_squeeze",
        ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=1),
        cost_single_side_bps=10,
        ambiguity_policy="sl_first",
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "sl_ambiguous"
    assert trade["gross_return"] == pytest.approx(-0.02)
    assert round(trade["net_return_ex_fee_slippage"], 6) == -0.022

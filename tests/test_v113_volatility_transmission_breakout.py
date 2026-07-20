from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v113_volatility_transmission_breakout import (
    V113Config,
    _forward_rv,
    build_v113_month_edges,
    simulate_v113_oco_leg,
)


def test_forward_rv_uses_only_bars_after_signal() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    values = pd.DataFrame({"A": np.arange(1.0, 9.0)}, index=index)

    result = _forward_rv(values, 2)

    assert result.loc[index[0], "A"] == np.sqrt(2.0**2 + 3.0**2)
    assert result.loc[index[5], "A"] == np.sqrt(7.0**2 + 8.0**2)
    assert np.isnan(result.loc[index[6], "A"])


def test_directed_absolute_shock_graph_recovers_planted_leader() -> None:
    rng = np.random.default_rng(7)
    rows = 1500
    leader = np.abs(rng.normal(size=rows))
    follower = np.r_[0.0, leader[:-1]] + np.abs(rng.normal(scale=0.03, size=rows))
    history = pd.DataFrame(
        {
            "LEADER": leader,
            "FOLLOWER": follower,
            "NOISE1": np.abs(rng.normal(size=rows)),
            "NOISE2": np.abs(rng.normal(size=rows)),
        }
    )
    cfg = V113Config(
        lags=(1,),
        min_edge_samples=1000,
        leaders_per_follower=1,
    )

    edges = build_v113_month_edges(
        history, pd.Timestamp("2026-02-01", tz="UTC"), cfg
    )
    planted = edges[
        edges["follower_symbol"].eq("FOLLOWER")
        & edges["leader_symbol"].eq("LEADER")
    ]

    assert len(planted) == 1
    assert planted.iloc[0]["lag_bars"] == 1
    assert planted.iloc[0]["direction_advantage"] > 0.8


def _ohlc_for_oco() -> tuple[pd.DataFrame, pd.Timestamp]:
    index = pd.date_range(
        "2026-01-01 00:15", periods=24, freq="15min", tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
        },
        index=index,
    )
    signal = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    frame.loc[:signal, "high"] = [100.2, 100.4, 100.8, 101.0]
    frame.loc[:signal, "low"] = [99.8, 99.6, 99.4, 99.2]
    frame.loc[signal + pd.Timedelta(minutes=15), ["high", "low"]] = [101.2, 99.5]
    frame.loc[signal + pd.Timedelta(hours=4), "close"] = 104.0
    return frame, signal


def test_oco_enters_first_long_break_and_exits_at_fixed_horizon() -> None:
    frame, signal = _ohlc_for_oco()

    result = simulate_v113_oco_leg(frame, signal, V113Config())

    assert result["filled"] is True
    assert result["ambiguous"] is False
    assert result["side"] == "long"
    assert result["entry_price"] == 101.0
    assert result["exit_time"] == signal + pd.Timedelta(hours=4)
    assert np.isclose(result["gross_return"], 104.0 / 101.0 - 1.0)


def test_oco_rejects_same_bar_dual_trigger() -> None:
    frame, signal = _ohlc_for_oco()
    trigger = signal + pd.Timedelta(minutes=15)
    frame.loc[trigger, ["high", "low"]] = [101.2, 99.0]

    result = simulate_v113_oco_leg(frame, signal, V113Config())

    assert result["filled"] is False
    assert result["ambiguous"] is True
    assert result["reason"] == "same_bar_dual_trigger"


def test_oco_short_return_uses_futures_entry_not_inverse_price() -> None:
    frame, signal = _ohlc_for_oco()
    trigger = signal + pd.Timedelta(minutes=15)
    frame.loc[trigger, ["high", "low"]] = [100.9, 99.0]
    frame.loc[signal + pd.Timedelta(hours=4), "close"] = 95.0

    result = simulate_v113_oco_leg(frame, signal, V113Config())

    assert result["side"] == "short"
    assert result["entry_price"] == 99.2
    assert np.isclose(result["gross_return"], 1.0 - 95.0 / 99.2)

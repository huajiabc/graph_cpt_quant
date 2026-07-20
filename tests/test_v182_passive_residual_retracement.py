from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v182_passive_residual_retracement import (
    V182Config,
    build_v182_events,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    index = pd.date_range(source, periods=7, freq="15min", tz="UTC")
    laggards = [f"L{i}USDT" for i in range(5)]
    leaders = [f"H{i}USDT" for i in range(5)]
    names = ["BTCUSDT", *laggards, *leaders]
    close = pd.DataFrame(100.0, index=index, columns=names)
    high = close.copy()
    low = close.copy()
    low.loc[index[1], laggards] = 99.8
    high.loc[index[1], leaders] = 100.2
    graph = pd.DataFrame(
        {
            "graph_month": source,
            "receiver": [*laggards, *leaders],
            "btc_beta": 0.0,
        }
    )
    signals = pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "graph_month": [source],
            "laggards": ["|".join(laggards)],
            "leaders": ["|".join(leaders)],
            "spread_beta": [0.0],
        }
    )
    return signals, graph, close, high, low


def test_all_touched_limits_fill_at_fixed_price_and_retain_pnl() -> None:
    signals, graph, close, high, low = _fixture()
    events, legs = build_v182_events(signals, graph, close, high, low)
    assert events.loc[0, "filled_count"] == 10
    assert legs["filled"].all()
    assert math.isclose(events.loc[0, "filled_alt_allocation"], 1.0)
    assert math.isclose(events.loc[0, "primary_cost_return"], 0.001)
    assert events.loc[0, "gross_return"] > 0


def test_unbalanced_partial_fill_is_not_discarded() -> None:
    signals, graph, close, high, low = _fixture()
    laggards = signals.loc[0, "laggards"].split("|")
    low.loc[:, laggards] = 100.0
    low.loc[low.index[1], laggards[0]] = 99.8
    high.loc[:, :] = 100.0
    events, _ = build_v182_events(signals, graph, close, high, low)
    assert events.loc[0, "laggard_fills"] == 1
    assert events.loc[0, "leader_fills"] == 0
    assert bool(events.loc[0, "traded"])
    assert events.loc[0, "filled_alt_allocation"] > 0


def test_delayed_order_moves_reference_fill_and_exit_times() -> None:
    signals, graph, close, high, low = _fixture()
    events, _ = build_v182_events(
        signals,
        graph,
        close,
        high,
        low,
        V182Config(),
        order_delay_bars=1,
    )
    source = signals.loc[0, "source_feature_time"]
    assert events.loc[0, "reference_time"] == source + pd.Timedelta(minutes=15)
    assert events.loc[0, "fill_time"] == source + pd.Timedelta(minutes=30)
    assert events.loc[0, "exit_time"] == source + pd.Timedelta(minutes=90)

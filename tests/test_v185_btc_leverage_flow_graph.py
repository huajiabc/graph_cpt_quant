from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BUILD,
    BUILD_CANDIDATE,
    UNWIND,
    V185Config,
    build_monthly_v185_graph,
    build_v185_events,
)


def test_monthly_graph_finds_engineered_signed_forward_edges() -> None:
    rng = np.random.default_rng(185)
    index = pd.date_range("2025-12-01", periods=2_200, freq="15min", tz="UTC")
    source = pd.Series(rng.normal(size=len(index)), index=index)
    returns = pd.DataFrame({"BTCUSDT": rng.normal(scale=0.001, size=len(index))}, index=index)
    impulses = {
        BUILD: pd.DataFrame({"BTCUSDT": source}, index=index),
        UNWIND: pd.DataFrame({"BTCUSDT": source}, index=index),
    }
    for position in range(12):
        name = f"A{position}USDT"
        sign = 1 if position % 2 == 0 else -1
        returns[name] = sign * source.shift(1) + rng.normal(scale=0.05, size=len(index))
        impulses[BUILD][name] = rng.normal(size=len(index))
        impulses[UNWIND][name] = rng.normal(size=len(index))
    cfg = V185Config(graph_min_samples=2_000)
    graph = build_monthly_v185_graph(
        returns,
        impulses,
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
        cfg,
    )
    selected = graph[graph["kind"].eq(BUILD) & graph["selected"]]
    assert len(selected) == 8
    assert selected["direction_advantage"].gt(0).all()
    expected_sign = selected["receiver"].str.extract(r"A(\d+)")[0].astype(int) % 2
    expected_sign = expected_sign.map({0: 1.0, 1: -1.0})
    assert selected["edge_sign"].tolist() == expected_sign.tolist()


def test_event_uses_frozen_edge_signs_and_beta_hedge() -> None:
    source = pd.Timestamp("2026-01-15", tz="UTC")
    exit_time = source + pd.Timedelta(minutes=30)
    receivers = [f"A{i}USDT" for i in range(5)]
    close = pd.DataFrame(
        100.0,
        index=pd.DatetimeIndex([source, exit_time]),
        columns=["BTCUSDT", *receivers],
    )
    close.loc[exit_time, "BTCUSDT"] = 101.0
    close.loc[exit_time, receivers] = [102.0, 98.0, 102.0, 98.0, 102.0]
    graph = pd.DataFrame(
        {
            "graph_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "kind": BUILD,
            "receiver": receivers,
            "btc_beta": 0.5,
            "edge_sign": [1.0, -1.0, 1.0, -1.0, 1.0],
            "selected": True,
            "receiver_rank": range(1, 6),
        }
    )
    signals = pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "kind": [BUILD],
            "candidate": [BUILD_CANDIDATE],
            "source_sign": [1.0],
        }
    )
    events = build_v185_events(
        signals, graph, close, V185Config(min_receiver_bucket=5)
    )
    alt_weights = np.array([0.2, -0.2, 0.2, -0.2, 0.2])
    hedge = -float((alt_weights * 0.5).sum())
    expected = (float((alt_weights * np.array([0.02, -0.02, 0.02, -0.02, 0.02])).sum()) + hedge * 0.01) / (1 + abs(hedge))
    assert math.isclose(events.loc[0, "btc_hedge_weight"], hedge)
    assert math.isclose(events.loc[0, "gross_return"], expected)

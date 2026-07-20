import numpy as np
import pandas as pd
import pytest

from pressure_graph.reports.v143_quiet_receiver_convergence import (
    V143Config,
    _future_node_volatility,
    build_v143_month_edges,
    prepare_v143_future_returns,
    source_release_signal,
)


def test_v143_future_volatility_uses_next_four_hours() -> None:
    values = pd.DataFrame({"A": np.arange(8, dtype=float)})
    future = _future_node_volatility(values)
    assert future.loc[0, "A"] == 2.5
    assert future.loc[3, "A"] == 5.5


def test_v143_edge_recovers_planted_volatility_receiver() -> None:
    rng = np.random.default_rng(143)
    leader = rng.normal(0, 1, 220)
    source = pd.DataFrame(
        {
            "A": leader,
            "B": rng.normal(0, 1, 220),
            "C": rng.normal(0, 1, 220),
            "D": rng.normal(0, 1, 220),
        }
    )
    target = pd.DataFrame(
        {
            "A": rng.normal(0, 1, 220),
            "B": leader + rng.normal(0, 0.05, 220),
            "C": rng.normal(0, 1, 220),
            "D": rng.normal(0, 1, 220),
        }
    )
    edges = build_v143_month_edges(
        source,
        target,
        pd.Timestamp("2026-01-01", tz="UTC"),
        V143Config(min_edge_samples=150),
    )
    planted = edges[
        edges["leader_community"].eq("A")
        & edges["follower_community"].eq("B")
    ]
    assert len(planted) == 1
    assert float(planted.iloc[0]["magnitude_advantage"]) > 0.8


def test_v143_release_requires_return_volatility_and_breadth() -> None:
    time = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    context = {
        "node_return_z": pd.DataFrame({"A": [3.0, 3.0]}, index=time),
        "node_volatility_z": pd.DataFrame({"A": [2.0, 0.0]}, index=time),
        "node_breadth": pd.DataFrame({"A": [0.8, 0.8]}, index=time),
        "node_member_count": pd.DataFrame({"A": [6, 6]}, index=time),
    }
    signal = source_release_signal(context, V143Config())
    assert signal.loc[time[0], "A"] == 1.0
    assert signal.loc[time[1], "A"] == 0.0


def test_v143_residualizes_before_dropping_btc() -> None:
    raw = pd.DataFrame({"BTCUSDT": [0.10], "ALTUSDT": [0.15]})
    naked, residual = prepare_v143_future_returns(
        raw, pd.Series({"BTCUSDT": 1.0, "ALTUSDT": 1.2}), ["ALTUSDT"]
    )
    assert naked.loc[0, "ALTUSDT"] == 0.15
    assert residual.loc[0, "ALTUSDT"] == pytest.approx(0.03)

import numpy as np
import pandas as pd

from pressure_graph.reports.v141_directed_taker_flow_graph import (
    CANDIDATES,
    V141Config,
    build_v141_month_edges,
    pressure_matrices,
    strict_15m_decision_time,
)


def test_v141_source_stamp_is_strictly_before_decision() -> None:
    stamps = pd.Series(
        pd.to_datetime(
            ["2026-01-01 00:10:00Z", "2026-01-01 00:15:00Z"], utc=True
        )
    )
    decisions = strict_15m_decision_time(stamps)
    assert decisions.iloc[0] == pd.Timestamp("2026-01-01 00:15:00Z")
    assert decisions.iloc[1] == pd.Timestamp("2026-01-01 00:30:00Z")
    assert bool((decisions > stamps).all())


def test_v141_edges_recover_planted_flow_leader() -> None:
    rng = np.random.default_rng(141)
    leader_flow = rng.normal(0, 1, 800)
    source = pd.DataFrame(
        {
            "LEADER": leader_flow,
            "FOLLOWER": rng.normal(0, 1, 800),
            "N1": rng.normal(0, 1, 800),
            "N2": rng.normal(0, 1, 800),
        }
    )
    target = pd.DataFrame(
        {
            "LEADER": rng.normal(0, 1, 800),
            "FOLLOWER": leader_flow + rng.normal(0, 0.1, 800),
            "N1": rng.normal(0, 1, 800),
            "N2": rng.normal(0, 1, 800),
        }
    )
    edges = build_v141_month_edges(
        source,
        target,
        pd.Timestamp("2026-01-01", tz="UTC"),
        V141Config(min_edge_samples=500),
    )
    planted = edges[
        edges["leader_symbol"].eq("LEADER")
        & edges["follower_symbol"].eq("FOLLOWER")
    ]
    assert len(planted) == 1
    assert float(planted.iloc[0]["direction_advantage"]) > 0.8


def test_v141_follower_requires_two_simultaneous_active_leaders() -> None:
    times = pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")
    symbols = ["L1", "L2", "F", "N"]
    flow = pd.DataFrame(0.0, index=times, columns=symbols)
    ret = pd.DataFrame(0.0, index=times, columns=symbols)
    flow.loc[times[0], "L1"] = 3.0
    flow.loc[times[1], ["L1", "L2"]] = 3.0
    ret.loc[:, ["L1", "L2"]] = 0.01
    context = {
        "symbols": symbols,
        "flow_z_15m": flow,
        "ret_15m": ret,
    }
    edges = pd.DataFrame(
        {
            "follower_symbol": ["F", "F"],
            "leader_symbol": ["L1", "L2"],
            "edge_weight": [1.0, 1.0],
            "edge_rank": [1, 2],
        }
    )
    result = pressure_matrices(context, edges, V141Config())
    positive = result[CANDIDATES[0]]
    assert positive["pressure"].loc[times[0], "F"] == 0.0
    assert positive["pressure"].loc[times[1], "F"] > 0.0
    assert positive["active_leaders"].loc[times[1], "F"] == 2


def test_v141_negative_flow_pressure_has_positive_magnitude() -> None:
    time = pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC")
    symbols = ["L1", "L2", "F", "N"]
    flow = pd.DataFrame(0.0, index=time, columns=symbols)
    ret = pd.DataFrame(0.0, index=time, columns=symbols)
    flow.loc[time[0], ["L1", "L2"]] = -3.0
    ret.loc[time[0], ["L1", "L2"]] = -0.01
    edges = pd.DataFrame(
        {
            "follower_symbol": ["F", "F"],
            "leader_symbol": ["L1", "L2"],
            "edge_weight": [1.0, 1.0],
            "edge_rank": [1, 2],
        }
    )
    result = pressure_matrices(
        {"symbols": symbols, "flow_z_15m": flow, "ret_15m": ret},
        edges,
        V141Config(),
    )
    assert result[CANDIDATES[1]]["pressure"].loc[time[0], "F"] == 1.0

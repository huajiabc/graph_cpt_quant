import numpy as np
import pandas as pd

from pressure_graph.reports.v142_community_volatility_transmission import (
    CANDIDATES,
    V142Config,
    build_v142_month_edges,
    causal_zscore,
    community_pressure_matrices,
)


def test_v142_zscore_uses_only_prior_values() -> None:
    values = pd.Series(np.arange(130, dtype=float))
    baseline = causal_zscore(values, lookback=120, min_periods=120)
    changed = values.copy()
    changed.iloc[-1] = 1_000_000.0
    revised = causal_zscore(changed, lookback=120, min_periods=120)
    assert baseline.iloc[-2] == revised.iloc[-2]
    expected = (changed.iloc[-1] - changed.iloc[-121:-1].mean()) / changed.iloc[
        -121:-1
    ].std(ddof=1)
    assert revised.iloc[-1] == min(expected, 5.0)


def test_v142_edges_recover_positive_and_negative_relations() -> None:
    rng = np.random.default_rng(142)
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
            "C": -leader + rng.normal(0, 0.05, 220),
            "D": rng.normal(0, 1, 220),
        }
    )
    edges = build_v142_month_edges(
        source,
        target,
        pd.Timestamp("2026-01-01", tz="UTC"),
        V142Config(min_edge_samples=150),
    )
    positive = edges[
        edges["leader_community"].eq("A")
        & edges["follower_community"].eq("B")
    ]
    negative = edges[
        edges["leader_community"].eq("A")
        & edges["follower_community"].eq("C")
    ]
    assert positive.iloc[0]["candidate"] == CANDIDATES[0]
    assert negative.iloc[0]["candidate"] == CANDIDATES[1]


def test_v142_continuation_and_reversal_pressures_have_opposite_sign() -> None:
    time = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    nodes = ["A", "B", "C", "D"]
    return_z = pd.DataFrame(0.0, index=time, columns=nodes)
    return_z.loc[time[0], "A"] = 3.0
    context = {
        "communities": nodes,
        "node_return_z": return_z,
        "node_volatility_z": pd.DataFrame(2.0, index=time, columns=nodes),
        "node_breadth": pd.DataFrame(0.8, index=time, columns=nodes),
        "node_member_count": pd.DataFrame(6, index=time, columns=nodes),
    }
    edges = pd.DataFrame(
        {
            "candidate": [CANDIDATES[0], CANDIDATES[1]],
            "leader_community": ["A", "A"],
            "follower_community": ["B", "C"],
            "relation_sign": [1.0, -1.0],
            "edge_weight": [1.0, 1.0],
            "edge_rank": [1, 1],
        }
    )
    pressure = community_pressure_matrices(context, edges, V142Config())
    assert pressure[CANDIDATES[0]].loc[time[0], "B"] == 1.0
    assert pressure[CANDIDATES[1]].loc[time[0], "C"] == -1.0

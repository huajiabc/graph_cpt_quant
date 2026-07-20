import numpy as np
import pandas as pd

from pressure_graph.reports.v108_oi_leader_bucket import (
    V108Config,
    build_v108_month_edges,
)


def test_v108_edges_recover_planted_oi_leader() -> None:
    rng = np.random.default_rng(19)
    leader_oi = rng.normal(0, 1, 800)
    target = leader_oi + rng.normal(0, 0.1, 800)
    source = pd.DataFrame(
        {
            "LEADER": leader_oi,
            "FOLLOWER": rng.normal(0, 1, 800),
            "N1": rng.normal(0, 1, 800),
            "N2": rng.normal(0, 1, 800),
        }
    )
    residual = pd.DataFrame(
        {
            "LEADER": rng.normal(0, 1, 800),
            "FOLLOWER": target,
            "N1": rng.normal(0, 1, 800),
            "N2": rng.normal(0, 1, 800),
        }
    )
    edges = build_v108_month_edges(
        source,
        residual,
        pd.Timestamp("2026-01-01", tz="UTC"),
        V108Config(min_edge_samples=500),
    )
    planted = edges[
        edges["leader_symbol"].eq("LEADER")
        & edges["follower_symbol"].eq("FOLLOWER")
    ]
    assert len(planted) == 1
    assert float(planted.iloc[0]["direction_advantage"]) > 0.8

import numpy as np
import pandas as pd

from pressure_graph.reports.v109_graph_dispersion_spread import (
    build_v109_communities,
)


def test_v109_mst_cut_recovers_two_correlated_groups() -> None:
    rng = np.random.default_rng(13)
    left = rng.normal(0, 1, 800)
    right = rng.normal(0, 1, 800)
    frame = pd.DataFrame(
        {
            "A": left + rng.normal(0, 0.05, 800),
            "B": left + rng.normal(0, 0.05, 800),
            "C": left + rng.normal(0, 0.05, 800),
            "D": right + rng.normal(0, 0.05, 800),
            "E": right + rng.normal(0, 0.05, 800),
            "F": right + rng.normal(0, 0.05, 800),
        }
    )
    communities, edges = build_v109_communities(
        frame, community_count=2, min_samples=500
    )
    assert {frozenset(values) for values in communities} == {
        frozenset({"A", "B", "C"}),
        frozenset({"D", "E", "F"}),
    }
    assert int(edges["tree_edge_retained"].sum()) == 4

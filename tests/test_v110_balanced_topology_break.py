import numpy as np
import pandas as pd

from pressure_graph.reports.v110_balanced_topology_break import (
    build_v110_communities,
)


def test_v110_balanced_spectral_split_recovers_planted_halves() -> None:
    rng = np.random.default_rng(21)
    left = rng.normal(0, 1, 800)
    right = rng.normal(0, 1, 800)
    frame = pd.DataFrame(
        {
            **{f"L{index}": left + rng.normal(0, 0.08, 800) for index in range(6)},
            **{f"R{index}": right + rng.normal(0, 0.08, 800) for index in range(6)},
        }
    )
    communities = build_v110_communities(frame, community_count=2, min_samples=500)
    assert sorted(len(group) for group in communities) == [6, 6]
    assert {frozenset(group) for group in communities} == {
        frozenset(f"L{index}" for index in range(6)),
        frozenset(f"R{index}" for index in range(6)),
    }

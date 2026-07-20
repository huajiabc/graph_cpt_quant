import numpy as np

from pressure_graph.reports.v202_community_peer_hedge_feature_audit import (
    build_peer_hedged_weights,
)


def test_peer_hedged_weights_are_dollar_neutral_with_equal_sleeves() -> None:
    weights = build_peer_hedged_weights(
        ["A", "B", "C"], ["D", "E"], source_sign=1.0
    )
    assert np.isclose(sum(weights.values()), 0.0)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert np.isclose(sum(abs(weights[name]) for name in ["A", "B", "C"]), 0.5)
    assert np.isclose(sum(abs(weights[name]) for name in ["D", "E"]), 0.5)
    assert all(weights[name] < 0 for name in ["A", "B", "C"])
    assert all(weights[name] > 0 for name in ["D", "E"])

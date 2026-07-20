import numpy as np
import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v197_rich_premium_short_feature_audit import (
    GLOBAL_CHEAP_LONG,
    GLOBAL_RICH_SHORT,
    V197FeatureConfig,
    build_v197_target,
)


def _local() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"A{i}" for i in range(10)],
            "community_id": ["C0"] * 5 + ["C1"] * 5,
            "peer_premium_z": np.arange(10.0),
            "btc_beta": np.linspace(0.5, 1.4, 10),
        }
    )


def test_v197_global_short_selects_richest_and_hedges_beta() -> None:
    cfg = V197FeatureConfig(
        global_bucket_size=3, minimum_global_cross_section=6
    )
    weights, selected = build_v197_target(_local(), GLOBAL_RICH_SHORT, cfg)
    assert selected == ["A7", "A8", "A9"]
    assert all(weights[symbol] < 0 for symbol in selected)
    assert weights[BTC] > 0
    beta = _local().set_index("symbol")["btc_beta"]
    residual = weights[BTC] + sum(
        weights[symbol] * beta[symbol] for symbol in selected
    )
    assert abs(residual) <= 1e-12
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)


def test_v197_global_long_selects_cheapest() -> None:
    cfg = V197FeatureConfig(
        global_bucket_size=3, minimum_global_cross_section=6
    )
    weights, selected = build_v197_target(_local(), GLOBAL_CHEAP_LONG, cfg)
    assert selected == ["A0", "A1", "A2"]
    assert all(weights[symbol] > 0 for symbol in selected)
    assert weights[BTC] < 0

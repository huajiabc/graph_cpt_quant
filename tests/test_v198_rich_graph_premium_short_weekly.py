import numpy as np
import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v198_rich_graph_premium_short_weekly import (
    V198Config,
    _orthogonal_short_target,
    _parse_weights,
)


def test_v198_parse_weights_round_trip() -> None:
    parsed = _parse_weights("A:-0.25|BTCUSDT:0.75")
    assert parsed == {"A": -0.25, "BTCUSDT": 0.75}


def test_v198_orthogonal_short_is_beta_neutral() -> None:
    local = pd.DataFrame(
        {
            "symbol": [f"A{i}" for i in range(10)],
            "community_id": ["C0"] * 5 + ["C1"] * 5,
            "funding_orthogonal_premium_z": np.arange(10.0),
            "btc_beta": np.linspace(0.5, 1.4, 10),
        }
    )
    cfg = V198Config(global_bucket_size=3, minimum_global_cross_section=6)
    weights = _orthogonal_short_target(local, "global", cfg)
    selected = [symbol for symbol in weights if symbol != BTC]
    assert set(selected) == {"A7", "A8", "A9"}
    assert all(weights[symbol] < 0 for symbol in selected)
    beta = local.set_index("symbol")["btc_beta"]
    residual = weights[BTC] + sum(
        weights[symbol] * beta[symbol] for symbol in selected
    )
    assert abs(residual) <= 1e-12

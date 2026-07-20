from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v165_fixed_core_satellite_fss3_tg1 import (
    V165Config,
    build_v165_portfolio,
)


def test_build_v165_portfolio_is_exact_fixed_linear_combination() -> None:
    sleeves = pd.DataFrame(
        {
            "fss3_price_return": [0.01, -0.02],
            "tg1_price_return": [0.03, 0.04],
            "fss3_funding_return": [0.001, 0.002],
            "tg1_funding_return": [0.003, -0.001],
            "fss3_primary_return": [0.02, -0.01],
            "tg1_primary_return": [0.01, 0.03],
            "fss3_stress_return": [0.015, -0.02],
            "tg1_stress_return": [0.005, 0.02],
        }
    )
    output = build_v165_portfolio(sleeves, V165Config())
    assert np.allclose(output["primary_net_return"], [0.018, -0.002])
    assert np.allclose(output["stress_net_return"], [0.013, -0.012])
    assert output["fss3_weight"].eq(0.8).all()
    assert output["tg1_weight"].eq(0.2).all()

from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import (
    V151Config,
    build_v151_portfolio,
    causal_tg1_weights,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tg1_primary_return": [0.01, -0.01] * 6,
            "fss3_primary_return": [0.04, -0.04] * 6,
            "tg1_stress_return": [0.009, -0.011] * 6,
            "fss3_stress_return": [0.039, -0.041] * 6,
            "tg1_price_return": np.zeros(12),
            "fss3_price_return": np.zeros(12),
            "tg1_funding_return": np.full(12, 0.002),
            "fss3_funding_return": np.full(12, 0.003),
        }
    )


def test_v151_first_eight_weeks_are_equal_weight() -> None:
    weights = causal_tg1_weights(_panel())
    assert np.allclose(weights[:8], 0.5)


def test_v151_inverse_volatility_uses_only_prior_eight_weeks() -> None:
    panel = _panel()
    weights = causal_tg1_weights(panel)
    tg1_vol = panel.iloc[:8]["tg1_primary_return"].std(ddof=1)
    fss3_vol = panel.iloc[:8]["fss3_primary_return"].std(ddof=1)
    expected = fss3_vol / (tg1_vol + fss3_vol)
    assert np.isclose(weights[8], 0.75)
    assert expected > 0.75


def test_v151_allocation_cost_is_charged_on_weight_change() -> None:
    cfg = V151Config()
    portfolio = build_v151_portfolio(_panel(), cfg)
    expected = cfg.primary_allocation_cost * portfolio["tg1_weight"].diff().abs().fillna(0)
    assert np.allclose(portfolio["primary_allocation_cost"], expected)

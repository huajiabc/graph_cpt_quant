from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v119_variance_risk_premium import (
    _annualized_rv,
    _normalized_payoff,
)


def test_annualized_rv_matches_constant_hourly_variance() -> None:
    hourly_vol = 0.01
    returns = pd.Series([hourly_vol] * (30 * 24))
    expected = np.sqrt((hourly_vol**2) * 30 * 24 * 365 / 30) * 100
    assert _annualized_rv(returns) == expected


def test_short_variance_payoff_positive_when_iv_exceeds_realized() -> None:
    assert _normalized_payoff(60.0, 40.0, 0.0) > 0
    assert _normalized_payoff(60.0, 40.0, 1.0) < _normalized_payoff(60.0, 40.0, 0.0)
    assert _normalized_payoff(40.0, 60.0, 0.0) < 0

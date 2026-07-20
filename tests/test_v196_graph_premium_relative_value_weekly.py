import numpy as np
import pandas as pd

from pressure_graph.reports.v196_graph_premium_relative_value_weekly import (
    _funding_interval_lookup,
    _portfolio_components,
)


def test_v196_funding_interval_is_entry_exclusive_exit_inclusive() -> None:
    entry = pd.Timestamp("2026-01-05", tz="UTC")
    exit_time = entry + pd.Timedelta(days=7)
    funding = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "funding_time": [entry, entry + pd.Timedelta(days=1), exit_time],
            "funding_rate_settled": [1.0, 2.0, 4.0],
        }
    )
    lookup = _funding_interval_lookup(funding, [(entry, exit_time)])
    assert lookup[(entry, exit_time, "A")] == 6.0


def test_v196_components_include_price_funding_and_beta_residual() -> None:
    local = pd.DataFrame(
        {"symbol": ["A", "B"], "btc_beta": [1.5, 0.5]}
    )
    weights = {"A": 0.25, "B": -0.25, "BTCUSDT": -0.25}
    price = pd.Series({"A": 0.04, "B": 0.01, "BTCUSDT": 0.02})
    funding = pd.Series({"A": -0.001, "B": 0.001, "BTCUSDT": 0.0001})
    result = _portfolio_components(local, weights, price, funding)
    expected_price = 0.25 * 0.04 - 0.25 * 0.01 - 0.25 * 0.02
    expected_funding = -0.25 * -0.001 - (-0.25) * 0.001 - (-0.25) * 0.0001
    expected_residual = 0.25 * (0.04 - 1.5 * 0.02) - 0.25 * (
        0.01 - 0.5 * 0.02
    ) + expected_funding
    assert np.isclose(result["price_return"], expected_price)
    assert np.isclose(result["funding_return"], expected_funding)
    assert np.isclose(result["residual_gross_return"], expected_residual)
